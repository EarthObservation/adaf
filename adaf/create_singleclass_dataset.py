import geopandas as gpd
import pandas as pd
from adaf.create_patches import uniform_grid, create_one_patch
from adaf.adaf_utils import safe_pool_size
from pathlib import Path
import multiprocessing as mp


# --- helpers ---------------------------------------------------------------

def _ensure_same_crs(gdf: gpd.GeoDataFrame, target_crs):
    if gdf is None or len(gdf) == 0:
        return gdf
    if gdf.crs is None:
        raise ValueError("Input GeoDataFrame has no CRS set.")
    if gdf.crs != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


def _union_of_polygons(gdf: gpd.GeoDataFrame):
    """Unary union of all geometries (returns shapely geometry)."""
    if gdf is None or len(gdf) == 0:
        return None
    # unary_union is faster than dissolve() for this use case
    return gdf.geometry.unary_union


def _grid_within_polygon(poly, crs, tile_size, offset=None):
    """
    Create a uniform grid over polygon bounds, then keep only tiles fully within polygon.
    Uses existing uniform_grid(bounds, crs, tile_size, offset).
    """
    if poly is None or poly.is_empty:
        return gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=crs)

    bounds = list(poly.bounds)  # [minx, miny, maxx, maxy]
    grid = uniform_grid(bounds, crs, tile_size, offset)

    # Keep only tiles fully within polygon (so we don't get cut tiles)
    mask = grid.geometry.within(poly)
    return grid.loc[mask].reset_index(drop=True)


def _filter_tiles_that_intersect_archaeology(grid: gpd.GeoDataFrame, archaeology: gpd.GeoDataFrame):
    """Keep tiles that intersect archaeology (positive-only tiles)."""
    if grid is None or len(grid) == 0:
        return grid
    if archaeology is None or len(archaeology) == 0:
        # no archaeology -> no positive tiles
        return grid.iloc[0:0].copy()

    arch_union = archaeology.geometry.unary_union
    mask = grid.geometry.intersects(arch_union)
    return grid.loc[mask].reset_index(drop=True)


def _add_tile_id(grid: gpd.GeoDataFrame):
    """Create a stable tile identifier from lower-left coords."""
    if grid is None or len(grid) == 0:
        return grid
    b = grid.bounds
    grid["tile_id"] = b[["minx", "miny"]].astype(int).astype(str).agg("_".join, axis=1)
    return grid


def _split_polygons(split_gdf: gpd.GeoDataFrame, split_field="split"):
    """Return (val_gdf, test_gdf) from split polygons."""
    if split_gdf is None or len(split_gdf) == 0:
        return (
            split_gdf.iloc[0:0].copy(),
            split_gdf.iloc[0:0].copy(),
        )

    if split_field not in split_gdf.columns:
        raise ValueError(f"split_gdf is missing required field '{split_field}'.")

    s = split_gdf[split_field].astype(str).str.lower()
    val = split_gdf.loc[s == "validation"].copy()
    test = split_gdf.loc[s == "test"].copy()
    return val, test


# --- main API --------------------------------------------------------------

def add_image_and_mask_paths(
    tiles_grid: gpd.GeoDataFrame,
    output_root: Path,
    ds_name: str,
):
    """
    Add image and segmentation mask paths to a tiles GeoDataFrame.

    Folder structure:
        output_root/
            train/
                images/
                segmentation_masks/
            validation/
                images/
                segmentation_masks/
            test/
                images/
                segmentation_masks/

    Added columns:
        - images
        - segmentation_masks

    Filenames follow:
        <tile_id>__<ds_name>__<type>.tif
    """

    output_root = Path(output_root)

    required_cols = {"split", "tile_id"}
    missing = required_cols - set(tiles_grid.columns)
    if missing:
        raise ValueError(f"gdf_tiles is missing required columns: {missing}")

    # Ensure splits are exactly what we expect
    valid_splits = {"training", "validation", "test"}
    if not set(tiles_grid["split"]).issubset(valid_splits):
        raise ValueError(
            f"Unexpected split values found: {set(tiles_grid['split']) - valid_splits}"
        )

    # Prepare output columns
    tiles_grid["images_path"] = (
        output_root.as_posix() + "/" +
        tiles_grid["split"] + "/images/" +
        tiles_grid["tile_id"] + "__" + ds_name + "__images.tif"
    )

    tiles_grid["segmentation_masks_path"] = (
        output_root.as_posix() + "/" +
        tiles_grid["split"] + "/segmentation_masks/" +
        tiles_grid["tile_id"] + "__" + ds_name + "__segmentation_masks.tif"
    )

    return tiles_grid


def build_singleclass_segmentation_grids(
    extents_gdf: gpd.GeoDataFrame,
    archaeology_gdf: gpd.GeoDataFrame,
    split_gdf: gpd.GeoDataFrame,
    *,
    class_name: str = "barrow",
    tile_size_px: int = 512,
    overlap_px: int = 256,
    pixel_size: float = 0.5,
    split_field: str = "split",
):
    """
    Build tile footprint grids for *single-class* semantic segmentation.

    Inputs
    ------
    extents_gdf:
        Polygon(s) describing where DEM/DFM exists (coverage / bounds).
    archaeology_gdf:
        Known archaeology polygons (optionally filtered to one class).
    split_gdf:
        Rectangles with attribute `split` = 'validation' or 'test'. Everything else is training.

    Behaviour
    ---------
    - Test: grid generated per test polygon; NOT filtered by archaeology (mimics inference).
    - Validation: grid generated over validation polygons; filtered to tiles that intersect archaeology.
    - Training: grid generated over extents minus (validation ∪ test); filtered to:
        * tiles fully within extents, and
        * tiles that intersect archaeology.

    Returns
    -------
    (train_gdf, val_gdf, test_gdf): GeoDataFrames with columns:
        - geometry
        - split  ('training'/'validation'/'test')
        - class  (class_name)
        - tile_id
    """
    if extents_gdf is None or len(extents_gdf) == 0:
        raise ValueError("extents_gdf must be a non-empty GeoDataFrame with geometries.")
    if archaeology_gdf is None:
        raise ValueError("archaeology_gdf must not be None.")
    if split_gdf is None:
        raise ValueError("split_gdf must not be None.")

    # Use extents CRS as the target CRS
    if extents_gdf.crs is None:
        raise ValueError("extents_gdf.crs is None. Please set it explicitly.")
    target_crs = extents_gdf.crs

    archaeology = _ensure_same_crs(archaeology_gdf, target_crs)
    splits = _ensure_same_crs(split_gdf, target_crs)

    # Tile geometry sizes in map units
    tile_size = (int(tile_size_px * pixel_size), int(tile_size_px * pixel_size))
    offset = (overlap_px * pixel_size) if overlap_px else None

    # Build unions
    extents_union = _union_of_polygons(extents_gdf)
    val_gdf, test_gdf = _split_polygons(splits, split_field=split_field)
    val_union = _union_of_polygons(val_gdf)
    test_union = _union_of_polygons(test_gdf)

    # ------------------------------------------------------------------ TEST
    # Per test polygon, full grid (no archaeology filtering)
    test_tiles_all = []
    for poly in test_gdf.geometry:
        g = _grid_within_polygon(poly, target_crs, tile_size, offset=offset)
        test_tiles_all.append(g)

    test_tiles = (
        gpd.GeoDataFrame(pd.concat(test_tiles_all, ignore_index=True), crs=target_crs)
        if len(test_tiles_all) else
        gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=target_crs)
    )
    if len(test_tiles) > 0:
        test_tiles["split"] = "test"
        test_tiles["class"] = class_name
        test_tiles = _add_tile_id(test_tiles)

    # -------------------------------------------------------------- VALIDATION
    # Full grid inside validation polygons, THEN keep only positive tiles
    val_tiles_all = []
    for poly in val_gdf.geometry:
        g = _grid_within_polygon(poly, target_crs, tile_size, offset=offset)
        val_tiles_all.append(g)

    val_tiles = (
        gpd.GeoDataFrame(pd.concat(val_tiles_all, ignore_index=True), crs=target_crs)
        if len(val_tiles_all) else
        gpd.GeoDataFrame({"geometry": []}, geometry="geometry", crs=target_crs)
    )
    val_tiles = _filter_tiles_that_intersect_archaeology(val_tiles, archaeology)
    if len(val_tiles) > 0:
        val_tiles["split"] = "validation"
        val_tiles["class"] = class_name
        val_tiles = _add_tile_id(val_tiles)

    # ----------------------------
    # TRAINING tiles (rest of extents, positive only)
    # ----------------------------
    exclusion = None
    if val_union is not None and test_union is not None:
        exclusion = val_union.union(test_union)
    elif val_union is not None:
        exclusion = val_union
    elif test_union is not None:
        exclusion = test_union

    train_area = extents_union
    if exclusion is not None and train_area is not None and not train_area.is_empty:
        train_area = train_area.difference(exclusion)

    train_tiles = _grid_within_polygon(train_area, target_crs, tile_size, offset=offset)

    # Ensure tiles are within extents (should already be true, but kept explicit)
    if extents_union is not None and len(train_tiles) > 0:
        train_tiles = train_tiles.loc[train_tiles.geometry.within(extents_union)].reset_index(drop=True)

    # Keep only positive tiles for training
    train_tiles = _filter_tiles_that_intersect_archaeology(train_tiles, archaeology)

    if len(train_tiles) > 0:
        train_tiles["split"] = "training"
        train_tiles["class"] = class_name
        train_tiles = _add_tile_id(train_tiles)

    # ----------------------------
    # Merge into ONE GeoDataFrame output
    # ----------------------------
    out = pd.concat([train_tiles, val_tiles, test_tiles], ignore_index=True)
    out = gpd.GeoDataFrame(out, geometry="geometry", crs=target_crs)

    # Optional: stable column order
    cols = [c for c in ["split", "class", "tile_id", "geometry"] if c in out.columns]
    other = [c for c in out.columns if c not in cols]
    out_gdf = out[cols + other]

    return out_gdf


def create_patches_from_tiles(
    gdf_tiles: gpd.GeoDataFrame,
    gdf_archaeology: gpd.GeoDataFrame,
    input_raster: Path,
    *,
    arch_type: str = "barrow",
    dfm_value: int = 1
):
    """
    Create image + segmentation mask patches for tiles in gdf_tiles.

    This wraps the existing create_one_patch(...) multiprocessing workflow.

    Parameters
    ----------
    gdf_tiles : geopandas.GeoDataFrame
        Tile footprints with required columns (at minimum): geometry, split, and
        path columns used by create_one_patch (e.g. 'images', 'segmentation_masks').
    gdf_archaeology : geopandas.GeoDataFrame
        Archaeology polygons. Will be ensured to contain:
          - arch_type column (default 'barrow')
          - DFM column (default 1)
    input_raster : str or pathlib.Path
        VRT or GeoTIFF of the calculated visualisation.
    arch_type : str
        Class name used in single-class workflow (default 'barrow').
    dfm_value : int
        DFM value to assign if missing (default 1).
        

    Returns
    -------
    list
        Results returned by create_one_patch for each tile (whatever that function returns).
    """

    # --- basic checks ---------------------------------------------------------
    if gdf_tiles is None or len(gdf_tiles) == 0:
        raise ValueError("gdf_tiles is empty. Nothing to create.")

    if gdf_archaeology is None:
        raise ValueError("gdf_archaeology must not be None.")

    # --- ensure required archaeology attributes ------------------------------
    gdf_arch = gdf_archaeology.copy()

    if "arch_type" not in gdf_arch.columns:
        gdf_arch["arch_type"] = arch_type

    if "DFM" not in gdf_arch.columns:
        gdf_arch["DFM"] = dfm_value

    # --- build args for multiprocessing --------------------------------------
    # (tile_dict, [df_segments], input_raster, od_label=is False, we do not need to create lblTxt!)
    input_process_list = [
        (tile_row.to_dict(), [gdf_arch], input_raster, False)
        for _, tile_row in gdf_tiles.iterrows()
    ]

    # --- multiprocessing ------------------------------------------------------
    n_proc = safe_pool_size()
    with mp.Pool(processes=n_proc) as pool:
        results = pool.starmap(create_one_patch, input_process_list)

    return results

