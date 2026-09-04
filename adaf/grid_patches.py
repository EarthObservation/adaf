from pathlib import Path

import rasterio
import geopandas as gpd
import pandas as pd

from adaf.create_patches import uniform_grid


def _patches_grid(ds, patch_size, overlap=None):
    """
    Create a GeoDataFrame of tiles (patches) over `ds`, filtered to areas
    where archaeology exists.

    Parameters
    ----------
    ds : str or Path
        File path of input raster (e.g. VRT / GeoTIFF).
    patch_size : int
        Patch size in pixels (square tiles).
    overlap : int or None
        Overlap (shift of secondary grid) in pixels, IN PIXELS.
        If None → no overlap (tiles are butt-jointed).

    Returns
    -------
    GeoDataFrame
        Grid of tiles with:
        - geometry
        - filestem (based on lower-left corner)
    """
    ds = Path(ds)

    # We only need raster resolution, crs and extents
    with rasterio.open(ds) as src:
        ds_crs = src.crs
        ds_res = src.res[0]
        ds_extents = list(src.bounds)

    # Convert to spatial units
    tile_size_m = patch_size * ds_res

    # Decide if we want grid to overlap
    if overlap is None:
        stagger_m = None
    else:
        stagger_m = overlap * ds_res

    # Build initial grid
    grid = uniform_grid(
        ds_extents,
        ds_crs,
        (tile_size_m, tile_size_m),
        stagger_m,
    )

    return grid


def _build_inner_outer(split_gdf, label, margin, join_style=2):
    """
    Buffer polygons +/- a margin value, return:
    (inner, outer) geometries.

    inner = geom.buffer(-margin)
    outer = geom.buffer(+margin)
    """
    subset = split_gdf[split_gdf["split"].str.lower() == label]
    if subset.empty:
        return None, None

    geom = subset.dissolve().geometry.unary_union

    outer = geom.buffer(margin, join_style=join_style)
    inner = geom.buffer(-margin, join_style=join_style)

    # Negative buffer can collapse small polygons; handle that
    if inner.is_empty:
        inner = None

    return inner, outer


def _assign_splits_to_grid(df_patches, split_gpkg, ds_crs, margin=10.0):
    """
    Assign each patch to train / val / test based on polygons in split_gpkg. Value is assigned to the 'split' column.

    Rules:
    - Validation: tile must be FULLY within the +margin (outer) buffer
      of validation polygons.
    - Test: tile must be FULLY within the +margin (outer) buffer
      of test polygons.
    - Train: tile must be FULLY outside the -margin (inner) buffers
      of BOTH validation and test polygons.
    - All other tiles (in-between, border-touching, ambiguous) are discarded.

    Parameters
    ----------
    df_patches : GeoDataFrame
        Grid of tiles with a 'geometry' column.
    split_gpkg : str or Path
        GPKG with polygons and attribute 'split' having values
        'validation' and/or 'test'.
    ds_crs : CRS
        CRS of the raster / tiles.
    margin : float
        Buffer distance in CRS units (meters) for inner/outer bounds.

    Returns
    -------
    GeoDataFrame
        df_patches with a new 'split' column ('train', 'val', 'test'),
        ambiguous tiles removed.
    """
    split_gdf = gpd.read_file(split_gpkg)

    # Reproject to raster CRS if needed
    if split_gdf.crs != ds_crs:
        split_gdf = split_gdf.to_crs(ds_crs)

    # Build inner/outer buffers for validation and test
    val_inner, val_outer = _build_inner_outer(split_gdf, "validation", margin)
    test_inner, test_outer = _build_inner_outer(split_gdf, "test", margin)

    def _tile_split(geom):
        # 1) Validation: fully within validation outer buffer
        if val_outer is not None and geom.within(val_outer):
            return "validation"

        # 2) Test: fully within test outer buffer
        if test_outer is not None and geom.within(test_outer):
            return "test"

        # 3) Training: fully outside *inner* bounds of BOTH val and test
        outside_val_inner = val_inner is None or not geom.intersects(val_inner)
        outside_test_inner = test_inner is None or not geom.intersects(test_inner)
        if outside_val_inner and outside_test_inner:
            return "training"

        # 4) Everything else (border cases, overlaps, between inner/outer) → discard
        return None

    df_patches["split"] = df_patches["geometry"].apply(_tile_split)

    # Drop tiles that are ambiguous / discarded
    df_patches = df_patches[df_patches["split"].notna()].reset_index(drop=True)
    return df_patches


def build_learning_dataset_grid(
    tile_size,
    input_raster,
    seg_masks_dict,
    split_gpkg,
    output_directory=None,
    save_gpkg=False,
    margin=10.0,
    overlap=None
):
    """
    Build a grid of training samples (tiles) for ML.

    - Reads archaeology vectors (up to N classes) and merges them.
    - Builds a grid of tiles over the input raster, filtered to archaeology.
    - Assigns each tile to train / val / test using split polygons.
    - Optionally adds standard paths for images / masks / labelTxt.
    - Optionally saves the grid as a GPKG.

    Parameters
    ----------
    tile_size : int
        Tile size in pixels (square patches).
    input_raster : str or Path
        Path to input raster (visualisation used for training).
    seg_masks_dict : dict
        Dictionary mapping archaeological class name to GPKG path, e.g.:
        {"barrow": "barrow.gpkg", "enclosure": "enclosure.gpkg"}.
        The key is written into 'arch_type'. If 'DFM' is missing, it is set to 1.
    split_gpkg : str or Path
        GPKG with polygons and attribute 'split' (values: 'validation', 'test';
        anything else is treated as training area context for the inner/outer logic).
    output_directory : str or Path or None, optional
        Root directory for output tiles. If provided, the GeoDataFrame will
        include columns:
          - images_path
          - segmentation_masks_path
          - labelTxt_path
        organized in subfolders: <output_directory>/<split>/<data_type>/.
    save_gpkg : bool, optional
        If True and output_directory is given, save the grid as
        "<output_directory>/tiles.gpkg".
    margin : float, optional
        Buffer distance (in CRS units) for building inner/outer split zones.
    overlap : int or None
        Overlap between tiles IN PIXELS (default: None)
        e.g. tile_size=512, overlap=256 = 50% overlap

    Returns
    -------
    GeoDataFrame
        Grid of tiles with:
         - geometry
         - filestem
         - split ('train', 'val', 'test')
         - optionally *_path columns if output_directory is provided.
    """
    input_raster = Path(input_raster)
    split_gdf = gpd.read_file(split_gpkg)

    # Read raster CRS (for reprojecting vectors)
    with rasterio.open(input_raster) as src:
        ds_crs = src.crs

    # --- Read and merge archaeology vectors ---
    df_list = []

    for arch_type, gpkg_path in seg_masks_dict.items():
        df = gpd.read_file(gpkg_path)

        # Reproject to raster CRS if needed
        if df.crs != ds_crs:
            df = df.to_crs(ds_crs)

        df["arch_type"] = arch_type

        # DFM value required by dataloader
        if "DFM" not in df.columns:
            df["DFM"] = 1

        df_list.append(df)

    df_segments = gpd.GeoDataFrame(pd.concat(df_list, ignore_index=True), crs=ds_crs)
    # print(f"Dataset contains {len(df_segments)} labels.")  # logging optional

    # --- Build grid of tiles over the raster and archaeology ---
    df_patches = _patches_grid(
        ds=input_raster,
        patch_size=tile_size,
        overlap=overlap,
    )

    # Keep only patches with archaeology and all from the "test" area
    if len(df_segments) > 0:
        # Find TEST AREAS and buffer them
        test_gdf = split_gdf[split_gdf["split"].str.lower() == "test"].copy()
        test_gdf["geometry"] = test_gdf.geometry.buffer(-margin)
        test_gdf = test_gdf[~test_gdf.geometry.is_empty]

        # Merge with archaeology
        merged = pd.concat([test_gdf,df_segments], ignore_index=True)
        merged = merged.dissolve().geometry.iloc[0]

        # Filter the patches
        grid_filter = df_patches["geometry"].intersects(merged)
        df_patches = df_patches[grid_filter].reset_index(drop=True)

    # --- Assign train / val / test based on split polygons ---
    df_splits = _assign_splits_to_grid(
        df_patches=df_patches,
        split_gpkg=split_gpkg,
        ds_crs=ds_crs,
        margin=margin,
    )

    # --- Optionally generate paths ---
    if output_directory is not None:
        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        ds_name = input_raster.stem

        # Create "filestem" for naming files (lower-left coordinates)
        df_splits["filestem"] = (
            df_splits.bounds[["minx", "miny"]]
            .astype(int)
            .astype(str)
            .agg("_".join, axis=1)
        )

        for idx, row in df_splits.iterrows():
            split = row["split"]       # 'train', 'val', or 'test'
            filestem = row["filestem"]

            for data_type in ["images", "segmentation_masks", "labelTxt"]:
                suff = ".txt" if data_type == "labelTxt" else ".tif"
                out_dir = output_directory / split / data_type
                out_path = out_dir / f"{filestem}__{ds_name}__{data_type}{suff}"
                df_splits.at[idx, f"{data_type}_path"] = out_path.as_posix()

    # --- Optionally save to disk ---
    if save_gpkg:
        gpkg_path = output_directory / "tiles.gpkg"
        df_splits.to_file(gpkg_path.as_posix(), driver="GPKG")

    return df_splits


if __name__ == "__main__":

    input_raster = r"r:\delovno\nejc\stone_visualisations\BiH_ALS_2025_DMO_05m_slrm4inference.vrt"
    split_pth = r"r:\ML podatki\ml_dataset_split_v2.gpkg"

    seg_masks_dict = {
        "barrow": r"r:\ML podatki\archaeology\gomile_2025-11-28.gpkg",
        # "enclosure": "...",
        # "ringfort": "...",
    }

    output_dir = r"r:\ML podatki\learning_samples\training_samples_BiH_v7_128px"

    df_grid = build_learning_dataset_grid(
        tile_size=128,
        input_raster=input_raster,
        seg_masks_dict=seg_masks_dict,
        split_gpkg=split_pth,
        output_directory=output_dir,
        save_gpkg=True,
        margin=10.0,
        overlap=64
    )
