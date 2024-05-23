"""
Create patches (training samples) for ML training from visualisations and labeled polygons.
Created on 26 May 2023
@author: Nejc Čož, ZRC SAZU, Novi trg 2, 1000 Ljubljana, Slovenia

Requires ADAF toolbox.
"""
import multiprocessing as mp
import warnings
from os import cpu_count
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import geometry_mask
from shapely.affinity import translate
from shapely.geometry import box

from adaf.adaf_utils import clip_tile

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


def create_one_patch(one_tile, segments_gdf, dem_pth):
    """Creates ML patch for one tile. Takes tile as dictionary of GeoSeries, vector files with segments and path to
    the source image. Creates a tile of image (raster), segmentation mask of the image (raster) and labelTxt (text file)
    and saves them to disk.

    Parameters
    ----------
    one_tile : dict
        Dictionary containing all the data for one tile from GeoDataFrame.
    segments_gdf : list
        A list of GeoDataFrames containing all labeled segments.
    dem_pth : str or pathlib.Path()
        Path to source image, from which the patches are made.

    """
    # ####################
    # 1 # Clip image tile
    # ####################
    out_image = Path(one_tile["images_path"])
    out_image.parent.mkdir(exist_ok=True)

    image_path = clip_tile(
        list(one_tile["geometry"].bounds),
        out_image,
        dem_pth,
        out_nodata=0
    )

    # ############################
    # 2 # Create segmentation mask
    # ############################
    out_seg = Path(one_tile["segmentation_masks_path"])
    out_seg.parent.mkdir(parents=True, exist_ok=True)

    _ = create_segmentation_mask(
        one_tile["geometry"],
        out_seg,
        image_path,
        segments_gdf,
    )

    # ####################
    # 3 # Create labelTxt
    # ####################
    label_pth = Path(one_tile["labelTxt_path"])
    label_pth.parent.mkdir(parents=True, exist_ok=True)

    with open(label_pth, "w") as dst:
        dst.write(one_tile["labelTxt"])


def uniform_grid(extents, crs, spacing_xy, stagger=None):
    """ Creates uniform grid over the total extents.

    Parameters
    ----------
    extents : list
        A list of extents [x_min, y_min, x_max, y_max]
    crs : rasterio.crs.CRS
        Projection as CRS instance.
    spacing_xy : (int, int)
        Width and height of grid tile in meters.
    stagger : float
        Distance to use for translating the cell for creation of staggered grid.
        Use None if grid should not be staggered.

    Returns
    -------
    gpd.GeoDataFrame
        Grid in GeoDataFrame format.
    """
    # total area for the grid [xmin, ymin, xmax, ymax]
    x_min, y_min, x_max, y_max = extents  # gdf.total_bounds
    tile_w, tile_h = spacing_xy

    # Target Aligned Pixels
    x_min = np.floor(x_min / tile_w) * tile_w
    # bottom = np.floor(extents.bottom / tile_h) * tile_h
    # right = np.ceil(extents.right / tile_w) * tile_w
    y_max = np.ceil(y_max / tile_h) * tile_h
    _, bottom, right, _ = extents  # ONLY TOP-LEFT NEEDS TO BE ROUNDED

    grid_cells = []
    for x0 in np.arange(x_min, x_max, tile_w):
        for y0 in np.arange(y_max, y_min, -tile_h):
            # bounds
            x1 = x0 + tile_w
            y1 = y0 - tile_h
            cell_1 = box(x0, y0, x1, y1)
            grid_cells.append(cell_1)
            if stagger:
                cell_2 = translate(cell_1, stagger, stagger)
                grid_cells.append(cell_2)

    # Generate GeoDataFrame
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)

    return grid


def relative_bounds(polygon, patch, res):
    """Calculates coordinates of a BBOX relative to the image array.

    Parameters
    ----------
    polygon : gpd.GeoSeries
        Segmentation polygon
    patch : gpd.GeoSeries
        Tile polygon (image extents)
    res
        Spatial resolution of image, needed to create array from image extents.

    Returns
    -------
    str
        A string containing relative coordinates in labelTxt format.
    """
    # Get bounds of the intersected segment
    seg_minx, seg_miny, seg_maxx, seg_maxy = polygon.bounds

    # Arrays are indexed from the top-left corner, so we need minx and maxy
    x0 = patch.bounds.minx.iloc[0]
    y0 = patch.bounds.maxy.iloc[0]
    max0 = int((patch.bounds.maxx.iloc[0] - patch.bounds.minx.iloc[0]) / res)

    # Construct bbox coordinates
    x1 = max(int((seg_minx - x0) / res), 0)
    x2 = min(int((seg_maxx - x0) / res), max0)
    y1 = min(int((y0 - seg_miny) / res), max0)
    y2 = max(int((y0 - seg_maxy) / res), 0)

    label_txt = f"{x1} {y2} {x2} {y2} {x2} {y1} {x1} {y1}"

    return label_txt


def prepare_labeltxt(one_patch, gdf, res):
    patch = gpd.GeoSeries(one_patch, crs=gdf.crs)
    bbox = patch.bounds.iloc[0]

    # Segments that fall into this patch (there can be more than 1 segments in one patch)
    patch_segments = gdf.cx[bbox.minx:bbox.maxx, bbox.miny:bbox.maxy].reset_index(drop=True)

    # # DISCARD SEGMENTS WITH LESS THAN 1/3 AREA ON THE PATCH
    # Calculate intersection of every segment with the patch extents
    patch_segments["geom_intersection"] = [segment.intersection(one_patch) for segment in patch_segments.geometry]
    # Calculate area of segments and fraction
    patch_segments["area1"] = patch_segments.area
    patch_segments["area2"] = [intersected.area for intersected in patch_segments["geom_intersection"]]
    # Only keep segments with more than 1/3 of the area over the patch
    patch_segments["area_fraction"] = patch_segments["area2"] / patch_segments["area1"]
    patch_segments = patch_segments.loc[patch_segments['area_fraction'] > 0.33].reset_index(drop=True)

    # Flag segments on the edge of a patch [1 ... is on the edge, 0 ... entire segment is present]
    patch_segments["edge_patch"] = np.where(patch_segments["area_fraction"] < 1, 1, 0)

    # Create labelTxt (use empty string if there is no segments in this patch)
    if len(patch_segments) > 0:
        # Prepare AABB with relative coordinates for labelTxt (Requires resolution!)
        patch_segments["relative_bounds"] = [
            relative_bounds(seg, patch, res) for seg in patch_segments["geom_intersection"]
        ]

        # If "DFM" attribute doesn't exist, set value to 1 everywhere
        if "DFM" not in patch_segments:
            patch_segments["DFM"] = 1

        # Other parameters for labelTxt
        patch_segments["labelTxt"] = (
                patch_segments[["relative_bounds", "arch_type"]].agg(' '.join, axis=1)
                + " " + patch_segments["DFM"].astype(str)
        )

        label_txt = "\n".join(patch_segments["labelTxt"].values.tolist())
    else:
        label_txt = ""

    return label_txt


def create_segmentation_mask(poly, file_path, image, segments_list):
    # Determine bounds of a patch
    bounds = poly.bounds

    # Determine bounds of a patch
    with rasterio.open(image) as src:
        out_image = src.read()
        out_profile = src.profile.copy()
        out_transform = src.transform

    # Prepare output array (all data set to 0)
    patch_shape = out_image.shape[1:]
    out_image = out_image.astype('uint8')
    out_image.fill(0)
    # The seg mask will always have 3 bands
    out_image = np.repeat(out_image, 3, axis=0)

    # Loop over all three classes (if less than 3, leave empty arrays)
    for i, seg in enumerate(segments_list):
        # Segments that fall into this patch (there can be more than 1 segments in one patch)
        patch_segments = seg.cx[bounds[0]:bounds[2], bounds[1]:bounds[3]].reset_index(drop=True)

        # # DISCARD SEGMENTS WITH LESS THAN 1/3 AREA ON THE PATCH
        # Calculate intersection of every segment with the patch extents
        patch_segments["geom_intersection"] = [segment.intersection(poly) for segment in patch_segments.geometry]
        # Calculate area of segments and fraction
        patch_segments["area1"] = patch_segments.area
        patch_segments["area2"] = [intersected.area for intersected in patch_segments["geom_intersection"]]
        # Only keep segments with more than 1/3 of the area over the patch
        patch_segments["area_fraction"] = patch_segments["area2"] / patch_segments["area1"]
        patch_segments = patch_segments.loc[patch_segments['area_fraction'] > 0.33].reset_index(drop=True)

        # If any polygons were found, add them to segmentation mask raster
        if len(patch_segments) > 0:
            # Write each polygon separately, to ensure correct DFM value is written to file
            for _, vector in patch_segments.iterrows():
                # Find pixels covered by polygon
                array = geometry_mask(
                    [(vector['geometry'], 1)],
                    patch_shape,
                    out_transform,
                    all_touched=True,
                    invert=True
                )
                # Populate pixels with DFM value
                out_image[i, array] = vector["DFM"]

    out_profile.update(
        dtype="uint8",
        count=3,
        driver="GTiff",
        compress="lzw",
        nodata=None
    )

    # !!! Don't use predictor 3 for lzw compression of uint8 array !!!
    with rasterio.open(file_path, "w", **out_profile, predictor=2) as dst:
        dst.write(out_image)

    return file_path


def patches_grid(ds, archaeology, patch_size, stagger):
    """Creates GeoDataFrame for patches with LabelTxt attribute.

    LabelTxt: <x1> <y1> <x2> <y2> <x3> <y3> <x4> <y4> <label> <DFM>

    Calls:
        - uniform_grid
        - prepare_labeltxt
            - relative_bounds

    Parameters
    ----------
    ds
    archaeology
    patch_size
    stagger

    Returns
    -------

    """
    with rasterio.open(ds) as src:
        ds_crs = src.crs
        ds_res = src.res[0]
        ds_extents = list(src.bounds)

    df_segments = archaeology

    if len(df_segments) > 0:
        # Convert pixels to meters
        tile_size = (patch_size * ds_res, patch_size * ds_res)
        if stagger:
            stagger = stagger * ds_res

        # Initial grid
        grid = uniform_grid(ds_extents, ds_crs, tile_size, stagger)

        # Keep only patches with archaeology
        grid_filter = grid["geometry"].intersects(df_segments.dissolve().geometry[0])
        grid = grid[grid_filter].reset_index(drop=True)

        # Prepare labelTxt for all tiles (empty string if there is less than 1/3 of object inside the patch)
        grid["labelTxt"] = ""

        grid["labelTxt"] = [prepare_labeltxt(one_tile, df_segments, ds_res) for one_tile in grid["geometry"]]

        # Filter (remove patches with empty labelTxt, i.e. containing less than 1/3 of a segment)
        grid = grid[grid["labelTxt"] != ""].reset_index(drop=True)

        # Create "filestem" that will be used for naming files, which is basically lower-left coordinates
        grid["filestem"] = grid.bounds[["minx", "miny"]].astype(int).astype(str).agg('_'.join, axis=1)

        return grid


def create_patches_main(input_raster, seg_masks_dict, output_directory):
    """Main routine for creating patches.

    Parameters
    ----------
    input_raster : str or pathlib.Path()
        Path to input raster.
    seg_masks_dict : dict
        Dictionary containing labels. E.g.: {"barrow": <path to barrow vectors file>}. At least one label, not more than
        three labels.
    output_directory : str or pathlib.Path()
        Path to directory where patches are saved.

    """
    # ##################################################################################################################
    # PROCESS INPUTS:
    input_raster = Path(input_raster)

    nr_processes = cpu_count() - 2

    output_directory = Path(output_directory)
    output_directory.mkdir(exist_ok=True)

    # ##################################################################################################################
    # PROCESSING STARTS HERE!

    # STEP 1: Join all DFs into one, add arch_type column containing label name of each polygon
    crs_df = None
    df_list = []
    for key, value in seg_masks_dict.items():
        df = gpd.read_file(value)
        df["arch_type"] = key
        df_list.append(df)
        crs_df = df.crs
    df_grid = gpd.GeoDataFrame(pd.concat(df_list, ignore_index=True), crs=crs_df)

    # STEP 2: Create grid for patches
    df_patches = patches_grid(
        input_raster,
        df_grid,
        patch_size=512,
        stagger=256
    )

    # STEP 3: Generate paths for saving patches
    for data_type in ["images", "segmentation_masks", "labelTxt"]:
        suff = ".txt" if data_type == "labelTxt" else ".tif"
        ds_name = input_raster.stem
        df_patches[f"{data_type}_path"] = [
            Path(output_directory / data_type / f"{x}__{ds_name}__{data_type}{suff}").as_posix()
            for x in df_patches["filestem"]
        ]

    # # Save GDF (for DEBUG)
    # df_patches.to_file("test-names.gpkg", driver="GPKG")

    # STEP 4: Multiprocessing run for create single patch
    input_process_list = []
    for _, in_tile in df_patches.iterrows():
        input_process_list.append(
            (
                in_tile.to_dict(),
                df_list,
                input_raster
            )
        )

    # print("Start multiproc")

    with mp.Pool(nr_processes) as p:
        _ = [p.apply_async(create_one_patch, r) for r in input_process_list]
        # for result in realist:
        #     pool_out = result.get()

    # # Single run (FOR DEBUG)
    # tiles_gpkg(*input_process_list[10])

    print("Finished creating patches")


if __name__ == "__main__":
    # Define paths to inputs and outputs
    input_image = r"../test_data/test_patches/my_DFM.vrt"
    output_dir = r"../test_data/training_samples"

    # The dictionary HAST TO BE!!! in this format - at least one label and max 3 labels (there is no check)
    # Key is name of label and value is path to vector file. Can use any label name, in example default names are used.
    # segmentation_masks = {
    #     "barrow": r"../test_data/test_patches/arch/barrow_segmentation_TM75.gpkg",
    #     "enclosure": r"../test_data/test_patches/arch/enclosure_segmentation_TM75.gpkg",
    #     "ringfort": r"../test_data/test_patches/arch/ringfort_segmentation_TM75.gpkg"
    # }
    segmentation_masks = {
        "barrow": r"../test_data/test_patches/arch/barrow_segmentation_TM75.gpkg",
        "enclosure": r"../test_data/test_patches/arch/enclosure_segmentation_TM75.gpkg",
        "ringfort": r"../test_data/test_patches/arch/ringfort_segmentation_TM75.gpkg"
    }

    create_patches_main(input_image, segmentation_masks, output_dir)
