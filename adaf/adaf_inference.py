# import multiprocessing as mp
# import shutil
import os
# from ipywidgets import interact, widgets
from pathlib import Path

import geopandas as gpd
import glob
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import box, shape

import grid_tools as gt
from adaf_vis import tiled_processing
from aitlas.models import FasterRCNN, HRNet
from utils import make_predictions_on_patches_object_detection, make_predictions_on_patches_segmentation
from vrt import build_vrt_from_list


# def create_patches(source_path, patch_size_px, save_dir, nr_processes):
#     multi_p = False
#
#     # Prepare paths
#     source_path = Path(source_path)
#     ds_path = source_path.parent
#     # Folder for patches
#     patch_dir = save_dir / "ml_patches"
#     patch_dir.mkdir(parents=True, exist_ok=True)
#
#     # Read metadata from source image
#     with rasterio.open(source_path) as src:
#         ds_res = src.res[0]
#         ds_crs = src.crs
#         ds_bounds = src.bounds
#
#     # -------- CREATE GRID --------
#     stride = 0.5
#     patch_size = (patch_size_px * ds_res, patch_size_px * ds_res)
#     stagger = stride * patch_size[0]
#
#     grid = uniform_grid(ds_bounds, ds_crs, patch_size, stagger)
#
#     # Filter grid (remove patches that fall outside scanned area)
#     vdm_path = list(ds_path.glob("*_validDataMask.*"))[0]
#     vdm = gpd.read_file(vdm_path)
#     # Make sure Valid Data Mask is in a correct CRS
#     if vdm.crs.to_epsg() != ds_crs.to_epsg():
#         vdm = vdm.to_crs(ds_crs)
#
#     vdm_filter = grid["geometry"].intersects(vdm.geometry[0])
#     grid = grid[vdm_filter].reset_index(drop=True)
#
#     if multi_p:
#         # Multiprocessing run
#         # TODO: Multiprocessing not working?!
#         # Create rasters/files and save them
#         input_process_list = []
#         for i, one_tile in grid.iterrows():
#             save_file = patch_dir / f"ml_patch_{i+1:06d}.tif"
#             out_nodata = 0
#             resample = False
#             input_process_list.append(
#                 (
#                     one_tile["geometry"],
#                     save_file,
#                     source_path,
#                     out_nodata,
#                     resample
#                 )
#             )
#         with mp.Pool(nr_processes) as p:
#             realist = [p.apply_async(clip_tile, r) for r in input_process_list]
#     else:
#         realist = [
#             clip_tile(
#                 p["geometry"],
#                 patch_dir / f"ml_patch_{i + 1:06d}.tif",
#                 source_path,
#                 out_nodata=0,
#                 resample=False
#             ) for i, p in grid.iterrows()
#         ]
#         # # -------- CLIP TILES --------
#         # for i, one_tile in grid.iterrows():
#         #     # # Source raster path
#         #     # src_pth = list(ds_path.glob(f"*{vis_type}.vrt"))[0]
#         #     out_nodata = 0
#         #     # out_nodata = -999
#         #
#         #     save_file = patch_dir / f"ml_patch_{i+1:06d}.tif"
#         #
#         #     # Create rasters/files and save them
#         #     clip_result = clip_tile(
#         #         one_tile["geometry"],
#         #         save_file,
#         #         source_path,
#         #         out_nodata=out_nodata,
#         #         resample=False
#         #     )
#         #     print(clip_result)
#
#     output = "#  - Finished creating " + str(len(grid)) + " patches for " + ds_path.stem
#
#     return output
#
#
# def uniform_grid(extents, crs, spacing_xy, stagger=None):
#     """ Creates uniform grid over the total extents.
#
#     Parameters
#     ----------
#     extents
#     crs
#     spacing_xy
#     stagger : float
#         Distance to use for translating the cell for creation of staggered grid.
#         Use None if grid should not be staggered.
#
#     Returns
#     -------
#
#     """
#     # total area for the grid [xmin, ymin, xmax, ymax]
#     x_min, y_min, x_max, y_max = extents  # gdf.total_bounds
#     tile_w, tile_h = spacing_xy
#
#     # Target Aligned Pixels
#     # x_min = np.floor(x_min / tile_w) * tile_w
#     # bottom = np.floor(extents.bottom / tile_h) * tile_h
#     # right = np.ceil(extents.right / tile_w) * tile_w
#     # y_max = np.ceil(y_max / tile_h) * tile_h
#     # _, bottom, right, _ = extents  # ONLY TOP-LEFT NEEDS TO BE ROUNDED
#
#     grid_cells = []
#     for x0 in np.arange(x_min, x_max, tile_w):
#         for y0 in np.arange(y_max, y_min, -tile_h):
#             # bounds
#             x1 = x0 + tile_w
#             y1 = y0 - tile_h
#             cell_1 = box(x0, y0, x1, y1)
#             grid_cells.append(cell_1)
#             if stagger:
#                 cell_2 = translate(cell_1, stagger, stagger)
#                 grid_cells.append(cell_2)
#
#     # Generate GeoDataFrame
#     grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)
#
#     return grid
#
#
# def clip_tile(poly, file_path, src_path, out_nodata=0, resample=False):
#     """Clips a single tile from a source raster."""
#     with rasterio.open(src_path) as src:
#         bounds = poly.bounds
#         orig_window = from_bounds(*bounds, src.transform)
#
#         out_image = src.read(window=orig_window, boundless=True)
#         out_transform = src.window_transform(orig_window)
#         out_profile = src.profile.copy()
#         src_nodata = src.nodata
#
#     if resample:
#         out_image, out_transform = reproject(
#             source=out_image,
#             src_crs=src.crs,
#             dst_crs=src.crs,
#             src_nodata=src_nodata,
#             dst_nodata=src_nodata,
#             src_transform=out_transform,
#             dst_resolution=0.5,
#             resampling=Resampling.bilinear
#         )
#
#     # Fill NaNs
#     if np.isnan(src_nodata):
#         out_image[np.isnan(out_image)] = out_nodata
#     else:
#         out_image[out_image == src_nodata] = out_nodata
#
#     # Assign correct nodata to metadata and clip to min/max values
#     if out_nodata == 0:
#         # This is used for all validations
#         meta_nd = None
#         out_image[out_image > 1] = 1
#         out_image[out_image < 0] = 0
#     else:
#         # This is used for DEM
#         meta_nd = out_nodata
#
#     # Update metadata and save geotiff
#     out_profile.update(
#         driver="GTiff",
#         compress="lzw",
#         width=out_image.shape[-1],
#         height=out_image.shape[-2],
#         transform=out_transform,
#         nodata=meta_nd
#     )
#     with rasterio.open(file_path, "w", **out_profile, predictor=3) as dst:
#         dst.write(out_image)
#
#     return file_path


def object_detection_vectors(path_to_patches, path_to_predictions):
    # Make sure paths are in Path object!
    path_to_patches = Path(path_to_patches)
    path_to_predictions = Path(path_to_predictions)
    # Prepare output path (GPKG file in the data folder)
    output_path = path_to_patches.parent / "object_detection.gpkg"

    appended_data = []
    for file in os.listdir(path_to_predictions):
        # Set path to individual PREDICTIONS FILE
        file_path = path_to_predictions / file

        # Only read files that are not empty
        if not os.stat(file_path).st_size == 0:
            # Find PATCH that belongs to the PREDICTIONS file
            patch_path = path_to_patches / (file[:-3] + "tif")

            # Arrays are indexed from the top-left corner, so we need minx and maxy
            with rasterio.open(patch_path) as src:
                crs = src.crs
                res = src.res[0]
                x_min = src.transform.c
                y_max = src.transform.f

            # Read predictions from TXT file
            data = pd.read_csv(file_path, sep=" ", header=None)
            data.columns = ["x0", "y0", "x1", "y1", "label", "score"]

            data.x0 = x_min + (res * data.x0)
            data.x1 = x_min + (res * data.x1)
            data.y0 = y_max - (res * data.y0)
            data.y1 = y_max - (res * data.y1)

            data["geometry"] = [box(*a) for a in zip(data.x0, data.y0, data.x1, data.y1)]
            data.drop(columns=["x0", "y0", "x1", "y1"], inplace=True)

            appended_data.append(data)

    appended_data = pd.concat(appended_data)

    appended_data = gpd.GeoDataFrame(appended_data, columns=["label", "score", 'geometry'], crs=crs)
    appended_data.to_file(output_path.as_posix(), driver="GPKG")

    return output_path.as_posix()


def semantic_segmentation_vectors(path_to_predictions):
    # TODO: Possible parameters
    threshold = 0.5
    labels = ["barrow", "ringfort", "enclosure"]  # TODO: Read this from model configuration

    # Prepare paths
    path_to_predictions = Path(path_to_predictions)
    # Output path (GPKG file in the data folder)
    output_path = path_to_predictions.parent / "semantic_segmentation.gpkg"

    grids = []
    for label in labels:
        tif_list = list(path_to_predictions.glob(f"*{label}*.tif"))

        # file = tif_list[4]
        poly = []
        for file in tif_list:
            with rasterio.open(file) as src:
                prob_mask = src.read()
                transform = src.transform
                crs = src.crs

                prediction = prob_mask.copy()

                feature = prob_mask >= threshold
                background = prob_mask < threshold

                prediction[feature] = 1
                prediction[background] = 0

                # Outputs a list of (polygon, value) tuples
                output = list(shapes(prediction, transform=transform))

                # Find polygon covering valid data (value = 1) and transform to GDF friendly format
                for polygon, value in output:
                    if value == 1:
                        poly.append(shape(polygon))

        # Make Geodataframe
        if poly:
            grid = gpd.GeoDataFrame(poly, columns=['geometry'], crs=crs)
            grid = grid.dissolve().explode(ignore_index=True)
            grid["label"] = label
            grids.append(grid)

    grids = gpd.GeoDataFrame(pd.concat(grids, ignore_index=True), crs=crs)
    grids.to_file(output_path.as_posix(), driver="GPKG")

    return output_path.as_posix()


def run_visualisations(dem_path, tile_size, save_dir, nr_processes=1):
    """

    dem_path:
        Can be any raster file (GeoTIFF and VRT supported.)
    tile_size:
        In pixels
    save_dir:
        Save directory
    nr_processes:
        Number of processes for parallel computing

    """
    # TODO: Probably good to create dict (JSOn) with all the paths that are created here?

    # Prepare paths
    in_file = Path(dem_path)
    ds_dir = in_file.parent

    # save_vis = save_dir / "vis"
    # save_vis.mkdir(parents=True, exist_ok=True)
    save_vis = save_dir  # TODO: fihgure out folder structure for outputs

    # === STEP 1 ===
    # We need polygon covering valid data
    valid_data_outline = gt.poly_from_valid(
        in_file.as_posix(),
        save_gpkg=save_vis  # directory where *_validDataMask.gpkg will be stored
    )

    # === STEP 2 ===
    # Create reference grid, filter it and save it to disk
    tiles_extents = gt.bounding_grid(
        in_file.as_posix(),
        tile_size,
        tag=False
    )
    refgrid_name = in_file.as_posix()[:-4] + "_refgrid.gpkg"
    tiles_extents = gt.filter_by_outline(
        tiles_extents,
        valid_data_outline,
        save_gpkg=True,
        save_path=refgrid_name
    )

    # === STEP 3 ===
    # Run visualizations
    print("Start RVT vis")
    out_path = tiled_processing(
        input_vrt_path=in_file.as_posix(),
        ext_list=tiles_extents,
        nr_processes=nr_processes,
        ll_dir=Path(save_vis)
    )

    # TODO remove refgrid and vdm HERE
    Path(valid_data_outline).unlink()
    Path(refgrid_name).unlink()

    return out_path


def main_routine(dem_path, ml_type, model_path, tile_size_px, nr_processes=1):
    # Save results to parent folder of input file
    save_dir = Path(dem_path).parent

    # ## 1 ## Create visualisation
    vis_path = run_visualisations(
        dem_path,
        tile_size_px,
        save_dir=save_dir.as_posix(),
        nr_processes=nr_processes
    )

    # # ## 2 ## Create patches
    # patches_dir = create_patches(
    #     vis_path,
    #     patch_size_px,
    #     save_dir,
    #     nr_processes=nr_processes
    # )
    # shutil.rmtree(vis_path)

    if ml_type == "object detection":
        # ## 3 ## Run the model
        model_config = {
            "num_classes": 4,  # Number of classes in the dataset
            "learning_rate": 0.001,  # Learning rate for training
            "pretrained": True,  # Whether to use a pretrained model or not
            "use_cuda": False,  # Set to True if you want to use GPU acceleration
            "metrics": ["map"]  # Evaluation metrics to be used
        }
        model = FasterRCNN(model_config)
        model.prepare()
        model.load_model(model_path)
        print("Model successfully loaded.")
        predictions_dir = make_predictions_on_patches_object_detection(
            model=model,
            patches_folder=vis_path.as_posix()
        )

        # ## 4 ## Create map
        vector_path = object_detection_vectors(vis_path, predictions_dir)

    elif ml_type == "segmentation":
        # ## 3 ## Run the model
        model_config = {
            "num_classes": 3,  # Number of classes in the dataset
            "learning_rate": 0.0001,  # Learning rate for training
            "pretrained": True,  # Whether to use a pretrained model or not
            "use_cuda": False,  # Set to True if you want to use GPU acceleration
            "threshold": 0.5,
            "metrics": ["map"]  # Evaluation metrics to be used
        }
        model = HRNet(model_config)
        model.prepare()
        model.load_model(model_path)
        print("Model successfully loaded.")
        predictions_dir = make_predictions_on_patches_segmentation(
            model=model,
            patches_folder=vis_path.as_posix()
        )
        # ## 4 ## Create map
        vector_path = semantic_segmentation_vectors(predictions_dir)
    else:
        raise Exception("Wrong ml_type: choose 'object detection' or 'segmentation'")

    # ## 5 ## Create VRT file for predictions
    for label in ["barrow", "ringfort", "enclosure"]:
        print("Creating vrt for", label)
        tif_list = glob.glob((Path(predictions_dir) / f"*{label}*.tif").as_posix())
        vrt_name = save_dir / (Path(predictions_dir).stem + f"_{label}.vrt")
        build_vrt_from_list(tif_list, vrt_name)

    return vector_path


if __name__ == "__main__":
    my_file = r"c:\Users\ncoz\GitHub\aitlas-TII-LIDAR\inference\data-small_debug\ISA-147_small.tif"

    my_ml_type = "segmentation"  # "segmentation" or "object detection"

    my_tile_size_px = 512

    # Specify the path to the model
    # OBJECT DETECTION:
    # my_model_path = r"c:\Users\ncoz\GitHub\aitlas-TII-LIDAR\inference\data\model_object_detection_BRE_12.tar"
    # SEGMENTATION:
    my_model_path = r"c:\Users\ncoz\GitHub\aitlas-TII-LIDAR\inference\data\model_semantic_segmentation_BRE_124.tar"

    rs = main_routine(my_file, my_ml_type, my_model_path, my_tile_size_px, nr_processes=6)

    # rs = object_detection_vectors(
    #     r"c:\Users\ncoz\GitHub\aitlas-TII-LIDAR\inference\data-147\slrm",
    #     r"c:\Users\ncoz\GitHub\aitlas-TII-LIDAR\inference\data-147\predictions_object_detection"
    # )

    # rs = run_visualisations(
    #     r"c:\Users\ncoz\GitHub\aitlas-TII-LIDAR\inference\data-small_debug\ISA-147_small.tif",
    #     1024,
    #     save_dir=r"c:\Users\ncoz\GitHub\aitlas-TII-LIDAR\inference\data-small_debug",
    #     nr_processes=6
    # )

    print(rs)
