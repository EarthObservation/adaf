import glob
import logging
import os
import shutil
import time
from pathlib import Path
from time import localtime, strftime

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from aitlas.models import FasterRCNN, HRNet
from pyproj import CRS
from rasterio.features import shapes
from shapely.geometry import box, shape
from torch import cuda

import adaf.grid_tools as gt
from adaf.adaf_utils import (
    make_predictions_on_patches_object_detection,
    make_predictions_on_patches_segmentation,
    build_vrt_from_list,
    Logger,
    image_tiling
)

from adaf.adaf_vis import tiled_processing

logging.disable(logging.INFO)


def object_detection_vectors(predictions_dirs_dict, threshold=0.5, keep_ml_paths=False, min_area=None):
    """Converts object detection bounding boxes from text to vector format.

    Parameters
    ----------
    predictions_dirs_dict : dict
        Key is ML label, value is path to directory with results for that label.
    threshold : float
        Probability threshold for predictions.
    keep_ml_paths : bool
        If true, add path to ML predictions file from which the label was created as an attribute.
    min_area : float
        Minimum area threshold in m^2 (max = 40 m^2).

    Returns
    -------
    output_path : str
        Path to vector file.
    """
    # Use Path from pathlib
    path_to_predictions = Path(list(predictions_dirs_dict.values())[0])
    # Prepare output path (GPKG file in the data folder)
    output_path = path_to_predictions.parent / "object_detection.gpkg"

    appended_data = []
    crs = None
    for label, predicts_dir in predictions_dirs_dict.items():
        predicts_dir = Path(predicts_dir)
        file_list = list(predicts_dir.glob(f"*.txt"))

        for file_path in file_list:
            # Only read files that are not empty
            if not os.stat(file_path).st_size == 0:
                # Read predictions from TXT file
                data = pd.read_csv(file_path, sep=" ", header=None)
                data.columns = ["x0", "y0", "x1", "y1", "label", "score", "epsg", "res", "x_min", "y_max"]

                # EPSG code is added to every bbox, doesn't matter which we chose, it has to be the same for all entries
                if crs is None:
                    crs = CRS.from_epsg(int(data.epsg[0]))

                data.x0 = data.x_min + (data.res * data.x0)
                data.x1 = data.x_min + (data.res * data.x1)
                data.y0 = data.y_max - (data.res * data.y0)
                data.y1 = data.y_max - (data.res * data.y1)

                data["geometry"] = [box(*a) for a in zip(data.x0, data.y0, data.x1, data.y1)]
                data.drop(columns=["x0", "y0", "x1", "y1", "epsg", "res", "x_min", "y_max"], inplace=True)

                # Filter by probability threshold
                data = data[data['score'] > threshold]

                # Convert pandas to geopandas
                data = gpd.GeoDataFrame(data, crs=crs)

                # Add paths to ML results
                agg_func = {'score': 'max', 'label': 'first'}
                if keep_ml_paths:
                    data["prediction_path"] = str(Path().joinpath(*file_path.parts[-3:]))
                    agg_func['prediction_path'] = 'first'

                # Don't append if there are no predictions left after filtering
                if data.shape[0] > 0:
                    # Join overlapping polygons (dissolve and keep disjoint separate)
                    data_ = gpd.GeoDataFrame(
                        geometry=[data.unary_union],
                        crs=data.crs
                    ).explode(index_parts=False).reset_index(drop=True)
                    # Keep attributes from original gdf, select max score
                    data_ = gpd.sjoin(data_, data, how='left').drop(columns=['index_right'])

                    data_ = data_.dissolve(data_.index, aggfunc=agg_func)

                    appended_data.append(data_)

    if appended_data:
        # We have at least one detection
        gdf = gpd.GeoDataFrame(pd.concat(appended_data, ignore_index=True), crs=crs)

        # Post-processing
        if min_area:
            gdf["area"] = gdf.geometry.area
            gdf = gdf[gdf["area"] > min_area]

        # Export file
        gdf.to_file(str(output_path), driver="GPKG")
    else:
        output_path = ""

    return str(output_path)


def semantic_segmentation_vectors(predictions_dirs_dict, threshold=0.5,
                                  keep_ml_paths=False, roundness=None, min_area=None):
    """Converts semantic segmentation probability masks to polygons using a threshold. If more than one class, all
    predictions are stored in the same vector file, class is stored as label attribute.

    Parameters
    ----------
    predictions_dirs_dict : dict
        Key is ML label, value is path to directory with results for that label.
    threshold : float
        Probability threshold for predictions.
    keep_ml_paths : bool
        If true, add path to ML predictions file from which the label was created as an attribute.
    roundness : float
        Roundness threshold for post-processing. Remove features that fall below the threshold.
        For perfect circle roundness is 1, for square 0.785, and goes towards 0 for irregular shapes.
    min_area : float
        Minimum area threshold in m^2 (max = 40 m^2).

    Returns
    -------
    output_path : str
        Path to vector file.
    """
    # Prepare paths, use Path from pathlib (select one from dict, we only need parent)
    path_to_predictions = Path(list(predictions_dirs_dict.values())[0])
    # Output path (GPKG file in the data folder)
    output_path = path_to_predictions.parent / "semantic_segmentation.gpkg"

    gdf_out = []
    for label, predicts_dir in predictions_dirs_dict.items():
        predicts_dir = Path(predicts_dir)
        tif_list = list(predicts_dir.glob(f"*.tif"))

        # file = tif_list[4]

        for file in tif_list:
            with rasterio.open(file) as src:
                prob_mask = src.read()
                transform = src.transform
                crs = src.crs

                prediction = prob_mask.copy()

                # Mask probability map by threshold for extraction of polygons
                feature = prob_mask >= float(threshold)
                background = prob_mask < float(threshold)

                prediction[feature] = 1
                prediction[background] = 0

                # Outputs a list of (polygon, value) tuples
                output = list(shapes(prediction, transform=transform))

                # Find polygon covering valid data (value = 1) and transform to GDF friendly format
                poly = []
                for polygon, value in output:
                    if value == 1:
                        poly.append(shape(polygon))

            # If there is at least one polygon, convert to GeoDataFrame and append to list for output
            if poly:
                predicted_labels = gpd.GeoDataFrame(poly, columns=['geometry'], crs=crs)
                predicted_labels = predicted_labels.dissolve().explode(ignore_index=True)
                predicted_labels["label"] = label
                if keep_ml_paths:
                    predicted_labels["prediction_path"] = str(Path().joinpath(*file.parts[-3:]))
                gdf_out.append(predicted_labels)

    if gdf_out:
        # We have at least one detection
        gdf = gpd.GeoDataFrame(pd.concat(gdf_out, ignore_index=True), crs=crs)

        # # If same object from two different tiles overlap, join them into one
        # In semantic segmentation this will never happen, because each pixel can belong to only one polygon (when
        # creating polygons from probability masks.

        # Post-processing
        if roundness:
            gdf["roundness"] = 4 * np.pi * gdf.geometry.area / (gdf.geometry.convex_hull.length ** 2)
            gdf = gdf[gdf["roundness"] > roundness]
        if min_area:
            gdf["area"] = gdf.geometry.area
            gdf = gdf[gdf["area"] > min_area]

        # Export file
        gdf.to_file(output_path.as_posix(), driver="GPKG")
    else:
        output_path = ""

    return str(output_path)


def run_visualisations(dem_path, tile_size, save_dir, nr_processes=1):
    """Calculates visualisations from DEM and saves them into VRT (Geotiff) file.

    Uses RVT (see adaf_vis.py).

    Parameters
    ----------
    dem_path : str or pathlib.Path()
        Can be any raster file (GeoTIFF and VRT supported).
    tile_size : int
        In pixels.
    save_dir : str
        Save directory.
    nr_processes : int
        Number of processes for parallel computing.

    Returns
    -------
    dict
        A python dictionary containing results from tiling, such as paths and processing time.
    """
    # Prepare paths
    in_file = Path(dem_path)

    # We need polygon covering valid data
    valid_data_outline, _ = gt.poly_from_valid(in_file.as_posix())

    # Create reference grid, filter it and save it to disk
    tiles_extents = gt.bounding_grid(in_file.as_posix(), tile_size, tag=False)
    tiles_extents = gt.filter_by_outline(tiles_extents, valid_data_outline)

    # Run visualizations
    logging.debug("Start RVT vis")
    out_paths = tiled_processing(
        input_raster_path=in_file.as_posix(),
        extents_list=tiles_extents,
        nr_processes=nr_processes,
        save_dir=Path(save_dir)
    )

    return out_paths


def run_tiling(dem_path, tile_size, save_dir, nr_processes=1):
    """Cuts visualisation into tiles.

    Parameters
    ----------
    dem_path : str or pathlib.Path()
        Can be any raster file (GeoTIFF and VRT supported).
    tile_size : int
        In pixels.
    save_dir : str
        Save directory.
    nr_processes : int
        Number of processes for parallel computing.

    Returns
    -------
    dict
        A python dictionary containing results from tiling, such as paths and processing time.
    """
    # Prepare paths
    in_file = Path(dem_path)

    # We need polygon covering valid data
    valid_data_outline, _ = gt.poly_from_valid(in_file.as_posix())

    # Create reference grid and filter it
    tiles_extents = gt.bounding_grid(in_file.as_posix(), tile_size, tag=False)
    tiles_extents = gt.filter_by_outline(tiles_extents, valid_data_outline)

    # Run tiling
    logging.debug("Start RVT vis")
    out_paths = image_tiling(
        source_path=in_file.as_posix(),
        ext_list=tiles_extents,
        nr_processes=nr_processes,
        save_dir=Path(save_dir)
    )

    return out_paths


def run_aitlas_object_detection(labels, images_dir, custom_model=None):
    """Runs AiTLAS for object detection. There are 4 trained models (binary classification) for four different classes
    (e.g. labels). The models are stored relatively to the script path in the "ml_models" folder.

    Parameters
    ----------
    labels : list
        A list of labels for which to run the model, can be barrow, enclosure, ringfort or AO.
    images_dir : str or pathlib.Path()
        Path to directory containing tiles for inference.
    custom_model : str or pathlib.Path()
        Path to tar file for custom model.

    Returns
    -------
    dict
        A dictionary with a list of paths for each label. The paths are of the result files of object detection.
    """
    images_dir = str(images_dir)

    # Paths to models are relative to the script path
    models = {
        "barrow": r".\ml_models\OD_barrow.tar",
        "enclosure": r".\ml_models\OD_enclosure.tar",
        "ringfort": r".\ml_models\OD_ringfort.tar",
        "AO": r".\ml_models\OD_AO.tar",
        "custom": custom_model
    }

    if cuda.is_available():
        logging.debug("> CUDA is available, running predictions on GPU!")
    else:
        logging.debug("> No CUDA detected, running predictions on CPU!")

    predictions_dirs = {}
    for label in labels:
        # Prepare the model
        model_config = {
            "num_classes": 2,  # Number of classes in the dataset
            "learning_rate": 0.0001,  # Learning rate for training
            "pretrained": True,  # Whether to use a pretrained model or not
            "use_cuda": cuda.is_available(),  # Set to True if you want to use GPU acceleration
            "metrics": ["map"]  # Evaluation metrics to be used
        }
        model = FasterRCNN(model_config)
        model.prepare()

        # Prepare path to the model
        model_path = models.get(label)
        # Path is relative to the Current script directory
        model_path = Path(__file__).resolve().parent / model_path
        # Load appropriate ADAF model
        model.load_model(model_path)
        logging.debug("Model successfully loaded.")

        preds_dir = make_predictions_on_patches_object_detection(
            model=model,
            label=label,
            patches_folder=images_dir
        )

        predictions_dirs[label] = preds_dir

    return predictions_dirs


def run_aitlas_segmentation(labels, images_dir, custom_model=None):
    """Runs AiTLAS for segmentation. There are 4 trained models (binary classification) for four different classes
    (e.g. labels). The models are stored relatively to the script path in the "ml_models" folder.

    Parameters
    ----------
    labels : list
        A list of labels for which to run the model, can be barrow, enclosure, ringfort or AO.
    images_dir : str or pathlib.Path()
        Path to directory containing tiles for inference.
    custom_model : str or pathlib.Path()
        Path to tar file for custom model.

    Returns
    -------
    dict
        A dictionary with a list of paths for each label. The paths are of the result files of segmentation.
    """
    images_dir = str(images_dir)

    # Paths to models are relative to the script path
    models = {
        "barrow": r".\ml_models\barrow_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
        "enclosure": r".\ml_models\enclosure_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
        "ringfort": r".\ml_models\ringfort_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
        "AO": r".\ml_models\AO_HRNet_SLRM_512px_pretrained_train_12_val_124_with_Transformation.tar",
        "custom": custom_model
    }

    if cuda.is_available():
        logging.debug("> CUDA is available, running predictions on GPU!")
    else:
        logging.debug("> No CUDA detected, running predictions on CPU!")

    predictions_dirs = {}
    for label in labels:
        # Prepare the model
        model_config = {
            "num_classes": 2,  # Number of classes in the dataset
            "learning_rate": 0.0001,  # Learning rate for training
            "pretrained": True,  # Whether to use a pretrained model or not
            "use_cuda": cuda.is_available(),  # Set to True if you want to use GPU acceleration
            "threshold": 0.5,
            "metrics": ["iou"]  # Evaluation metrics to be used
        }
        model = HRNet(model_config)
        model.prepare()

        logging.debug(label)

        # Prepare path to the model
        model_path = models.get(label)
        # Path is relative to the Current script directory
        model_path = Path(__file__).resolve().parent / model_path

        logging.debug(model_path)

        # Load appropriate ADAF model
        model.load_model(model_path)
        logging.debug("Model successfully loaded.")

        # Run inference
        preds_dir = make_predictions_on_patches_segmentation(
            model=model,
            label=label,
            patches_folder=images_dir
        )

        predictions_dirs[label] = preds_dir

    return predictions_dirs


def main_routine(inp):
    """Main processing routine of ADAF. It is started by pressing the RUN button on the widget.

    Parameters
    ----------
    inp : adaf_utils.ADAFInput()
        An object containing all the input parameters from the widget.

    Returns
    -------
    str
        Path to GPKG (vector) file with results of the ML detection.
    """
    dem_path = Path(inp.dem_path)
    out_dir = Path(inp.out_dir)

    # Create unique name for results
    time_started = localtime()
    t0 = time.time()

    # Create folder for results (time-stamped)
    if inp.ml_type == "object detection":
        suff = "_obj"
    else:
        suff = "_seg"
    save_dir = out_dir / (dem_path.stem + strftime("_%Y%m%d_%H%M%S", time_started) + suff)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Create logfile
    log_path = save_dir / "logfile.txt"
    logger = Logger(log_path, log_time=time_started)

    # --- VISUALIZATIONS ---
    logger.log_vis_inputs(dem_path, inp.vis_exist_ok)
    t1 = time.time()

    # Determine nr_processes from available CPUs (leave two free)
    my_cpus = os.cpu_count() - 2
    if my_cpus < 1:
        my_cpus = 1
    # The processing of the image is done on tiles (for better performance)
    tile_size_px = 1024  # Tile size has to be in base 2 (512, 1024) for inference to work!

    # vis_path is folder where visualizations are stored
    if inp.vis_exist_ok:
        # Create tiles (because image pix size has to be divisible by 32)
        out_paths = run_tiling(
            dem_path,
            tile_size_px,
            save_dir=save_dir.as_posix(),
            nr_processes=my_cpus
        )
    else:
        # Create visualisations
        out_paths = run_visualisations(
            dem_path,
            tile_size_px,
            save_dir=save_dir.as_posix(),
            nr_processes=my_cpus
        )

    vis_path = out_paths["output_directory"]
    vrt_path = out_paths["vrt_path"]

    t1 = time.time() - t1
    logger.log_vis_results(vis_path, vrt_path, inp.save_vis, t1)

    # Make sure it is a Path object!
    vis_path = Path(vis_path)

    # --- INFERENCE ---
    # Select name of the label for custom model
    if inp.ml_model_custom == "Custom model":
        labels = ["custom"]
    else:
        labels = inp.labels

    logger.log_inference_inputs(inp.ml_type,  labels, inp.ml_model_custom, inp.custom_model_pth)
    # For logger
    save_raw = []
    t2 = time.time()
    if inp.ml_type == "object detection":
        logging.debug("Running object detection")
        predictions_dict = run_aitlas_object_detection(labels, vis_path, inp.custom_model_pth)

        vector_path = object_detection_vectors(
            predictions_dict,
            keep_ml_paths=inp.save_ml_output,
            min_area=inp.min_area
        )
        if vector_path != "":
            logging.debug("Created vector file", vector_path)
        else:
            logging.debug("No archaeology detected")

        # Remove predictions files (bbox txt)
        if not inp.save_ml_output:
            for _, p_dir in predictions_dict.items():
                shutil.rmtree(p_dir)
        else:
            save_raw = [a for _, a in predictions_dict.items()]

    elif inp.ml_type == "segmentation":
        logging.debug("Running segmentation")
        predictions_dict = run_aitlas_segmentation(labels, vis_path, inp.custom_model_pth)

        vector_path = semantic_segmentation_vectors(
            predictions_dict,
            keep_ml_paths=inp.save_ml_output,
            roundness=inp.roundness,
            min_area=inp.min_area
        )
        if vector_path != "":
            logging.debug("Created vector file", vector_path)
        else:
            logging.debug("No archaeology detected")

        # Save predictions files (probability masks)
        if inp.save_ml_output:
            # Create VRT file for predictions
            for label, p_dir in predictions_dict.items():
                logging.debug("Creating vrt for", label)
                tif_list = glob.glob((Path(p_dir) / f"*{label}*.tif").as_posix())
                vrt_name = save_dir / (Path(p_dir).stem + ".vrt")
                build_vrt_from_list(tif_list, vrt_name)
                save_raw.append(vrt_name)
        else:
            for _, p_dir in predictions_dict.items():
                shutil.rmtree(p_dir)

    else:
        raise Exception("Wrong ml_type: choose 'object detection' or 'segmentation'")
    t2 = time.time() - t2

    logger.log_inference_results(vector_path, t2, save_raw, inp.min_area, inp.roundness)

    # Remove visualizations
    if not inp.save_vis:
        shutil.rmtree(vis_path)
        if vrt_path:
            Path(vrt_path).unlink()

    # TOTAL PROCESSING TIME
    t0 = time.time() - t0
    logger.log_total_time(t0)

    logging.debug("\n--\nFINISHED!")

    return vector_path
