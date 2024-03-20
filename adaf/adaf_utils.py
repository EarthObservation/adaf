"""
ADAF - utility functions
Created on 26 May 2023
@author: Nejc Čož, ZRC SAZU, Novi trg 2, 1000 Ljubljana, Slovenia
"""
import logging
import multiprocessing as mp
import os
import warnings
from pathlib import Path
from time import localtime, strftime

import numpy as np
import rasterio
from aitlas.transforms import ResizeV2
from aitlas.transforms import Transpose
from osgeo import gdal
from rasterio.windows import from_bounds

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


def make_predictions_on_single_patch_store_preds_single_class(
        model,
        label,
        image_path,
        image_filename,
        predictions_dir
):
    transform = ResizeV2()

    with rasterio.open(image_path) as image_tiff:
        image = image_tiff.read()
        # The following are required to construct vector from txt
        epsg = image_tiff.crs.to_epsg()
        res = image_tiff.res[0]
        x_min = image_tiff.transform.c
        y_max = image_tiff.transform.f

    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)
    image = np.transpose(image, (1, 2, 0))
    predictions_single_patch_str = ""
    predicted = model.detect_objects_v2(image, [None], transform)

    for i in range(0, len(predicted['boxes'])):
        box = predicted['boxes'][i].detach().numpy()
        score = predicted['scores'][i].detach().numpy()
        predictions_single_patch_str += (
            f'{round(box[0])} '
            f'{round(box[1])} '
            f'{round(box[2])} '
            f'{round(box[3])} '
            f'{label} '
            f'{score:.4f} '
            f'{epsg} {res} {x_min} {y_max}'
            f'\n'
        )
    filepath = os.path.join(predictions_dir, f"{os.path.splitext(image_filename)[0]}_{label}_bounding_boxes.txt")
    file = open(filepath, "w")
    file.write(predictions_single_patch_str)
    file.close()


def make_predictions_on_patches_object_detection(model, label, patches_folder, predictions_dir=None):
    """Generates predictions on patches (the model performs binary object detection).

    Parameters
    ----------
    model
        Selected AITLAS ML model.
    label : str
        One of the allowed classes (barrow, enclosure, ringfort, AO).
    patches_folder : str or pathlib.Path()
        Path to folder containing images for inference.
    predictions_dir : str or pathlib.Path()
        Optional - user can specify a custom folder. Otherwise, a folder called "predictions_segmentation_{label}" is
        created.

    Returns
    -------
    str
        Path to directory with predictions.
    """
    patches_folder = Path(patches_folder)
    # If predictions_dir is not given, results are saved into a default folder
    if predictions_dir is None:
        predictions_dir = patches_folder.parent / f"predictions_object_detection_{label}"
    else:
        predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    logging.debug("Generating predictions:")
    for file in os.listdir(patches_folder):
        if file.endswith(".tif"):
            logging.debug(">>> ", file)
            image_path = os.path.join(patches_folder, file)
            image_filename = file
            make_predictions_on_single_patch_store_preds_single_class(
                model,
                label,
                image_path,
                image_filename,
                str(predictions_dir)
            )

    return str(predictions_dir)


def make_predictions_on_patches_segmentation(model, label, patches_folder, predictions_dir=None):
    """Generates predictions on patches (the model performs binary semantic segmentation).

    Parameters
    ----------
    model
        Selected AITLAS ML model.
    label : str
        One of the allowed classes (barrow, enclosure, ringfort, AO).
    patches_folder : str or pathlib.Path()
        Path to folder containing images for inference.
    predictions_dir : str or pathlib.Path()
        Optional - user can specify a custom folder. Otherwise, a folder called "predictions_segmentation_{label}" is
        created.

    Returns
    -------
    str
        Path to directory with predictions.
    """
    patches_folder = Path(patches_folder)
    # If predictions_dir is not given, results are saved into a default folder
    if predictions_dir is None:
        predictions_dir = patches_folder.parent / f"predictions_segmentation_{label}"
    else:
        predictions_dir = Path(predictions_dir)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    logging.debug("Generating predictions:")
    for file in os.listdir(patches_folder):
        logging.debug(">>> ", file)
        if file.endswith(".tif"):
            image_path = os.path.join(patches_folder, file)
            model.predict_masks_tiff_probs_binary(
                image_path=image_path,
                label=label,
                data_transforms=Transpose(),
                predictions_dir=str(predictions_dir)
            )

    return str(predictions_dir)


def build_vrt_from_list(tif_list, vrt_path):
    """Create a VRT file from a list of GeoTIFF files. The file is created on the same level as the input directory.

    Parameters
    ----------
    tif_list : list
        A aist of paths to the individual files.
    vrt_path : pathlib.Path()
        Path to output VRT file.

    Returns
    -------
    str
        Path to output VRT file.
    """
    vrt_options = gdal.BuildVRTOptions()
    my_vrt = gdal.BuildVRT(vrt_path.as_posix(), tif_list, options=vrt_options)
    my_vrt = None

    return vrt_path


def build_vrt(ds_dir, vrt_name):
    """Create a VRT file from directory containing TIFFs. The file is created on the same level as the input directory.

    Parameters
    ----------
    ds_dir : str or pathlib.Path()
        Path to directory containing raster files to be joined into a VRT.
    vrt_name : str
        File name of the VRT file.

    Returns
    -------
    str
        Path to VRT file.
    """
    ds_dir = Path(ds_dir)
    vrt_path = ds_dir.parents[0] / vrt_name
    tif_list = [a.as_posix() for a in Path(ds_dir).glob("*.tif")]

    vrt_options = gdal.BuildVRTOptions()
    my_vrt = gdal.BuildVRT(vrt_path.as_posix(), tif_list, options=vrt_options)
    my_vrt = None

    return vrt_path


class Logger:
    def __init__(self, log_file_path, log_time=None):
        """Initiates logfile, creates file and writes the header of the log file."""

        self.log_file_path = log_file_path

        if log_time is None:
            self.log_time = localtime()
        else:
            self.log_time = log_time

        time_stamp = strftime("%d/%m/%Y %H:%M:%S", self.log_time)

        log_entry = (
            f"=================================================================================\n"
            f"Automatic Detection of Archaeological Features (ADAF)\n\n"
            f"Processing log - {time_stamp}\n\n"
        )

        with open(self.log_file_path, 'w') as log:
            log.write(log_entry)

    def log(self, message):
        """Adds a line with datetime and message to logfile."""
        timestamp = strftime('%Y-%m-%d %H:%M:%S', localtime())
        log_entry = f'[{timestamp}] {message}\n'

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)

    def log_section(self, section):
        """Creates a header for a new section in the log file"""
        log_entry = (
            f"=================================================================================\n"
            f"{section.capitalize()} log:\n\n"
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)

    def log_vis_inputs(self, image_path, vis_exist):
        """Adds parameters of visualization module to log file"""
        # Load image metadata
        image_log = self.log_input_image(image_path)

        if vis_exist:
            comment = "Visualization already exist"
        else:
            comment = "Processing visualization from DEM file"

        log_entry = (
            f"=================================================================================\n"
            f"Visualizations log:\n"
            f"\n"
            f"- {comment}:\n"
            f"\n"
            + image_log
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)

    def log_vis_results(self, vis_dir, vrt_path, save_vis, processing_time):
        """Adds results of visualization module to log file"""
        vis_dir = Path(vis_dir)
        vrt_path = Path(vrt_path)

        # Count number of created tiles
        tiles_count = len(list(vis_dir.glob('*.tif')))

        # PROCESSING TIME IS IN SECONDS
        if processing_time >= 60:
            processing_time = processing_time / 60
            time_unit = "min"
        else:
            time_unit = "sec"

        if save_vis:
            sv = (
                f"    save visualization: YES\n"
                f"    tiles location:     {vis_dir}\n"
                f"    tiles count:        {tiles_count}\n"
                f"    VRT file path:      {vrt_path}\n"
                f"\n"
            )
        else:
            sv = f"    save visualization: NO\n\n"

        log_entry = (
            f"{sv}"
            f"TIME: {processing_time:.1f} {time_unit}\n"
            f"\n"
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)

    @staticmethod
    def log_input_image(image_path):
        """Gets metadata of the input image."""

        # Get input image
        with rasterio.open(image_path) as src:
            crs = src.crs.to_epsg()
            width = src.width
            height = src.height
            if src.compression:
                compress = src.compression.value
            else:
                compress = "None"

        file_size_bytes = os.path.getsize(image_path)
        if file_size_bytes >= (1024.0 ** 3):
            # Convert to GB
            file_size = file_size_bytes / (1024.0 ** 3)
            fs_unit = "GB"
        elif file_size_bytes >= (1024.0 ** 2):
            # Convert to GB
            file_size = file_size_bytes / (1024.0 ** 2)
            fs_unit = "MB"
        else:
            file_size = file_size_bytes / 1024
            fs_unit = "kB"

        log_entry = (
            f"{image_path}\n\n"
            f"    image size:  {width}x{height} pixels\n"
            f"    CRS:         EPSG:{crs}\n"
            f"    size:        {file_size:.2f} {fs_unit}\n"
            f"    compression: {compress}\n"
            f"\n"
        )

        return log_entry

    def log_inference_inputs(self, ml_method, ml_labels, ml_model="ADAF", custom_path=''):
        """Adds parameters for inference to log file-"""

        if ml_method == "segmentation":
            ml_method = "Semantic segmentation"

        if ml_method == "object detection":
            ml_method = ml_method.capitalize()

        if ml_model == "Custom model":
            custom_str = f"    - Path: {custom_path}\n"
        else:
            custom_str = ""

        # Write labels
        # log_labels = ""
        # for lbl in ml_labels:
        #     log_labels += 17 * " " + f"* {lbl}\n"

        lbl = "\n" + 22 * " " + "* "
        log_labels = lbl.join(ml_labels)

        log_entry = (
            f"=================================================================================\n"
            f"Inference log:\n"
            "\n"
            f"    ML method: {ml_method}\n"
            f"    ML model: {ml_model}\n{custom_str}"
            f"    Selected classes: * {log_labels}"
            f"\n"
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)

    def log_inference_results(self, vector_path, processing_time, list_to_raw_files, min_area, roundness=None):
        """Adds results of inference to the log file"""

        vector_path = Path(vector_path)

        # Count number of created tiles
        log_raw = ""
        if list_to_raw_files:
            for raw in list_to_raw_files:
                log_raw += f"      > {raw}\n"
            log_raw = f"    Save RAW results:  YES\n{log_raw}"
        else:
            log_raw = f"    Save RAW results:  NO\n"

        # PROCESSING TIME IS IN SECONDS
        if processing_time >= 60:
            processing_time = processing_time / 60
            time_unit = "min"
        else:
            time_unit = "sec"

        if roundness:
            roundness = f"      > Minimum roundness [-]: {roundness}\n"

        log_entry = (
            f"\n"
            f"    Postprocessing options:\n"
            f"      > Minimum area [m^2]: {min_area}\n{roundness}"
            f"\n"
            f"    Results vector file:\n"
            f"      > {vector_path}\n"
            f"\n{log_raw}"
            f"\n"
            f"TIME: {processing_time:.1f} {time_unit}\n"
            f"\n"
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)

    def log_total_time(self, processing_time):
        """Creates a header for a new section in the log file"""
        # PROCESSING TIME IS IN SECONDS
        if processing_time >= 60:
            processing_time = processing_time / 60
            time_unit = "min"
        else:
            time_unit = "sec"

        log_entry = (
            f"=================================================================================\n"
            f"TOTAL TIME: {processing_time:.1f} {time_unit}\n\n"
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)


class ADAFInput:
    """ADAF input parameters."""
    def __init__(self):
        self.input_file_list = None
        self.vis_exist_ok = None
        self.save_vis = None
        self.ml_type = None
        self.labels = None
        self.ml_model_custom = None
        self.custom_model_pth = None
        self.roundness = None
        self.min_area = None
        self.save_ml_output = None
        self.out_dir = None
        self.tiles_to_vrt = None
        self.dem_path = None

    # def __getattr__(self, attr):
    #     category, key, value = attr.split('.')
    #     if category in ("vis", "inference") and key in self.__dict__[category]:
    #         return self.__dict__[category][key]
    #     else:
    #         raise AttributeError(f"'MyInput' object has no attribute '{attr}'")

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
                # logging.debug(f"{key} updated to {value}")
            else:
                logging.debug(f"Invalid parameter: {key}")


def clip_tile(bounds, out_file_path, src_path, out_nodata=0):
    """Clips a single tile from a source raster and saves it to disk (GeoTIFF).

    Parameters
    ----------
    bounds : list
        Geographical extents of the tile ["minx", "miny", "maxx", "maxy"].
    out_file_path : str or pathlib.Path()
        Path of output file.
    src_path : str or pathlib.Path()
        Path of source raster, from which we are cutting out the tile.
    out_nodata : float
        Value of nodata pixels for the output tile.

    Returns
    -------
        Path to output file.
    """
    with rasterio.open(src_path) as src:
        orig_window = from_bounds(*bounds, src.transform)

        out_image = src.read(window=orig_window, boundless=True)
        out_transform = src.window_transform(orig_window)
        out_profile = src.profile.copy()
        src_nodata = src.nodata

    # Fill NaNs
    if np.isnan(src_nodata):
        out_image[np.isnan(out_image)] = out_nodata
    else:
        out_image[out_image == src_nodata] = out_nodata

    # Assign correct nodata to metadata and clip to min/max values
    if out_nodata == 0:
        # This is used for all validations
        meta_nd = None
        out_image[out_image > 1] = 1
        out_image[out_image < 0] = 0
    else:
        # This is used for DEM
        meta_nd = out_nodata

    # Update metadata and save geotiff
    out_profile.update(
        driver="GTiff",
        compress="lzw",
        width=out_image.shape[-1],
        height=out_image.shape[-2],
        transform=out_transform,
        nodata=meta_nd
    )
    with rasterio.open(out_file_path, "w", **out_profile, predictor=3) as dst:
        dst.write(out_image)

    return out_file_path


def image_tiling(
        source_path,
        ext_list,
        nr_processes=7,
        save_dir=None
):
    """Multiprocessing for clip_tile().

    Parameters
    ----------
    source_path : str or pathlib.Path()
        Path of source raster, from which we are cutting out the tile.
    ext_list : gpd.GeoDataFrame
        A list of geographical extents of all the tiles in ["minx", "miny", "maxx", "maxy"] format.
    nr_processes : int
        Number of processes for multiprocessing.
    save_dir : pathlib.Path()
        Path to directory containing output files.

    Returns
    -------
    dict
        A pythin dictionary containing paths to created files.
    """
    # Prepare paths
    source_path = Path(source_path)
    src_stem = source_path.stem
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = source_path.parent

    # Folder for patches
    patch_dir = save_dir / "tiled_image"
    patch_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    input_process_list = []
    ext_list1 = ext_list[["minx", "miny", "maxx", "maxy"]].values.tolist()
    for i, input_dem_extents in enumerate(ext_list1):
        # Prepare file name for each tile
        out_name = f"tile_{i:06}_{src_stem}.tif"
        out_path = patch_dir / out_name

        # Append variable parameters
        to_append = [input_dem_extents, out_path, source_path, 0]
        # Change list to tuple and append
        input_process_list.append(tuple(to_append))

    # Create rasters/files and save them
    if nr_processes > 1 and len(input_process_list) > 40:
        all_tiles_paths = []
        with mp.Pool(nr_processes) as p:
            realist = [p.apply_async(clip_tile, r) for r in input_process_list]
            for result in realist:
                all_tiles_paths.append(result.get())
    else:
        all_tiles_paths = [
            clip_tile(*i) for i in input_process_list
        ]

    # Build VRTs
    vrt_name = src_stem + "_tiled.vrt"
    vrt_path = build_vrt(patch_dir, vrt_name)

    return {"output_directory": patch_dir, "files_list": all_tiles_paths, "vrt_path": vrt_path}
