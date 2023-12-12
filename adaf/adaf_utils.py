import logging
import os
import warnings
from pathlib import Path
from time import localtime, strftime

import numpy as np
import rasterio
from aitlas.transforms import ResizeV2
from aitlas.transforms import Transpose
from osgeo import gdal

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
    """Generates predictions on patches (the model performs binary semantic segmentation)


    Parameters
    ----------
    model
    label
    patches_folder
    predictions_dir : str or object
        Optional - user can specify a custom folder. Otherwise, a folder called "predictions_segmentation_{label}" is
        created.

    Returns
    -------

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
    vrt_options = gdal.BuildVRTOptions()
    my_vrt = gdal.BuildVRT(vrt_path.as_posix(), tif_list, options=vrt_options)
    my_vrt = None

    return vrt_path


def build_vrt(ds_dir, vrt_name):
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
        """Creates a header for a new section in the log file"""
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
        """Creates a header for a new section in the log file"""

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

    def log_inference_inputs(self, ml_method, ml_labels, ml_model="ADAF"):
        """Creates a header for a new section in the log file"""

        if ml_method == "segmentation":
            ml_method = "Semantic segmentation"

        if ml_method == "object detection":
            ml_method = ml_method.capitalize()

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
            f"    ML model: {ml_model}\n"
            f"    Selected classes: * {log_labels}"
            f"\n"
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)

    def log_inference_results(self, vector_path, processing_time, list_to_raw_files):
        """Creates a header for a new section in the log file"""

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

        log_entry = (
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
