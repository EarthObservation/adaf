import os

import rasterio

from aitlas.transforms import ResizeV2
from aitlas.transforms import MinMaxNormTranspose
from PIL import Image
import numpy as np
from osgeo import gdal
from pathlib import Path
from time import localtime, strftime


import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def make_predictions_on_single_patch_store_preds(model, image_path, image_filename, predictions_dir):
    labels = [None, 'enclosure', 'barrow', 'ringfort']
    transform = ResizeV2()
    image = Image.open(image_path)
    image = np.asarray(image)
    image = image * 255.0
    image = image.astype(np.float64)  # Convert to double
    image = Image.fromarray(image).convert('RGB')
    image = np.asarray(image) / 255.0
    predicted = model.detect_objects_v2(image, labels, transform)
    print("predicted", predicted)
    predictions_single_patch_str = ""
    labels = [None, 'enclosure', 'barrow', 'ringfort']
    for i in range(0, len(predicted['boxes'])):
        box = predicted['boxes'][i].detach().numpy()
        label = predicted['labels'][i].numpy()
        score = predicted['scores'][i].detach().numpy()
        predictions_single_patch_str += f'{round(box[0])} {round(box[1])} {round(box[2])} {round(box[3])} {labels[label]} {score}\n'
    file = open(predictions_dir + image_filename.split(".")[0] + ".txt", "w")
    file.write(predictions_single_patch_str)
    file.close()


def make_predictions_on_patches_object_detection(model, patches_folder):
    predictions_dir = patches_folder.split("/")[:-1]
    predictions_dir.append("predictions_object_detection/")
    predictions_dir = '/'.join(predictions_dir)

    print("Generating predictions:")
    if not os.path.isdir(predictions_dir):
        os.makedirs(predictions_dir)
    for file in os.listdir(patches_folder):
        print(">>> ", file)
        if file.endswith(".tif"):
            image_path = os.path.join(patches_folder, file)
            image_filename = file
            make_predictions_on_single_patch_store_preds(model, image_path, image_filename, predictions_dir)

    return predictions_dir


def make_predictions_on_patches_segmentation(model, patches_folder):
    predictions_dir = patches_folder.split("/")[:-1]
    predictions_dir.append("predictions_segmentation/")
    predictions_dir = '/'.join(predictions_dir)

    print("Generating predictions:")
    if not os.path.isdir(predictions_dir):
        os.makedirs(predictions_dir)
    for file in os.listdir(patches_folder):
        print(">>> ", file)
        if file.endswith(".tif"):
            image_path = os.path.join(patches_folder, file)
            model.predict_masks_tiff_probs(
                image_path=image_path,
                labels=['barrow', 'enclosure', 'ringfort'],
                data_transforms=MinMaxNormTranspose(),
                predictions_dir=predictions_dir
            )

    return predictions_dir


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

    def log_vis_results(self, vis_dir, vrt_path, processing_time):
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

        log_entry = (
            f"    tiling:             YES\n"
            f"    save visualization: YES\n"
            f"    tiles location:     {vis_dir}\n"
            f"    tiles count:        {tiles_count}\n"
            f"    VRT file path:      {vrt_path}\n"
            f"\n"
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

    def log_inference_inputs(self, ml_method):
        """Creates a header for a new section in the log file"""

        if ml_method == "segmentation":
            ml_method = "Semantic segmentation"

        if ml_method == "object detection":
            ml_method = ml_method.capitalize()

        log_entry = (
            f"=================================================================================\n"
            f"Inference log:\n"
            "\n"
            f"ML method: {ml_method}\n"
            f"\n"
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)


class ADAFInput:
    def __init__(self):
        self.dem_path = None
        self.vis_exist_ok = None
        self.ml_type = None
        self.model_path = None

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
                # print(f"{key} updated to {value}")
            else:
                print(f"Invalid parameter: {key}")

    # TODO: Do I need to run some checks?
    def check_data(self):
        """ Check Attributes """
        if self.dem_path == "percent":
            self.dem_path = "perc"
