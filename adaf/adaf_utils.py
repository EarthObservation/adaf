import os
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
            f"===============================================================================================\n"
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
            f"===============================================================================================\n"
            f"{section.capitalize()} log:\n\n"
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)

    def log_vis(self, params):
        """Creates a header for a new section in the log file"""
        if params:
            tiling = ""
        else:
            tiling = ""

        if params:
            save_vis = ""
        else:
            save_vis = ""

        log_entry = (
            f"input image: {params}\n"
            f"image size: {params}\n"
            f"CRS: {params}\n"
            f"size: {params}\n"
            f"\n"
            f"{tiling}"  # yes/no
            f"{save_vis}"  # yes/no
            f"\n"
            f"processing time: {params} seconds\n\n"
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)

    def log_inference(self, params):
        """Creates a header for a new section in the log file"""
        if params:
            custom_model = ""
        else:
            custom_model = f"path to custom model: {params}"

        if params:
            save_vis = ""
        else:
            save_vis = ""

        log_entry = (
            f"selected ML method: {params}\n"
            f"selected model: {params}\n"
            f"{custom_model}\n"
            f"\n"
            f"location of results: {params}"  # yes/no
            f"\n"
            f"processing time: {params} seconds\n\n"
        )

        with open(self.log_file_path, 'a') as log_file:
            log_file.write(log_entry)
