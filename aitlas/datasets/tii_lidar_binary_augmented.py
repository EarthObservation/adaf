import glob
import os
import cv2
import numpy as np
import rasterio
import imageio

from ..utils import image_loader
from .semantic_segmentation import SemanticSegmentationDataset
# from .schemas import TiiLIDARDatasetBinarySchema



class TiiLIDARDatasetBinaryAugmented(SemanticSegmentationDataset):
    # schema = TiiLIDARDatasetBinarySchema
    url = ""

    labels = ["Background","AO"]
    color_mapping = [[0,0,0],[255, 255, 255]]
    name = "TII LIDAR Binary"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.images = []
        self.masks = []
        self.load_dataset(self.config.data_dir, self.config.csv_file)

    def __getitem__(self, index):
        # image = image_loader(self.images[index])
        with rasterio.open(self.images[index]) as image_tiff:
            image = image_tiff.read()
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        image = np.transpose(image, (1, 2, 0))
        mask = image_loader(self.masks[index])
        mask = (mask/255).astype(np.uint8) 
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)
        
    def load_dataset(self, data_dir, csv_file):
        is_train = 'train' in csv_file
        vizuelization_type = 'SLRM'
        for mask_filename in os.listdir(csv_file):
            if os.path.isfile(os.path.join(csv_file, mask_filename)) and (".DS_Store" not in mask_filename):
                if is_train:
                    mask_path = os.path.join(csv_file, mask_filename)
                    image_path = f'{data_dir}/{mask_filename.split("__")[0]}__{mask_filename.split("__")[1]}__{vizuelization_type}__{(mask_filename.split("segmentation_mask__")[1].split(".")[0])}.tif'
                    self.masks.append(mask_path)
                    self.images.append(image_path)
                else:
                    mask_path = os.path.join(csv_file, mask_filename)
                    image_path = f'{data_dir}/{mask_filename.rsplit("__", 1)[0]}__{vizuelization_type}.tif'
                    self.masks.append(mask_path)
                    self.images.append(image_path)


