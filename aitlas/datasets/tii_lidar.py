import os
import numpy as np
import rasterio
import csv

from ..utils import image_loader
from .semantic_segmentation import SemanticSegmentationDataset



class TiiLIDARDataset(SemanticSegmentationDataset):
    url = ""

    labels = ["barrow", "enclosure", "ringfort"]
    color_mapping = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    name = "TII LIDAR"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)

    def __getitem__(self, index):
        with rasterio.open(self.images[index]) as image_tiff:
            image = image_tiff.read()
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        image = np.transpose(image, (1, 2, 0))
        with rasterio.open(self.masks[index]) as mask_tiff:
            mask = mask_tiff.read()/255
        mask = np.transpose(mask, (1,2,0))
        n_channels = mask.shape[2]
        masks = [mask[:,:,i] for i in range(0, n_channels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)
    
    # the commented code assumes that the images are in RGB jpg format
    # def load_dataset(self, data_dir, csv_file=None):
    #     if not self.labels:
    #         raise ValueError("You need to provide the list of labels for the dataset")

    #     vizuelization_type = data_dir.split("_")[-1]
    #     for mask_filename in os.listdir(csv_file):
    #         if os.path.isfile(os.path.join(csv_file, mask_filename)):
    #             mask_path = os.path.join(csv_file, mask_filename)
    #             image_path = f'{data_dir}/{mask_filename.rsplit("__", 1)[0]}__{vizuelization_type}.jpg'
    #             self.masks.append(mask_path)
    #             self.images.append(image_path)

    def load_dataset(self, data_dir, csv_file=None):
        if not self.labels:
            raise ValueError("You need to provide the list of labels for the dataset")

        vizuelization_type = '_'.join(data_dir.split("images_")[1:])
        vizuelization_type = vizuelization_type.replace('_v2', '')
        for mask_filename in os.listdir(csv_file):
            if os.path.isfile(os.path.join(csv_file, mask_filename)):
                mask_path = os.path.join(csv_file, mask_filename)
                image_path = f'{data_dir}/{mask_filename.rsplit("__", 1)[0]}__{vizuelization_type}.tif'
                self.masks.append(mask_path)
                self.images.append(image_path)


