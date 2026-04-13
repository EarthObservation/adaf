import glob
import os
import cv2
import numpy as np
import imageio

from ..utils import image_loader
from .semantic_segmentation import SemanticSegmentationDataset
# from .schemas import TiiLIDARDatasetBinarySchema



class TiiLIDARDatasetBinary(SemanticSegmentationDataset):
    # schema = TiiLIDARDatasetBinarySchema
    url = ""

    labels = ["Background","AO"]
    color_mapping = [[255, 255, 255]]
    name = "TII LIDAR Binary"

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.images = []
        self.masks = []
        self.load_dataset(self.config.data_dir, self.config.csv_file)

    def __getitem__(self, index):
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)
        
    def load_dataset(self, data_dir, csv_file):
        for image_filename in os.listdir(data_dir):
            if os.path.isfile(os.path.join(data_dir, image_filename)):
                image_path = os.path.join(data_dir, image_filename)
                mask_path = image_path.replace("images", "annot")
                self.masks.append(mask_path)
                self.images.append(image_path)


