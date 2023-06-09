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
    color_mapping = [[0,0,0],[255, 255, 255]]
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
        mask = (mask/255).astype(np.uint8) 
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)
        
    def load_dataset(self, data_dir, csv_file):
        vizuelization_type = data_dir.split("_")[-1]
        for mask_filename in os.listdir(csv_file):
            if os.path.isfile(os.path.join(csv_file, mask_filename)):
                mask_path = os.path.join(csv_file, mask_filename)
                image_path = f'{data_dir}/output/{mask_filename.rsplit("__", 1)[0]}__{vizuelization_type}.jpg'
                self.masks.append(mask_path)
                self.images.append(image_path)


