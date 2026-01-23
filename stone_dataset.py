# Dataset class

import os
import glob
import numpy as np
import rasterio
import imageio
import torch
from collections import defaultdict

from aitlas.utils import image_loader
from aitlas.datasets.semantic_segmentation import SemanticSegmentationDataset
from aitlas.datasets.schemas import TiiLIDARDatasetBinaryWithPreprocessingSchema

'''
    The format of rasters:
    Band 0 - barrow
    Band 1 - enclosure
    Band 2 - ringfort
    shape = [band, height, width]

    Each band is coded with the following pixel values:
    0 - background
    1 - DFM = 1
    2 - DFM = 2
    3 - DFM = 3
    4 - DFM = 4
    5 - new object (Archaeological False Positive from the ML results analysis)
'''


class StoneDatasetSegmentation(SemanticSegmentationDataset):
    schema = TiiLIDARDatasetBinaryWithPreprocessingSchema
    url = ""

    labels = ["Background","Archaeology"]
    color_mapping = [[0,0,0],[255, 255, 255]]
    name = "Stone ALS barrows"

    @staticmethod
    def get_fixed_model_config():
        """
        These model parameters are fixed and saved here for simple use.
        It is a config dictionary, user can acces it by calling this method on the class.
        """
        return {
            "num_classes": 2,
            "learning_rate": 0.0001,
            "pretrained": True,
            "use_cuda": torch.cuda.is_available(),
            "threshold": 0.5,
            "metrics": ["iou"]
        }

    def __init__(self, config):
        # now call the constructor to validate the schema and split the data
        super().__init__(config)
        self.images = []
        self.masks = []
        self.load_dataset(self.config.data_dir)

    # ---------- data access ----------
    def __getitem__(self, index):
        """
        Read image and transpose into the correct shape
        (C,H,W) -> (H,W,C)
        """
        # Open the selected image
        with rasterio.open(self.images[index]) as image_tiff:
            image = image_tiff.read()
        # Make sure it has 3 bands!
        if image.shape[0] == 1:
            image = np.repeat(image, 3, axis=0)
        # (C,H,W) -> (H,W,C)
        image = np.transpose(image, (1, 2, 0))

        # Read/construct mask (binary 0/1), (H, W) uint8
        mask = self.process_single_mask(self.masks[index])
        
        # TODO: Check what this does, I'm not sure it is needed
        masks = [(mask == v) for v, label in enumerate(self.labels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        
        return self.apply_transformations(image, mask)

    
    # ---------- mask reading (AO-aware) ----------
    def process_single_mask(self, mask_path):
        """
        mask_path: str
        DFM_setting: dict  # i.e. self.DFM_quality
        object_class: str
        object_class_band_id: int
        """
        with rasterio.open(mask_path) as src:
            if self.object_class == "AO":
                # Read all 3 bands
                arr = src.read()  # (B, H, W)
                # Keep only selected DFM quality and collapse 3 bands to 1 (boolean)
                mask_bool = np.isin(arr, self.DFM_quality).any(axis=0)  # (H, W) bool
            else:
                # Read specific band
                band = src.read(self.object_class_band_id + 1)  # (H, W)
                # Keep only selected DFM quality
                mask_bool = np.isin(band, self.DFM_quality)  # (H, W) bool
            
        return mask_bool.astype(np.uint8)  # {0,1}


    # ---------- Helper for load_dataset, cheks if mask is empty ----------
    def should_include_mask(self, mask_path, DFM_setting, keep_empty_patches, object_class, object_class_band_id):
        """
        Checks if mask is empty
        
        - Because masks have 3 bands, if training for only 1 class, some images will have empty masks (for example, image has
        only barrows, so if trainning for ringforts, this image will have an empty mask for ringfort).
        - It can also happen if we have images with only background. It is set to False when training since "empty" data can't 
        be used for training. For testing or validation, True can be used to check how the model handles empty patches.
        """
        if(keep_empty_patches):
            return True
        else:
            mask = imageio.imread(mask_path)
            # Check if transpose is necessary
            if mask.ndim == 3 and mask.shape[0] == 3:
                mask = np.transpose(mask, (1, 2, 0))
            
            mask = np.isin(mask, DFM_setting).astype(np.uint8)
            
            if (object_class!='AO'):
                mask = mask[:,:,object_class_band_id] # filter object_class
            else: 
                mask = np.squeeze(np.max(mask, axis=2, keepdims=True))
            
            return not np.all(mask == 0)

        
        
    def load_dataset(self, data_dir):
        """
        Build the dataset by pairing masks with images using the first two '__' parts.
        Assumes every mask is valid (contains at least one positive), so we SKIP
        per-mask filtering for speed/simplicity.
        """
        # Cache config
        self.object_class = self.config.object_class  # e.g., 'barrow' or 'AO'
        self.object_class_band_id = self.config.object_class_band_id  # 0/1/2 for single-class
        self.DFM_quality = [int(item) for item in self.config.DFM_quality.split(',')]  # e.g., "1" -> [1]
        self.keep_empty_patches = self.config.keep_empty_patches
        annotations_dir = self.config.annotations_dir

        # Build image index once
        image_index = self._build_image_index(
            images_dir=data_dir,
            prefer_visualisation=getattr(self.config, "visualisation_type", None),
        )

        self.images.clear()
        self.masks.clear()

        for mask_filename in os.listdir(annotations_dir):
            if not mask_filename.lower().endswith((".tif", ".tiff")):
                continue

            key = self._two_part_prefix(mask_filename)
            if not key:
                # Skip masks that don't follow 'A__B__...' naming
                continue

            image_path = image_index.get(key)
            if not image_path:
                # No matching image for this key
                continue

            mask_path = os.path.join(annotations_dir, mask_filename)

            #TODO: Also add check for empty masks, at the moment it is removed (i.e. should_include_mask())

            # Since preprocessing guarantees positives, just append
            self.images.append(image_path)
            self.masks.append(mask_path)

        if not self.images:
            raise RuntimeError(
                "No image/mask pairs loaded. Check directory paths and filename prefixes."
            )

        
        # for mask_filename in os.listdir(annotations_dir):
        #     if (mask_filename.endswith(".tif")):
        #         image_path = f'{data_dir}/{mask_filename.split("__")[0]}__{mask_filename.split("__")[1]}__{self.config.visualisation_type}.tif'
        #         if (os.path.isfile(image_path) and 
        #             os.path.isfile(os.path.join(annotations_dir, mask_filename)) and 
        #             self.should_include_mask(os.path.join(annotations_dir, mask_filename), self.DFM_quality, self.keep_empty_patches, self.object_class, self.object_class_band_id)) :
        #             mask_path = os.path.join(annotations_dir, mask_filename)
        #             self.masks.append(mask_path)
        #             self.images.append(image_path)

    
    # ---------- helpers: matching by two-part prefix ----------

    @staticmethod
    def _two_part_prefix(filename: str) -> str:
        """
        Extract the match key as the first two '__'-separated parts.
        Example: 'A__B__anything_else.tif' -> 'A__B'
        """
        parts = filename.split("__")
        return f"{parts[0]}__{parts[1]}" if len(parts) >= 2 else ""

    @staticmethod
    def _build_image_index(images_dir, prefer_visualisation):
        """
        Scan the images directory once and build an index:
            key = first two '__' parts (e.g., 'A__B')
            value = best-matching image path for that key
        If multiple images share a key, prefer those whose filename contains
        the requested visualisation token (e.g., 'slrm'); else pick a stable default.
        """
        idx = defaultdict(list)

        # Collect candidate image files (.tif/.tiff, any case)
        for ext in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
            for p in glob.glob(os.path.join(images_dir, ext)):
                key = StoneDatasetSegmentation._two_part_prefix(os.path.basename(p))
                if key:
                    idx[key].append(p)

        # Choose the "best" path per key
        best = {}
        pref = (prefer_visualisation or "").lower()
        for key, paths in idx.items():
            if pref:
                cand = [p for p in paths if pref in os.path.basename(p).lower()]
                if cand:
                    best[key] = sorted(cand)[0]
                    continue
            # Fallback: deterministic choice: shortest filename, then lexicographic
            best[key] = sorted(
                paths,
                key=lambda p: (len(os.path.basename(p)), os.path.basename(p).lower())
            )[0]
        return best


if __name__ == '__main__':

    train_data = r"r:\delovno\nejc\test_adaf_retrain\samples\train"
    train_mask = r"r:\delovno\nejc\test_adaf_retrain\labels\segmentation_masks\train"

    batch_size = 16
    num_workers = 2
    object_class = "barrow"
    object_class_band_id = 1
    visualisation_type = "SLRM"


    train_dataset_config = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "object_class": object_class,
        "object_class_band_id": object_class_band_id,
        "visualisation_type": visualisation_type,
        "DFM_quality": '1',
        "shuffle": True,
        "keep_empty_patches": False,
        "data_dir": train_data,
        "annotations_dir": train_mask,
        "joint_transforms": ["aitlas.transforms.FlipHVRandomRotate"],
        "transforms": ["aitlas.transforms.Transpose"],
        "target_transforms": ["aitlas.transforms.Transpose"]
    }

    train_dataset = StoneDatasetSegmentation(train_dataset_config)
