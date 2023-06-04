import glob
import os
import cv2
import numpy as np
import imageio

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
        image = image_loader(self.images[index])
        mask = image_loader(self.masks[index])/255
        n_channels = mask.shape[2]
        masks = [mask[:,:,i] for i in range(0, n_channels)]
        mask = np.stack(masks, axis=-1).astype("float32")
        return self.apply_transformations(image, mask)



def prepare(imgs_dir, masks_dir, output_dir, csv_dir, vizuelization_type, dataset_role):
    img_paths = glob.glob(os.path.join(imgs_dir, "*.tif"))
    mask_paths = glob.glob(os.path.join(masks_dir, "*.tif"))

    img_paths.sort()
    mask_paths.sort()
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    csv_file= open(csv_dir+dataset_role+".txt", 'w+')
    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        img = cv2.imread(img_path,  cv2.IMREAD_UNCHANGED)
        # Convert color format from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        mask = cv2.imread(mask_path)
        
        assert img_filename == mask_filename.replace('__segmentation_mask', '__{}'.format(vizuelization_type.replace("_v2",""))) and img.shape[:2] == mask.shape[:2]

        out_img_path = os.path.join(
            output_dir, "{}.jpg".format(img_filename)
        )
        cv2.imwrite(out_img_path, img)

        out_mask_path = os.path.join(
            output_dir, "{}_m.png".format(img_filename)
        )
        cv2.imwrite(out_mask_path, mask)
        csv_file.write(img_filename+'\n')
        # if i > 20:
        #     break

