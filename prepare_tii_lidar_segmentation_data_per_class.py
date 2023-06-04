import glob
import os
import cv2
import numpy as np
import imageio


def process_mask(mask, object_class, DFM_setting):
    # Band 1 - burrow, DFM 1
    # Band 2 - enclosure, DFM 1
    # Band 3 - ringfort, DFM 1
    # --
    # Band 4 - burrow, DFM 2
    # Band 5 - enclosure, DFM 2
    # Band 6 - ringfort, DFM 2
    # --
    # Band 7 - burrow, DFM 3
    # Band 8 - enclosure, DFM 3
    # Band 9 - ringfort, DFM 3
    # --
    # Band 10 - burrow, DFM 4
    # Band 11 - enclosure, DFM 4
    # Band 12 - ringfort, DFM 4

    object_classes = ['barrow', 'enclosure', 'ringfort']
    o_i = object_classes.index(object_class)
    bands = [(DFM-1)*3+o_i for DFM in DFM_setting]
    mask = mask[:,:,bands]
    mask = np.squeeze(np.max(mask, axis=2, keepdims=True))
    _, mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    return mask


def prepare_data_per_class(masks_dir, new_masks_dir, patch_size, object_class, DFM_setting):
    mask_paths = glob.glob(os.path.join(masks_dir, "*.tif"))

    if not os.path.exists(new_masks_dir):
        os.makedirs(new_masks_dir)

    for i, mask_path in enumerate(mask_paths):
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        mask = imageio.imread(mask_path)
        mask = np.transpose(mask, (1, 2, 0)) if mask.shape == (12, patch_size, patch_size) else mask

        mask = process_mask(mask, object_class, DFM_setting)
        if np.all(mask == 0):
            continue
        # save new mask
        out_mask_path = os.path.join(
            new_masks_dir, "{}.png".format(mask_filename)
        )
        cv2.imwrite(out_mask_path, mask)


for patch_size in [256, 512]:
    for dataset_role in ['train', 'test', 'validation']:
        masks_dir = f"/home/dragik/IrishArchaeology/segmentation_masks_DFM/samples_{patch_size}px/{dataset_role}/segmentation_mask"
        for object_class in ['barrow','enclosure','ringfort']:
            for DFM_setting in [[1],[2],[3],[4], [1,2], [1,2,4], [1,2,3,4]]:
                new_masks_dir =  f"/home/dragik/IrishArchaeology/samples_{patch_size}px/{dataset_role}/segmentation_masks_per_class/{object_class}/DFM_{'_'.join(str(num) for num in DFM_setting)}"
                prepare_data_per_class(masks_dir, new_masks_dir, patch_size, object_class, DFM_setting)


