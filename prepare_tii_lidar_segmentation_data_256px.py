from aitlas.datasets import TiiLIDARDataset

from aitlas.datasets.tii_lidar import prepare
import matplotlib.pyplot as plt
import cv2

# vizuelization_type = "SLRM"
# for vizuelization_type in ['SLRM', 'DEM', 'DEM_clipped', 'e2MSTP_v2', 'e3MSTP_v2', 'e4MSTP_v2', 'VAT_flat_3B']:
for vizuelization_type in ['e4MSTP']:
    for dataset_role in ['train', 'test', 'validation']:
        output_dir = f"/home/dragik/IrishArchaeology/samples_256px/{dataset_role}/segmentation/seg_{vizuelization_type}/output"
        csv_dir = f"/home/dragik/IrishArchaeology/samples_256px/{dataset_role}/segmentation/seg_{vizuelization_type}/"
        imgs_dir =  f"/home/dragik/IrishArchaeology/samples_256px/{dataset_role}/images_{vizuelization_type}"
        masks_dir = f"/home/dragik/IrishArchaeology/TII_ADAF/samples_256px/{dataset_role}/segmentation_mask"
        prepare(imgs_dir,masks_dir,output_dir, csv_dir, vizuelization_type, dataset_role)

    train_dataset_config = {
        "data_dir": f"/home/dragik/IrishArchaeology/samples_256px/train/segmentation/seg_{vizuelization_type}/output",
        "csv_file": f"/home/dragik/IrishArchaeology/samples_256px/train/segmentation/seg_{vizuelization_type}/train.txt"
    }
    test_dataset_config = {
        "data_dir": f"/home/dragik/IrishArchaeology/samples_256px/test/segmentation/seg_{vizuelization_type}/output",
        "csv_file": f"/home/dragik/IrishArchaeology/samples_256px/test/segmentation/seg_{vizuelization_type}/test.txt"
    }
    validation_dataset_config = {
        "data_dir": f"/home/dragik/IrishArchaeology/samples_256px/validation/segmentation/seg_{vizuelization_type}/output",
        "csv_file": f"/home/dragik/IrishArchaeology/samples_256px/validation/segmentation/seg_{vizuelization_type}/validation.txt"
    }
    train_dataset = TiiLIDARDataset(train_dataset_config)
    test_dataset = TiiLIDARDataset(test_dataset_config)
    validation_dataset = TiiLIDARDataset(validation_dataset_config)
    print(f"Total number of patches in train: {len(train_dataset)}")
    print(f"Total number of patches in test: {len(test_dataset)}")
    print(f"Total number of patches in validation: {len(validation_dataset)}")