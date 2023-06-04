from aitlas.datasets import TiiLIDARDataset
from aitlas.datasets.tii_lidar import prepare_with_filter

object_type_filter = ['ringfort']
DFM_filter = [1]

# for vizuelization_type in ['SLRM', 'DEM', 'DEM_clipped', 'e2MSTP_v2', 'e3MSTP_v2', 'e4MSTP_v2', 'VAT_flat_3B']:
for vizuelization_type in ['SLRM']:
    for dataset_role in ['train', 'test', 'validation']:
        output_dir = f"/home/dragik/IrishArchaeology/samples_256px/{dataset_role}/segmentation/seg_{vizuelization_type}_R_1/output"
        csv_dir = f"/home/dragik/IrishArchaeology/samples_256px/{dataset_role}/segmentation/seg_{vizuelization_type}_R_1/"
        imgs_dir =  f"/home/dragik/IrishArchaeology/samples_256px/{dataset_role}/images_{vizuelization_type}"
        masks_dir = f"/home/dragik/IrishArchaeology/segmentation_masks_DFM/samples_256px/{dataset_role}/segmentation_mask"
        prepare_with_filter(imgs_dir,masks_dir,output_dir, csv_dir, vizuelization_type, dataset_role, object_type_filter, DFM_filter)

    train_dataset_config = {
        "data_dir": f"/home/dragik/IrishArchaeology/samples_256px/train/segmentation/seg_{vizuelization_type}_R_1/output",
        "csv_file": f"/home/dragik/IrishArchaeology/samples_256px/train/segmentation/seg_{vizuelization_type}_R_1/train.txt"
    }
    test_dataset_config = {
        "data_dir": f"/home/dragik/IrishArchaeology/samples_256px/test/segmentation/seg_{vizuelization_type}_R_1/output",
        "csv_file": f"/home/dragik/IrishArchaeology/samples_256px/test/segmentation/seg_{vizuelization_type}_R_1/test.txt"
    }
    validation_dataset_config = {
        "data_dir": f"/home/dragik/IrishArchaeology/samples_256px/validation/segmentation/seg_{vizuelization_type}_R_1/output",
        "csv_file": f"/home/dragik/IrishArchaeology/samples_256px/validation/segmentation/seg_{vizuelization_type}_R_1/validation.txt"
    }
    train_dataset = TiiLIDARDataset(train_dataset_config)
    test_dataset = TiiLIDARDataset(test_dataset_config)
    validation_dataset = TiiLIDARDataset(validation_dataset_config)
    print(f"Total number of patches in train: {len(train_dataset)}")
    print(f"Total number of patches in test: {len(test_dataset)}")
    print(f"Total number of patches in validation: {len(validation_dataset)}")




# def process_mask_with_filter(mask, object_type_filter, DFM_filter):
#     # Band 1 - burrow, DFM 1
#     # Band 2 - enclosure, DFM 1
#     # Band 3 - ringfort, DFM 1
#     # --
#     # Band 4 - burrow, DFM 2
#     # Band 5 - enclosure, DFM 2
#     # Band 6 - ringfort, DFM 2
#     # --
#     # Band 7 - burrow, DFM 3
#     # Band 8 - enclosure, DFM 3
#     # Band 9 - ringfort, DFM 3
#     # --
#     # Band 10 - burrow, DFM 4
#     # Band 11 - enclosure, DFM 4
#     # Band 12 - ringfort, DFM 4

#     assert len(object_type_filter) > 0 and len(DFM_filter) > 0
#     mask_shape = (256, 256, 3)
#     mask_new = np.zeros(mask_shape, dtype=np.uint8)

#     for i, object_type in enumerate(['barrow', 'enclosure', 'ringfort']):
#         if not object_type in object_type_filter:
#             continue
#         DFM_mask = np.zeros((256,256), dtype=np.uint8)
#         for DFM in range(1,5):
#             if DFM not in DFM_filter:
#                 continue
#             DFM_mask =  np.maximum(DFM_mask,mask[:,:,(DFM-1)*3+i])
#         mask_new[:,:,i] = DFM_mask
#     return mask_new

# def prepare_with_filter(imgs_dir, masks_dir, output_dir, csv_dir, vizuelization_type, dataset_role, object_type_filter, DFM_filter):
#     img_paths = glob.glob(os.path.join(imgs_dir, "*.tif"))
#     mask_paths = glob.glob(os.path.join(masks_dir, "*.tif"))

#     img_paths.sort()
#     mask_paths.sort()
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     csv_file= open(csv_dir+dataset_role+".txt", 'w+')
#     for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
#         img_filename = os.path.splitext(os.path.basename(img_path))[0]
#         mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
#         img = cv2.imread(img_path,  cv2.IMREAD_UNCHANGED)
#         # Convert color format from RGB to BGR
#         img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#         img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
#         mask = imageio.imread(mask_path)
#         mask = np.transpose(mask, (1, 2, 0)) if mask.shape == (12, 256, 256) else mask
#         mask = process_mask_with_filter(mask, object_type_filter, DFM_filter)


#         assert img_filename == mask_filename.replace('__segmentation_mask', '__{}'.format(vizuelization_type.replace("_v2",""))) and img.shape[:2] == mask.shape[:2]
#         if np.all(mask == 0):
#             continue
#         out_img_path = os.path.join(
#             output_dir, "{}.jpg".format(img_filename)
#         )
#         cv2.imwrite(out_img_path, img)

#         out_mask_path = os.path.join(
#             output_dir, "{}_m.png".format(img_filename)
#         )
#         cv2.imwrite(out_mask_path, mask)
#         csv_file.write(img_filename+'\n')
#         # if i > 20:
#         #     break





