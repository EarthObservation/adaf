import os
from aitlas.transforms import ResizeV2
from aitlas.utils import image_loader
from aitlas.transforms import Transpose

import warnings
warnings.filterwarnings("ignore", category=UserWarning) 
warnings.filterwarnings("ignore", category=DeprecationWarning) 


def make_predictions_on_single_patch_store_preds(model, image_path, image_filename, predctions_dir):
    labels = [None, 'enclosure', 'barrow', 'ringfort']
    transform = ResizeV2()
    image = image_loader(image_path)
    predicted = model.detect_objects_v2(image, labels, transform)
    predictions_single_patch_str = ""
    labels = [None, 'enclosure', 'barrow', 'ringfort']
    for i in range(0, len(predicted['boxes'])):
        box = predicted['boxes'][i].detach().numpy()
        label = predicted['labels'][i].numpy()
        score = predicted['scores'][i].detach().numpy()
        predictions_single_patch_str += f'{round(box[0])} {round(box[1])} {round(box[2])} {round(box[3])} {labels[label]} {score}\n'
    file = open(predctions_dir+image_filename.split(".")[0]+".txt", "w")
    file.write(predictions_single_patch_str)
    file.close()

    return predictions_single_patch_str


def make_predictions_on_patches_object_detection(model, patches_folder):
    predictions_dir = patches_folder.split("/")[:-1]
    predictions_dir.append("predictions_object_detection/")
    predictions_dir = '/'.join(predictions_dir)

    print("Generating predictions:")
    if not os.path.isdir(predictions_dir):
        os.makedirs(predictions_dir)

    out_str = ""
    for file in os.listdir(patches_folder):
        print(">>> ", file)
        if file.endswith(".tif"):
            image_path = os.path.join(patches_folder, file)
            image_filename = file
            out_str += make_predictions_on_single_patch_store_preds(model, image_path, image_filename, predictions_dir)

    return out_str


def make_predictions_on_single_patch_show_detected_objects(model, image_path):
    labels = [None, 'enclosure', 'barrow', 'ringfort']
    transform = ResizeV2()
    image = image_loader(image_path)
    fig = model.detect_objects(image, labels, transform)


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
            model.predict_masks_tiff_probs(image_path = image_path, labels = ['barrow', 'enclosure', 'ringfort'], data_transforms=Transpose(), predictions_dir= predictions_dir);

   