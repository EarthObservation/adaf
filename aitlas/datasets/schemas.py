from marshmallow import fields, validate

from ..base.schemas import BaseDatasetSchema


class MatDatasetSchema(BaseDatasetSchema):
    mat_file = fields.String(
        missing=None, description="mat file on disk", example="./data/dataset.mat",
    )
    mode = fields.String(
        missing="train",
        description="Which split to use, train or test.",
        example="train",
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )
    download = fields.Bool(
        missing=False, description="Whether to download the dataset", example=True
    )


class NPZDatasetSchema(BaseDatasetSchema):
    npz_file = fields.String(
        missing=None, description="npz file on disk", example="./data/dataset.npz",
    )
    mode = fields.String(
        missing="train",
        description="Which split to use, train or test.",
        example="train",
    )
    labels = fields.List(
        fields.String,
        missing=None,
        required=False,
        description="List of labels",
    )


class ClassificationDatasetSchema(BaseDatasetSchema):
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )


class SegmentationDatasetSchema(BaseDatasetSchema):
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv",
    )

# class TiiLIDARDatasetBinarySchema(BaseDatasetSchema):
#     data_dir = fields.String(
#         missing="/", description="Dataset path on disk", example="./data/BigEarthNet/"
#     )
#     masks_dir = fields.String(
#         missing="/", description="Segmentation masks path on disk", example="./segmentation_masks/BigEarthNet/"
#     )
#     vizuelization_type = fields.String(
#         missing=None, description="Vizuelization type name", example="SLRM"
#     )


class ObjectDetectionPascalDatasetSchema(BaseDatasetSchema):
    imageset_file = fields.String(
        missing="/",
        description="File with the image ids in the set",
        example="./data/DIOR/train.txt",
    )
    image_dir = fields.String(
        missing="/", description="Folder to the images on disk", example="./data/DIOR/"
    )
    annotations_dir = fields.String(
        missing="/",
        description="Folder with the XML annotations in VOC format",
        example="./data/DIOR/Annons/",
    )


class ObjectDetectionCocoDatasetSchema(BaseDatasetSchema):
    data_dir = fields.String(
        missing="/", description="Dataset path on disk", example="./data/DIOR/"
    )
    json_file = fields.String(
        missing=None,
        description="JSON Coco file format on disk",
        example="./data/train.json",
    )
    hardcode_background = fields.Bool(
        missing=True, description="Do we need to hardcode the background as a class?"
    )


class BigEarthNetSchema(BaseDatasetSchema):
    csv_file = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    lmdb_path = fields.String(missing=None, description="Path to the lmdb storage")
    data_dir = fields.String(
        missing=None, description="Dataset path on disk", example="./data/BigEarthNet/"
    )
    selection = fields.String(
        missing="rgb", description="Read RGB channels or 13 channels", example="all/rgb"
    )
    version = fields.String(
        missing="19 labels",
        description="43 or 19 labels",
        example="43 labels/19 labels",
    )
    import_to_lmdb = fields.Bool(
        missing=False, description="Should the data be moved to LMDB"
    )
    bands10_mean = fields.List(
        fields.Float,
        missing=(429.9430203, 614.21682446, 590.23569706),
        required=False,
        description="List of mean values for the 3 channels",
    )
    bands10_std = fields.List(
        fields.Float,
        missing=(572.41639287, 582.87945694, 675.88746967),
        required=False,
        description="List of std values for the 3 channels",
    )


class SpaceNet6DatasetSchema(BaseDatasetSchema):
    orients = fields.String(
        required=False,
        example="path/to/data/train/AOI_11_Roterdam/SummaryData/SAR_orientations.csv",
        description="Absolute path pointing to the SAR orientations text file "
        "(output of the pre-processing task",
    )
    root_directory = fields.String(
        required=False,
        example="path/to/data/train/AOI_11_Rotterdam/",
        description="Root directory for the raw SpaceNet6 data set",
    )
    start_val_epoch = fields.Int(
        required=False,
        description="From which epoch should the validation period start",
    )
    # Train & val
    folds_path = fields.String(
        required=False,
        example="path/to/results/folds",
        description="Path to the fold csv files",
    )
    segmentation_directory = fields.String(
        required=False,
        example="path/to/results/segmentation",
        description="Source directory with the target segmentation masks",
    )
    gt_csv = fields.String(
        required=False,
        description="Source file containing the ground truth segmentation data on the buildings",
    )
    pred_csv = fields.String(
        required=False,
        description="Destination file for saving the predictions from the current fold",
    )
    pred_folder = fields.String(
        required=False,
        description="Destination directory for saving the predictions from all folds",
    )
    edge_weight = fields.Int(
        required=False, description="Weight for the building edges pixels"
    )
    contact_weight = fields.Int(
        required=False, description="Weight for the building contact pixels"
    )
    # Test
    test_directory = fields.String(
        required=False,
        example="path/to/data/train/AOI_11_Rotterdam/",
        description="Root directory for the raw SpaceNet6 data set",
    )
    merged_pred_dir = fields.String(
        required=False,
        example="path/to/data/train/AOI_11_Rotterdam/",
        description="Destination directory for merging the predictions from all folds",
    )
    solution_file = fields.String(
        required=False,
        example="path/to/data/results/solution.csv",
        description="SpaceNet6-compliant csv destination file used for grading the challenge",
    )
    # Prepare
    num_folds = fields.Int(
        required=False, missing=10, description="Number of fold splits for the data set"
    )
    orients_output = fields.String(
        required=False,
        example="path/to/data/train/AOI_11_Roterdam/SummaryData/SAR_orientations.txt",
        description="Absolute path pointing to the output SAR orientations csv file",
    )
    num_threads = fields.Int(
        required=False,
        missing=1,
        description="Number of threads for parallel execution",
    )
    edge_width = fields.Int(
        required=False,
        default=3,
        description="Width of the edge of buildings (in pixels)",
    )
    contact_width = fields.Int(
        required=False,
        default=9,
        description="Width of the contact between (in pixels)",
    )
    folds_dir = fields.String(
        required=False,
        example="path/to/results/folds",
        description="Source directory with the fold csv files",
    )


class BreizhCropsSchema(BaseDatasetSchema):
    regions = fields.List(
        fields.String,
        required=True,
        description="Brittany region (frh01..frh04)",
        example="['frh01','frh01']",
    )

    root = fields.String(
        required=True,
        description="Dataset path on disk",
        example="./breizhcrops_dataset",
    )
    year = fields.Integer(
        missing=2017, description="year", validate=validate.OneOf([2017, 2018])
    )
    filter_length = fields.Integer(missing=0, description="filter_length")
    level = fields.String(
        required=True,
        description="L1C or L2A",
        example="L1C",
        validate=validate.OneOf(["L1C", "L2A"]),
    )
    verbose = fields.Bool(missing=False, description="verbose")  # change to true
    load_timeseries = fields.Bool(missing=True, description="load_timeseries")
    recompile_h5_from_csv = fields.Bool(
        missing=False, description="recompile_h5_from_csv"
    )
    preload_ram = fields.Bool(missing=False, description="preload_ram")


class CropsDatasetSchema(BaseDatasetSchema):
    csv_file_path = fields.String(
        missing=None, description="CSV file on disk", example="./data/train.csv"
    )
    root = fields.String(
        required=True, description="Dataset path on disk", example="./slovenia-crops"
    )
    verbose = fields.Bool(missing=False, description="verbose")
    level = fields.String(
        missing="L1C",
        description="L1C or L2A",
        example="L1C",
        validate=validate.OneOf(["L1C", "L2A"]),
    )
    regions = fields.List(
        fields.String,
        required=True,
        description="Brittany region (frh01..frh04) or train/val/test",
        example="['frh01','frh01']",
    )


class So2SatDatasetSchema(BaseDatasetSchema):
    h5_file = fields.String(
        required=True, description="H5 file on disk", example="./data/train.h5"
    )
