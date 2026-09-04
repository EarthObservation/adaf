import multiprocessing as mp
import geopandas as gpd
import pandas as pd
from adaf.create_patches import create_one_patch, create_patches_main


if __name__ == "__main__":
    # # Define paths to inputs and outputs
    # input_image = r"r:\delovno\nejc\stone_visualisations\BiH_ALS_2025_DMO_05m_slrm4inference.vrt"
    # output_dir = r"r:\delovno\nejc\stone_barrow_v5"
    #
    # # The dictionary HAST TO BE!!! in this format - at least one label and max 3 labels (there is no check)
    # # Key is name of label and value is path to vector file. Can use any label name, in example default names are used.
    # # segmentation_masks = {
    # #     "barrow": r"../test_data/test_patches/arch/barrow_segmentation_TM75.gpkg",
    # #     "enclosure": r"../test_data/test_patches/arch/enclosure_segmentation_TM75.gpkg",
    # #     "ringfort": r"../test_data/test_patches/arch/ringfort_segmentation_TM75.gpkg"
    # # }
    # segmentation_masks = {
    #     "barrow": r"r:\delovno\nejc\stone_patches\gomile_11-28-205.gpkg"
    #     #"enclosure": r"../test_data/test_patches/arch/enclosure_segmentation_TM75.gpkg",
    #     #"ringfort": r"../test_data/test_patches/arch/ringfort_segmentation_TM75.gpkg"
    # }
    # split_dataset = r"c:\test_adaf_retrain\als_ml_testna_obmocja2.gpkg"
    #
    # create_patches_main(input_image, segmentation_masks, split_dataset, output_dir)

    input_raster = r"r:\delovno\nejc\stone_visualisations\BiH_ALS_2025_DMO_05m_slrm4inference.vrt"

    df_splits = gpd.read_file(r"r:\ML podatki\learning_samples\training_samples_BiH_v7_512px\tiles-test.gpkg")

    print("Reading segmentation polygons...")

    seg_masks_dict = {
        "barrow": r"r:\ML podatki\archaeology\gomile_2025-11-28.gpkg"
        # "enclosure": r"../test_data/test_patches/arch/enclosure_segmentation_TM75.gpkg",
        # "ringfort": r"../test_data/test_patches/arch/ringfort_segmentation_TM75.gpkg"
    }

    ###################### to function #################################33
    df_list = []
    for arch_type, gpkg_path in seg_masks_dict.items():
        df = gpd.read_file(gpkg_path)
        df["arch_type"] = arch_type
        # DFM value required by dataloader
        if "DFM" not in df.columns:
            df["DFM"] = 1
        df_list.append(df)
    df_segments = gpd.GeoDataFrame(pd.concat(df_list, ignore_index=True), crs=df.crs)
    print(f"Dataset contains {len(df_segments)} labels.")
    #############################################################33

    # create_one_patch(
    #     df_splits.iloc[0].to_dict(),
    #     [df_segments],
    #     input_raster,
    #     od_labels=False
    # )

    print("Creating patches...")
    input_process_list = []
    for _, in_tile in df_splits.iterrows():
        input_process_list.append(
            (
                in_tile.to_dict(),
                [df_segments],
                input_raster,
                False
            )
        )

    with mp.Pool(10) as p:
        # _ = [p.apply_async(create_one_patch, r) for r in input_process_list]
        results = p.starmap(create_one_patch, input_process_list)  # each item is a tuple of args