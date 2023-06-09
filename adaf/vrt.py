from pathlib import Path

from osgeo import gdal
import glob
import time
import os
import geopandas as gpd


def make_vrts(dir):
    """Finds all folders with DEM subfolders, containing tiles, and creates a VRT file.

    dir has to be path in this format:
    z:\\TII_ADAF\\3_tiled_dems
    """

    # Locations of datasets (contains subdirs <DATASET NAME>, that have subdirs DEM)
    folders_loc = glob.glob(dir + "\\*\\", recursive=True)

    # Select all folders that don't have VRT file
    todo = [fold for fold in folders_loc if not glob.glob(fold + "*.vrt")]

    # Create missing VRTS
    for ds in todo:
        if glob.glob(ds + "DEM"):
            t1 = time.time()

            # Name of the dataset
            subdir = os.path.basename(os.path.dirname(ds))

            # Find all tiles for mosaic
            tif_list = glob.glob(ds + "\\DEM\\*.tif")
            print("There are", len(tif_list), "TIFs in", subdir)

            # Create output name for VRT
            # TODO: Change how the name is created (only add suffixes!)
            resolution = os.path.basename(tif_list[0])[:-4].split("_")[-1]  # Get resolution from first TIF file
            vrt_path = os.path.join(ds, f"{subdir}_dem_{resolution}.vrt")

            # Build VRT
            vrt_options = gdal.BuildVRTOptions()
            my_vrt = gdal.BuildVRT(vrt_path, tif_list, options=vrt_options)
            my_vrt = None

            t1 = time.time() - t1
            print(f"Done in {t1:02} sec.")


def shp2gpkg(dir):
    folders_loc = glob.glob(dir + "\\*\\*.shp", recursive=True)

    for file in folders_loc:
        print(os.path.basename(file).split("_")[0])

        shp = gpd.read_file(file)

        file_name = file.split("_LAZ_")[0] + "_dem_05_validDataMask.gpkg"
        file_name = file_name.replace("_dems", "_dems_GPKG")
        os.makedirs(os.path.dirname(file_name), exist_ok=True)

        # Make GPKG if it doesn't exist
        if not glob.glob(file_name):
            # shp.drop(columns=["FID"]).to_file(file_name, driver="GPKG")
            shp.to_file(file_name, driver="GPKG")


def repair_crs_paths(work_dir):
    for path in Path(work_dir).rglob('*_TM75.tif'):

        print(path.name)

        dem_file = Path(str(path).replace("_TM75", ""))
        dem_file.rename(dem_file.parents[0] / path.name.replace("_TM75", "_wrongCRS"))

        path.rename(path.parents[0] / path.name.replace("_TM75", ""))


def build_vrt(ds_dir, vrt_name):
    ds_dir = Path(ds_dir)
    vrt_path = ds_dir.parents[0] / vrt_name
    tif_list = glob.glob(Path(ds_dir / "*.tif").as_posix())

    vrt_options = gdal.BuildVRTOptions()
    my_vrt = gdal.BuildVRT(vrt_path.as_posix(), tif_list, options=vrt_options)
    my_vrt = None

    return vrt_path


def build_vrt_from_list(tif_list, vrt_path):
    vrt_options = gdal.BuildVRTOptions()
    my_vrt = gdal.BuildVRT(vrt_path.as_posix(), tif_list, options=vrt_options)
    my_vrt = None

    return vrt_path


if __name__ == "__main__":

    # # VRT
    make_vrts("n:\\Ziga\\G-LiHT\\ALS")

    # shp2gpkg
    # shp2gpkg("d:\\TII_lidar\\*")

    # Rename files with wrong CRS
    # repair_crs_paths("z:\\TII_ADAF\\2_small_dems")

