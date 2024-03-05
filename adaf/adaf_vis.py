"""
ADAF - visualisations module

Creates visualisations from DEM and stores them as VRT.

DEMO VERSION: At the moment it only creates NORMALISED SLRM visualisations (min/max -0.5/0.5)

"""
import logging
import multiprocessing as mp
import os
import time
from math import ceil
from pathlib import Path

import numpy as np
import rasterio
import rvt.blend
import rvt.default
import rvt.vis
from rasterio.windows import from_bounds
from rvt.blend_func import normalize_image

from adaf.adaf_utils import build_vrt


def tiled_processing(
        input_vrt_path,
        ext_list,
        nr_processes=7,
        ll_dir=None
):
    # Start timer
    t0 = time.time()

    # This is the main dataset folder (where DEM file is located)
    output_dir_path = Path(input_vrt_path).parent

    # Get resolution of the dataset, because buffering is dependent on resolution!
    res = get_resolution(input_vrt_path)

    # DEFAULTS (for low-level visualizations)
    # ================================================
    # Default 1 (Slope, SLRM, MSTP, SVF, Openness +/-)
    default_1 = rvt.default.DefaultValues()
    # slrm  -  10 m (divide by pixel size!), can't be smaller than 10 pixels
    default_1.slrm_rad_cell = ceil(10 / res) if res < 1 else 10

    # Prepare folder for saving results
    if ll_dir:
        low_levels_dir = ll_dir
        low_levels_dir.mkdir(parents=True, exist_ok=True)
    else:
        # If not specified, save results next to the input file
        low_levels_dir = output_dir_path

    # Prepare for multiprocessing
    const_params = [
        default_1,           # const 1
        input_vrt_path,      # const 3
        low_levels_dir       # const 4
    ]

    # Get basename of VRT file, required for building output name
    input_process_list = []
    # Extents are calculated HERE!
    ext_list1 = ext_list[["minx", "miny", "maxx", "maxy"]].values.tolist()
    for i, input_dem_extents in enumerate(ext_list1):
        # --> USE THIS IF YOU WANT TO ADD INDEX TO FILENAME
        # # tile_id = ext_list.tile_ID.iloc[i]
        # # out_name = f"{left:.0f}_{bottom:.0f}_rvt_id-{tile_id}.tif"
        #

        # Prepare file name as left-bottom coordinates
        left = ext_list.minx.iloc[i]
        bottom = ext_list.miny.iloc[i]
        out_name = f"{left:.0f}_{bottom:.0f}_rvt.tif"

        # Append variable parameters to the list for multiprocessing
        to_append = const_params.copy()  # Copy the constant parameters
        # Append variable parameters
        to_append.append(input_dem_extents)  # var 1
        to_append.append(out_name)  # var 2
        to_append.append(i)  # var 3
        # Change list to tuple
        input_process_list.append(tuple(to_append))

    # # DEBUG: RUN SINGLE INSTANCE
    # one_instance = input_process_list[13]
    # res = compute_save_low_levels(*one_instance)
    # logging.debug(res)

    # multiprocessing
    skipped_tiles = []
    with mp.Pool(nr_processes) as p:
        realist = [p.apply_async(process_one_tile, r) for r in input_process_list]
        for result in realist:
            pool_out = result.get()
            # Check if tile was all NaN's (remove it from REFGRID!)
            if pool_out[0] == 1:
                logging.debug("Skipped (tile_ID:", pool_out[1], ");", pool_out[2])
                skipped_tiles.append(pool_out[1])
            else:
                logging.debug("tile_ID:", pool_out[1], ";", pool_out[2])

    # # Remove tiles from REFGRID if any (that was the case in Noise mapping)
    # if skipped_tiles:
    #     ext_list = ext_list[~ext_list["tile_ID"].isin(skipped_tiles)]
    #     refg_pth = list(output_dir_path.glob("*_refgrid*"))[0]  # Find path to "refgrid" file
    #     ext_list.to_file(refg_pth, driver="GPKG")

    # Prepare list with all output tiles paths
    all_tiles_paths = [pth[3].as_posix() for pth in input_process_list]

    # Build VRTs
    # TODO: hardcoded for slrm, change if different vis will be available
    #  FILE NAMING IS DONE HERE
    ds_dir = low_levels_dir / 'slrm'
    vrt_name = Path(input_vrt_path).stem + "_" + Path(ds_dir).name + ".vrt"
    out_path = build_vrt(ds_dir, vrt_name)
    logging.debug("  - Created:", out_path)

    t1 = time.time() - t0
    logging.debug(f"Done with computing low-level visualizations in {round(t1/60, ndigits=None)} min.")

    return {"output_directory": ds_dir, "files_list": all_tiles_paths, "vrt_path": out_path, "processing_time": t1}


# Function which is multiprocessing
def process_one_tile(
        default_1,
        vrt_path,
        low_levels_dir,
        input_dem_extents,
        dem_name,
        tile_id
):
    # We only have SLRM, but potentially other visualizations can be added
    buffer_dict = {
        "slrm": default_1.slrm_rad_cell
    }

    # Select the largest required buffer
    max_buff = max(buffer_dict, key=buffer_dict.get)
    buffer = buffer_dict[max_buff]

    # Read array into RVT dictionary format
    dict_arrays = get_raster_vrt(vrt_path, input_dem_extents, buffer)

    # Add default path
    dict_arrays["default_path"] = dem_name

    # Change nodata value to np.nan, to avoid problems later
    dict_arrays["array"][dict_arrays["array"] == dict_arrays["no_data"]] = np.nan
    dict_arrays["no_data"] = np.nan

    # Then check, if output slice (w/o buffer) is all NaNs, then skip this tile if yes
    if (dict_arrays["array"][buffer:-buffer, buffer:-buffer] == np.nan).all():
        # If all NaNs encountered, output the tile ID
        return 1, tile_id, f"Skipping, all NaNs in: {dem_name}"

    # --- START VISUALIZATION WITH RVT ---

    for vis_type in buffer_dict:
        # Obtain buffer for current visualization type
        arr_buff = buffer_dict[vis_type]
        # Slice raster to minimum required size
        arr_slice = buffer_dict[max_buff] - arr_buff
        if arr_slice == 0:
            sliced_arr = dict_arrays["array"]
        else:
            sliced_arr = dict_arrays["array"][arr_slice:-arr_slice, arr_slice:-arr_slice]

        # Run visualization
        if vis_type == "slrm":
            slrm = default_1.get_slrm(sliced_arr)
            out_slrm = normalize_image(
                visualization="slrm",
                image=slrm.squeeze(),
                min_norm=-0.5,
                max_norm=0.5,
                normalization="value"
            )
            out_slrm[np.isnan(out_slrm)] = 0
            vis_out = {
                vis_type: out_slrm
            }
            # Determine output name
            vis_paths = {
                vis_type: low_levels_dir / vis_type / default_1.get_slrm_file_name(dem_name)
            }
        else:
            raise ValueError("Wrong vis_type in the visualization for loop")

        # Save visualization to file
        for i in vis_out:
            # Slice away buffer
            if arr_buff == 0:
                arr_out = vis_out[i]
            else:
                arr_out = vis_out[i][..., arr_buff:-arr_buff, arr_buff:-arr_buff]
            # Make sure the dimensions of array are correct
            if arr_out.ndim == 2:
                arr_out = np.expand_dims(arr_out, axis=0)
            # Determine output name
            arr_save_path = vis_paths[i]
            os.makedirs(os.path.dirname(arr_save_path), exist_ok=True)

            # Save using rasterio
            out_profile = dict_arrays["profile"].copy()
            out_profile.update(dtype=arr_out.dtype,
                               count=arr_out.shape[0],
                               nodata=0)  # was NaN, use 0 for SLRM in ADAF
            with rasterio.open(arr_save_path, "w", **out_profile) as dst:
                dst.write(arr_out)

    return 0, tile_id, f"Finished processing: {dem_name}"


def get_raster_vrt(vrt_path, extents, buffer):
    """
    Extents have to be transformed into rasterio Window object, it is passed into the function as tuple.
    (left, bottom, right, top)

    Parameters
    ----------
    vrt_path : str
        Path to raster file. Can be any rasterio readable format.
    extents : tuple
        Extents to be read (left, bottom, right, top).
    buffer : int
        Buffer in pixels.

    Returns
    -------
        A dictionary containing the raster array and all the required metadata.

    """
    with rasterio.open(vrt_path) as vrt:
        # Read VRT metadata
        vrt_res = vrt.res
        vrt_nodata = vrt.nodata
        vrt_transform = vrt.transform
        vrt_crs = vrt.crs

        # ADD BUFFER TO EXTENTS (LBRT) - transform pixels to meters!
        buffer_m = buffer * vrt_res[0]
        buff_extents = (
            extents[0] - buffer_m,
            extents[1] - buffer_m,
            extents[2] + buffer_m,
            extents[3] + buffer_m
        )

        # Pack extents into rasterio's Window object
        buff_window = from_bounds(*buff_extents, vrt_transform)
        orig_window = from_bounds(*extents, vrt_transform)

        # Read windowed array (with added buffer)
        # boundless - if window falls out of bounds, read it and fill with NaNs
        win_array = vrt.read(window=buff_window, boundless=True)

        # Save transform object of both extents (original and buffered)
        buff_transform = vrt.window_transform(buff_window)
        orig_transform = vrt.window_transform(orig_window)

    # For raster with only one band, remove first axis from the array (RVT requirement)
    if win_array.shape[0] == 1:
        win_array = np.squeeze(win_array, axis=0)

    # Prepare output metadata profile
    out_profile = {
        'driver': 'GTiff',
        'nodata': None,
        'width':  win_array.shape[1] - 2 * buffer,
        'height':  win_array.shape[0] - 2 * buffer,
        'count':  1,
        'crs': vrt_crs,
        'transform': orig_transform,
        "compress": "lzw"
    }

    output = {
        "array": win_array,
        "resolution": vrt_res,
        "no_data": vrt_nodata,
        "buff_transform": buff_transform,
        "orig_transform": orig_transform,
        "crs": vrt_crs,
        "profile": out_profile
    }

    return output


def get_resolution(path):
    with rasterio.open(path) as src:
        resolution = src.res[0]

    return resolution
