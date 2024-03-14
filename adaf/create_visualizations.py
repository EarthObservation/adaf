"""
Create SLRM visualisation from DFM for ADAF ML training dataset.
Created on 26 May 2023
@author: Nejc Čož, ZRC SAZU, Novi trg 2, 1000 Ljubljana, Slovenia

Requires ADAF toolbox.
"""
from pathlib import Path

import adaf.grid_tools as gt
from adaf.adaf_vis import tiled_processing


def run_visualisations(dem_path, tile_size, save_dir, nr_processes=1):
    """Calculates visualisations from DEM and saves them into VRT (Geotiff) file.

    Uses RVT (see adaf_vis.py).

    Parameters
    ----------
    dem_path : str or pathlib.Path()
        Can be any raster file (GeoTIFF and VRT supported).
    tile_size : int
        In pixels.
    save_dir : str
        Save directory.
    nr_processes : int
        Number of processes for parallel computing.

    Returns
    -------
    dict
        A python dictionary containing results from tiling, such as paths and processing time.
    """
    # Prepare paths
    in_file = Path(dem_path)

    # We need polygon covering valid data
    valid_data_outline, _ = gt.poly_from_valid(in_file.as_posix())

    # Create reference grid, filter it and save it to disk
    tiles_extents = gt.bounding_grid(in_file.as_posix(), tile_size, tag=False)
    tiles_extents = gt.filter_by_outline(tiles_extents, valid_data_outline)

    # Run visualizations
    out_paths = tiled_processing(
        input_raster_path=in_file.as_posix(),
        extents_list=tiles_extents,
        nr_processes=nr_processes,
        save_dir=Path(save_dir)
    )

    return out_paths
