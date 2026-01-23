"""
Create SLRM visualisation from DFM for ADAF ML training dataset.
Created on 26 May 2023
@author: Nejc Čož, ZRC SAZU, Novi trg 2, 1000 Ljubljana, Slovenia

Requires ADAF toolbox.
"""
from pathlib import Path

import pandas as pd

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

    # Create grid of tiles (in memory GeoDataFrame)
    tiles_extents = gt.bounding_grid(in_file.as_posix(), tile_size, tag=False)

    # Add cell_ID and extents columns. Extents are (L, B, R, T).
    out_grid = tiles_extents.reset_index()
    out_grid = out_grid.rename(columns={'index': 'tile_ID'})
    out_grid["extents"] = out_grid.bounds.apply(lambda x: (x.minx, x.miny, x.maxx, x.maxy), axis=1)

    # Because tuple can't be saved into file, split extents into separate columns
    out_grid[["minx", "miny", "maxx", "maxy"]] = pd.DataFrame(out_grid['extents'].tolist(), index=out_grid.index)
    out_grid = out_grid.drop(columns=['extents'])

    # Run visualizations
    out_paths = tiled_processing(
        input_raster_path=in_file.as_posix(),
        extents_list=out_grid,
        nr_processes=nr_processes,
        save_dir=Path(save_dir)
    )

    return out_paths
