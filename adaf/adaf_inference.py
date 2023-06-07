import rasterio
import numpy as np
import geopandas as gpd
from shapely.geometry import box
from shapely.affinity import translate
# from ipywidgets import interact, widgets
from pathlib import Path
import grid_tools as gt
from rasterio.windows import from_bounds
from rasterio.warp import reproject, Resampling
from adaf_vis import tiled_processing
import multiprocessing as mp
import shutil


def create_patches(source_path, patch_size_px, save_dir, nr_processes):
    multi_p = False

    # Prepare paths
    source_path = Path(source_path)
    ds_path = source_path.parent
    # Folder for patches
    patch_dir = save_dir / "ml_patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    # Read metadata from source image
    with rasterio.open(source_path) as src:
        ds_res = src.res[0]
        ds_crs = src.crs
        ds_bounds = src.bounds

    # -------- CREATE GRID --------
    stride = 0.5
    patch_size = (patch_size_px * ds_res, patch_size_px * ds_res)
    stagger = stride * patch_size[0]

    grid = uniform_grid(ds_bounds, ds_crs, patch_size, stagger)

    # Filter grid (remove patches that fall outside scanned area)
    vdm_path = list(ds_path.glob("*_validDataMask.*"))[0]
    vdm = gpd.read_file(vdm_path)
    # Make sure Valid Data Mask is in a correct CRS
    if vdm.crs.to_epsg() != ds_crs.to_epsg():
        vdm = vdm.to_crs(ds_crs)

    vdm_filter = grid["geometry"].intersects(vdm.geometry[0])
    grid = grid[vdm_filter].reset_index(drop=True)

    if multi_p:
        # Multiprocessing run
        # TODO: Multiprocessing not working?!
        # Create rasters/files and save them
        input_process_list = []
        for i, one_tile in grid.iterrows():
            save_file = patch_dir / f"ml_patch_{i+1:06d}.tif"
            out_nodata = 0
            resample = False
            input_process_list.append(
                (
                    one_tile["geometry"],
                    save_file,
                    source_path,
                    out_nodata,
                    resample
                )
            )
        with mp.Pool(nr_processes) as p:
            realist = [p.apply_async(clip_tile, r) for r in input_process_list]
    else:
        realist = [
            clip_tile(
                p["geometry"],
                patch_dir / f"ml_patch_{i + 1:06d}.tif",
                source_path,
                out_nodata=0,
                resample=False
            ) for i, p in grid.iterrows()
        ]
        # # -------- CLIP TILES --------
        # for i, one_tile in grid.iterrows():
        #     # # Source raster path
        #     # src_pth = list(ds_path.glob(f"*{vis_type}.vrt"))[0]
        #     out_nodata = 0
        #     # out_nodata = -999
        #
        #     save_file = patch_dir / f"ml_patch_{i+1:06d}.tif"
        #
        #     # Create rasters/files and save them
        #     clip_result = clip_tile(
        #         one_tile["geometry"],
        #         save_file,
        #         source_path,
        #         out_nodata=out_nodata,
        #         resample=False
        #     )
        #     print(clip_result)

    output = "#  - Finished creating " + str(len(grid)) + " patches for " + ds_path.stem

    return output


def uniform_grid(extents, crs, spacing_xy, stagger=None):
    """ Creates uniform grid over the total extents.

    Parameters
    ----------
    extents
    crs
    spacing_xy
    stagger : float
        Distance to use for translating the cell for creation of staggered grid.
        Use None if grid should not be staggered.

    Returns
    -------

    """
    # total area for the grid [xmin, ymin, xmax, ymax]
    x_min, y_min, x_max, y_max = extents  # gdf.total_bounds
    tile_w, tile_h = spacing_xy

    # Target Aligned Pixels
    x_min = np.floor(x_min / tile_w) * tile_w
    # bottom = np.floor(extents.bottom / tile_h) * tile_h
    # right = np.ceil(extents.right / tile_w) * tile_w
    y_max = np.ceil(y_max / tile_h) * tile_h
    _, bottom, right, _ = extents  # ONLY TOP-LEFT NEEDS TO BE ROUNDED

    grid_cells = []
    for x0 in np.arange(x_min, x_max, tile_w):
        for y0 in np.arange(y_max, y_min, -tile_h):
            # bounds
            x1 = x0 + tile_w
            y1 = y0 - tile_h
            cell_1 = box(x0, y0, x1, y1)
            grid_cells.append(cell_1)
            if stagger:
                cell_2 = translate(cell_1, stagger, stagger)
                grid_cells.append(cell_2)

    # Generate GeoDataFrame
    grid = gpd.GeoDataFrame(grid_cells, columns=['geometry'], crs=crs)

    return grid


def clip_tile(poly, file_path, src_path, out_nodata=0, resample=False):
    """Clips a single tile from a source raster."""
    with rasterio.open(src_path) as src:
        bounds = poly.bounds
        orig_window = from_bounds(*bounds, src.transform)

        out_image = src.read(window=orig_window, boundless=True)
        out_transform = src.window_transform(orig_window)
        out_profile = src.profile.copy()
        src_nodata = src.nodata

    if resample:
        out_image, out_transform = reproject(
            source=out_image,
            src_crs=src.crs,
            dst_crs=src.crs,
            src_nodata=src_nodata,
            dst_nodata=src_nodata,
            src_transform=out_transform,
            dst_resolution=0.5,
            resampling=Resampling.bilinear
        )

    # Fill NaNs
    if np.isnan(src_nodata):
        out_image[np.isnan(out_image)] = out_nodata
    else:
        out_image[out_image == src_nodata] = out_nodata

    # Assign correct nodata to metadata and clip to min/max values
    if out_nodata == 0:
        # This is used for all validations
        meta_nd = None
        out_image[out_image > 1] = 1
        out_image[out_image < 0] = 0
    else:
        # This is used for DEM
        meta_nd = out_nodata

    # Update metadata and save geotiff
    out_profile.update(
        driver="GTiff",
        compress="lzw",
        width=out_image.shape[-1],
        height=out_image.shape[-2],
        transform=out_transform,
        nodata=meta_nd
    )
    with rasterio.open(file_path, "w", **out_profile, predictor=3) as dst:
        dst.write(out_image)

    return file_path


def run_visualisations(dem_path, tile_size, save_dir, nr_processes=1):
    """

    dem_path:
        Can be any raster file (GeoTIFF and VRT supported.)
    tile_size:
        In pixels
    save_dir:
        Save directory
    nr_processes:
        Number of processes for parallel computing

    """
    # TODO: Probably good to create dict (JSOn) with all the paths that are created here?

    # Prepare paths
    in_file = Path(dem_path)
    ds_dir = in_file.parent

    save_vis = save_dir / "vis"
    save_vis.mkdir(parents=True, exist_ok=True)

    # === STEP 1 ===
    # We need polygon covering valid data
    vdm_file = list(ds_dir.glob("*_validDataMask*"))
    if vdm_file:
        # If validDataMask exists, read it from file
        valid_data_outline = vdm_file[0]
    else:
        # If it doesn't exist, try creating it from raster
        valid_data_outline = gt.poly_from_valid(
            in_file.as_posix(),
            save_gpkg=save_vis.as_posix()
        )

    # === STEP 2 ===
    # Create reference grid, filter it and save it to disk
    tiles_extents = gt.bounding_grid(
        in_file.as_posix(),
        tile_size,
        tag=True
    )
    refgrid_name = in_file.as_posix()[:-4] + "_refgrid.gpkg"
    tiles_extents = gt.filter_by_outline(
        tiles_extents,
        valid_data_outline.as_posix(),
        save_gpkg=True,
        save_path=refgrid_name
    )

    # === STEP 3 ===
    # Run visualizations
    print("Start RVT vis")
    out_path = tiled_processing(
        input_vrt_path=in_file.as_posix(),
        ext_list=tiles_extents,
        nr_processes=nr_processes,
        ll_dir=save_vis
    )

    return out_path


def main_routine(dem_path, patch_size_px, save_dir, nr_processes=1):
    # Prepare directory for saving results
    # Make sure save folder exist
    save_dir = Path(save_dir)
    if save_dir.exists():
        dir_status = "already exists"
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        dir_status = "created new folder"

    # Save Geotif metadata (CRS, etc.)

    # ## 1 ## Create visualisation
    vis_path = run_visualisations(
        dem_path,
        4000,
        save_dir=save_dir,
        nr_processes=nr_processes
    )

    # # ## 2 ## Create patches
    # patches_dir = create_patches(
    #     vis_path,
    #     patch_size_px,
    #     save_dir,
    #     nr_processes=nr_processes
    # )
    # shutil.rmtree(vis_path)

    # ## 3 ## Run the model

    # HERE IS AiTLAS

    # ## 4 ## Create map

    return vis_path


if __name__ == "__main__":
    my_file = r"C:\Users\ncoz\GitHub\TII-demo\data\ISA-15_Kilkee\ISA-15_Kilkee_dem_05m.vrt"
    my_results = r"C:\Users\ncoz\GitHub\TII-demo\results"

    my_patch_size = 256

    rs = main_routine(my_file, my_patch_size, my_results, nr_processes=6)

    print(rs)
