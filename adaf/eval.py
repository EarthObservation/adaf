import evaluate
import geopandas as gpd
from pathlib import Path
from shapely.geometry import MultiPolygon, Polygon
import copy
import shapely.wkt
import numpy as np
import geopandas as gpd
from rasterio.features import geometry_mask
from rasterio.transform import from_bounds


def gdf_to_multipolygon_nondissolving(gdf: gpd.GeoDataFrame) -> MultiPolygon:
    """
    Convert a GeoDataFrame to a single MultiPolygon without dissolving geometries.

    - Polygons are kept as-is
    - MultiPolygons are flattened into individual Polygons
    - Empty / None geometries are ignored
    """
    polygons = []

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Polygon":
            polygons.append(geom)
        elif geom.geom_type == "MultiPolygon":
            polygons.extend(list(geom.geoms))
        else:
            raise TypeError(f"Unsupported geometry type: {geom.geom_type}")

    return MultiPolygon(polygons)


def gdf_to_bool_mask_in_rect(
    gdf: gpd.GeoDataFrame,
    rect_polygon,
    resolution: float = 0.5,
    all_touched: bool = True,
) -> np.ndarray:
    """
    Rasterize geometries from `gdf` into a boolean mask covering `rect_polygon`.

    Parameters
    ----------
    gdf : GeoDataFrame
        Input geometries to burn into the mask.
    rect_polygon : shapely Polygon
        Rectangle polygon defining the output area (bounds define the raster extent).
    resolution : float, default 0.5
        Pixel size in units of the CRS (e.g., meters). 0.5 => 0.5 m pixels.
    all_touched : bool, default True
        If True, any pixel touched by geometry becomes True.

    Returns
    -------
    np.ndarray (bool)
        Mask of shape (height, width), True where geometries cover pixels.
    """
    minx, miny, maxx, maxy = rect_polygon.bounds

    width = int(np.ceil((maxx - minx) / resolution))
    height = int(np.ceil((maxy - miny) / resolution))

    transform = from_bounds(minx, miny, maxx, maxy, width, height)

    mask = geometry_mask(
        geometries=gdf.geometry,
        out_shape=(height, width),
        transform=transform,
        invert=True,          # True inside polygons
        all_touched=all_touched
    )

    return mask.astype(bool)


def run_eval_metrics(pred_path, gt_path, splti_pth, iou_threshold=0):
    pred_path=Path(pred_path)
    gt_path=Path(gt_path)
    splti_pth=Path(splti_pth)

    # Load data
    pred_gdf = gpd.read_file(pred_path)
    gt_gdf = gpd.read_file(gt_path)
    split_gdf = gpd.read_file(splti_pth)

    # Post-processing of ML results
    pred_gdf = pred_gdf[pred_gdf["area"] < 1500].reset_index(drop=True)
    pred_gdf = pred_gdf[(pred_gdf["area"] >= 10) & (pred_gdf["roundness"] > 0.7)].reset_index(drop=True) 

    # Filter by 'test' and convert into shaply multipolygon 
    split_test = split_gdf[split_gdf['split'] == 'test'].copy()   
    test_geom = split_test.unary_union
    test_one = gpd.GeoDataFrame({"geometry": [test_geom]}, crs=pred_gdf.crs)

    pred_test = (
    gpd.sjoin(pred_gdf, test_one, how="inner", predicate="intersects")
        .drop(columns=["index_right"])
        .reset_index(drop=True)
    )

    gt_test = (
        gpd.sjoin(gt_gdf, test_one, how="inner", predicate="intersects")
            .drop(columns=["index_right"])
            .reset_index(drop=True)
    )

    # Convert a GeoDataFrame to a single MultiPolygon without dissolving geometries.
    pred_test_mp = gdf_to_multipolygon_nondissolving(pred_test)
    gt_test_mp   = gdf_to_multipolygon_nondissolving(gt_test)

    # Centroid-based evaluation
    centroid_based = evaluate.compute_iou_metric_centroid(
        ["barrow"], 
        {"barrow": list(pred_test_mp.geoms)}, 
        {"barrow": list(gt_test_mp.geoms)}, 
        iou_threshold
    )

    # Pixel-based evaluation
    total_pixel_based = np.zeros(4, dtype=np.int64)
    # accumulator: [TP, FP, FN, TN]
    for _, row in split_test.iterrows():
        rect = row.geometry

        pred_mask = gdf_to_bool_mask_in_rect(pred_test, rect, resolution=0.5)
        gt_mask   = gdf_to_bool_mask_in_rect(gt_test, rect, resolution=0.5)

        pixel_based = evaluate.compute_metrics(gt_mask, pred_mask)
        total_pixel_based += np.asarray(pixel_based, dtype=np.int64)

    # TODO: calculate f1 score for both results

    # TODO: export GDF with TP and FP geometries (save results to log file instead of print)    

    return centroid_based, total_pixel_based


if __name__ == "__main__":
    

    res_eval = run_eval_metrics(
        pred_path=r"r:\ML podatki\ml_results\adaf_retrained_2-512px.gpkg",
        gt_path=r"r:\ML podatki\archaeology\gomile_2025-11-28.gpkg",
        splti_pth=r"r:\ML podatki\learning_samples\tmp.gpkg",
        iou_threshold=0
    )

    print(res_eval)
    