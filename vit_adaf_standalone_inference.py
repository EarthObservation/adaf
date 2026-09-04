from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
from rasterio.features import shapes
from rasterio.windows import Window
from shapely.geometry import box, shape


class PatchEmbedding(nn.Module):
    def __init__(self, patch_dim: int, emb_dim: int):
        super().__init__()
        self.linear_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, emb_dim),
            nn.Dropout(0.3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_proj(x)


class Encoder(nn.Module):
    def __init__(self, emb_dim: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=emb_dim,
            num_heads=8,
            dropout=0.2,
            batch_first=True,
        )
        self.linear1 = nn.Linear(emb_dim, emb_dim * 4)
        self.linear2 = nn.Linear(emb_dim * 4, emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        self.dropout3 = nn.Dropout(0.3)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x_tmp = self.norm1(x)
        x_tmp, _ = self.attn(
            query=x_tmp,
            key=x_tmp,
            value=x_tmp,
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
        )
        x_tmp = self.dropout1(x_tmp)
        x = x + x_tmp

        x_tmp = self.norm2(x)
        x_tmp = self.linear1(x_tmp)
        x_tmp = F.gelu(x_tmp)
        x_tmp = self.dropout2(x_tmp)
        x_tmp = self.linear2(x_tmp)
        x_tmp = self.dropout3(x_tmp)
        x = self.norm1(x + x_tmp)
        return x


class VisionTransformerSegmentation(nn.Module):
    def __init__(
        self,
        device: str,
        in_channels: int,
        out_channel: int,
        emb_dim: int,
        image_size: int = 300,
        patch_size: int = 10,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.num_patches = num_patches
        self.patch_embedding = PatchEmbedding(patch_dim, emb_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, num_patches, emb_dim))

        self.encoder1 = Encoder(emb_dim)
        self.encoder2 = Encoder(emb_dim)
        self.encoder3 = Encoder(emb_dim)
        self.encoder4 = Encoder(emb_dim)
        self.encoder5 = Encoder(emb_dim)
        self.encoder6 = Encoder(emb_dim)

        self.fc_segmentation = nn.Conv2d(emb_dim, out_channel, kernel_size=3)
        self.seg_drop = nn.Dropout(0.3)

        self.patch_size = patch_size
        self.img_size = image_size
        self.emb_dim = emb_dim
        self.device = device

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_patches, channels, patch_size, _ = x.size()
        x = x.view(batch_size, num_patches, -1)
        x = self.patch_embedding(x)
        x = x + self.position_embedding[:, :num_patches]

        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        x = self.encoder5(x)
        x = self.encoder6(x)

        x = x.view(batch_size, self.num_patches, self.emb_dim)
        x = x.permute(0, 2, 1).contiguous().view(
            batch_size,
            self.emb_dim,
            self.img_size // self.patch_size,
            self.img_size // self.patch_size,
        )
        x = self.fc_segmentation(x)
        x = self.seg_drop(x)
        x = F.interpolate(
            x,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )
        return x


class _VitConfig:
    def __init__(
        self,
        image_size: int = 300,
        patch_size: int = 10,
        in_channels: int = 3,
        out_channels: int = 1,
        emb_dim: int = 256,
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.emb_dim = emb_dim


def _load_vit_model(
    model_weights: str | Path,
    device: torch.device,
    config: _VitConfig,
) -> VisionTransformerSegmentation:
    model = VisionTransformerSegmentation(
        device=str(device),
        in_channels=config.in_channels,
        out_channel=config.out_channels,
        emb_dim=config.emb_dim,
        image_size=config.image_size,
        patch_size=config.patch_size,
    )

    try:
        state = torch.load(model_weights, map_location=device, weights_only=False)
    except TypeError:
        state = torch.load(model_weights, map_location=device)

    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    elif isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state

    # Accept checkpoints saved from DataParallel/DDP as well.
    if isinstance(state_dict, dict):
        state_dict = {
            k.replace("module.", "", 1) if k.startswith("module.") else k: v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def _tile_to_float_3band(tile: np.ndarray) -> np.ndarray:
    """Return array as float32, shape (3, 300, 300), with values in [0, 1]."""
    if tile.ndim != 3:
        raise ValueError(f"Expected tile with shape (bands, rows, cols), got {tile.shape}")

    if tile.shape[0] == 1:
        tile = np.repeat(tile, 3, axis=0)
    elif tile.shape[0] == 2:
        tile = np.concatenate([tile, tile[:1]], axis=0)
    elif tile.shape[0] > 3:
        tile = tile[:3]

    if np.issubdtype(tile.dtype, np.integer):
        tile = tile.astype(np.float32)
        max_val = float(tile.max()) if tile.size else 0.0
        if max_val > 1.0:
            # The training notebook used ToTensor on image files; uint8 inputs become 0..1.
            scale = 255.0 if max_val <= 255.0 else max_val
            tile = tile / scale
    else:
        tile = tile.astype(np.float32)

    tile = np.nan_to_num(tile, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(tile, 0.0, 1.0)


def _extract_vit_patches(tile: np.ndarray, patch_size: int, device: torch.device) -> torch.Tensor:
    """Convert one 3-band image tile to [1, num_patches, 3, patch_size, patch_size]."""
    image = torch.from_numpy(tile).float()
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(3, -1, patch_size, patch_size)
    patches = patches.permute(1, 0, 2, 3).unsqueeze(0)
    return patches.to(device, non_blocking=True)


def _read_tile(src: rasterio.io.DatasetReader, window: Window, tile_size: int) -> np.ndarray:
    indexes = list(range(1, min(src.count, 3) + 1))
    fill_value = 0
    tile = src.read(indexes=indexes, window=window, boundless=True, fill_value=fill_value)

    # Some rasterio/GDAL combinations can return edge arrays smaller than requested.
    if tile.shape[-2:] != (tile_size, tile_size):
        out = np.zeros((tile.shape[0], tile_size, tile_size), dtype=tile.dtype)
        rows = min(tile.shape[-2], tile_size)
        cols = min(tile.shape[-1], tile_size)
        out[:, :rows, :cols] = tile[:, :rows, :cols]
        tile = out

    if src.nodata is not None:
        nodata = src.nodata
        if isinstance(nodata, float) and np.isnan(nodata):
            tile = np.where(np.isnan(tile), 0, tile)
        else:
            tile = np.where(tile == nodata, 0, tile)

    return tile


def _mask_to_polygons(
    mask: np.ndarray,
    transform,
    crs,
    label: str,
    bounds_polygon,
    tile_id: int,
) -> list[dict]:
    if not mask.any():
        return []

    rows = []
    mask_u8 = mask.astype("uint8")
    for geom_mapping, value in shapes(mask_u8, mask=mask_u8 == 1, transform=transform):
        if int(value) != 1:
            continue
        geom = shape(geom_mapping).intersection(bounds_polygon)
        if geom.is_empty:
            continue
        rows.append({"label": label, "tile_id": int(tile_id), "geometry": geom})
    return rows


def _calculate_area_roundness(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    gdf = gdf.copy()
    gdf["area"] = gdf.geometry.area.astype(float)
    convex_perimeter = gdf.geometry.convex_hull.length.astype(float)
    gdf["roundness"] = np.where(
        convex_perimeter > 0,
        4.0 * math.pi * gdf["area"] / (convex_perimeter ** 2),
        0.0,
    )
    gdf["area"] = gdf["area"].round(2)
    gdf["roundness"] = gdf["roundness"].round(3)
    return gdf


def _write_empty_gpkg(output_gpkg: Path, layer: str, crs) -> None:
    import fiona

    schema = {
        "geometry": "Polygon",
        "properties": {
            "label": "str",
            "area": "float",
            "roundness": "float",
        },
    }
    crs_wkt = crs.to_wkt() if hasattr(crs, "to_wkt") else None
    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()
    with fiona.open(
        output_gpkg,
        mode="w",
        driver="GPKG",
        layer=layer,
        schema=schema,
        crs_wkt=crs_wkt,
    ):
        pass


def _finalise_polygons(
    rows: list[dict],
    crs,
    label: str,
    min_area: float,
    min_roundness: float,
) -> gpd.GeoDataFrame:
    if not rows:
        return gpd.GeoDataFrame(
            {"label": pd.Series(dtype="str"), "area": pd.Series(dtype="float"), "roundness": pd.Series(dtype="float")},
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=crs)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    if gdf.empty:
        return gdf.drop(columns=[c for c in ["tile_id"] if c in gdf.columns], errors="ignore")

    # Fix simple self-intersections before the final union.
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    # Join components across tile boundaries and split disjoint parts back to rows.
    gdf = (
        gdf.dissolve()
        .explode(index_parts=False)
        .reset_index(drop=True)
    )
    gdf["label"] = label
    gdf = _calculate_area_roundness(gdf)

    if min_area is not None:
        gdf = gdf[gdf["area"] >= float(min_area)]
    if min_roundness is not None:
        gdf = gdf[gdf["roundness"] >= float(min_roundness)]

    return gdf[["label", "area", "roundness", "geometry"]].reset_index(drop=True)


def run_vit_detection_on_visualisation(
    input_raster: str | Path,
    model_weights: str | Path,
    output_gpkg: str | Path,
    *,
    min_area: float = 40.0,
    min_roundness: float = 0.5,
    threshold: float = 0.5,
    label: str = "AO",
    layer: str = "vit_detection",
    tile_size: int = 300,
    vit_patch_size: int = 10,
    emb_dim: int = 256,
    batch_size: int = 1,
    device: str | torch.device | None = None,
    verbose: bool = True,
) -> Path:
    """
    Run the experimental ViT segmentation model on a visualisation raster and save detections to GPKG.

    Parameters
    ----------
    input_raster:
        GeoTIFF or VRT containing an already calculated visualisation.
    model_weights:
        Path to the ViT checkpoint. The function accepts a raw state_dict or a dict with key "model".
    output_gpkg:
        Output GeoPackage path.
    min_area:
        Minimum polygon area in map units squared, normally m2 if the raster CRS is metric.
    min_roundness:
        Minimum roundness, calculated as 4*pi*area/(convex hull perimeter**2).
    threshold:
        Probability threshold applied after sigmoid.
    label:
        Value written to the label field.
    layer:
        GPKG layer name.
    tile_size:
        ViT input image size. Keep this at 300 for the model from the notebook.
    vit_patch_size:
        ViT inner patch size. Keep this at 10 for the model from the notebook.
    emb_dim:
        Embedding size. The notebook used 128*2 = 256.
    batch_size:
        Number of 300 px tiles passed to the model together.
    device:
        "cuda", "cpu", or None for auto.
    verbose:
        Print basic progress.

    Returns
    -------
    pathlib.Path
        Path to the saved GeoPackage.
    """
    input_raster = Path(input_raster)
    model_weights = Path(model_weights)
    output_gpkg = Path(output_gpkg)

    if not input_raster.exists():
        raise FileNotFoundError(f"Input raster does not exist: {input_raster}")
    if not model_weights.exists():
        raise FileNotFoundError(f"Model weights do not exist: {model_weights}")
    if tile_size % vit_patch_size != 0:
        raise ValueError("tile_size must be divisible by vit_patch_size")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    config = _VitConfig(
        image_size=tile_size,
        patch_size=vit_patch_size,
        in_channels=3,
        out_channels=1,
        emb_dim=emb_dim,
    )
    model = _load_vit_model(model_weights, device, config)

    rows: list[dict] = []
    batch_patches: list[torch.Tensor] = []
    batch_transforms = []
    batch_tile_ids: list[int] = []

    with rasterio.open(input_raster) as src:
        if src.crs is None:
            raise ValueError("Input raster has no CRS. A CRS is required for the output GPKG.")

        raster_bounds = box(*src.bounds)
        n_cols = math.ceil(src.width / tile_size)
        n_rows = math.ceil(src.height / tile_size)
        total_tiles = n_cols * n_rows
        tile_id = 0

        def flush_batch() -> None:
            nonlocal batch_patches, batch_transforms, batch_tile_ids, rows
            if not batch_patches:
                return
            x = torch.cat(batch_patches, dim=0)
            with torch.no_grad():
                probs = torch.sigmoid(model(x)).detach().cpu().numpy()[:, 0, :, :]
            for prob, tr, tid in zip(probs, batch_transforms, batch_tile_ids):
                binary = prob >= float(threshold)
                rows.extend(_mask_to_polygons(binary, tr, src.crs, label, raster_bounds, tid))
            batch_patches = []
            batch_transforms = []
            batch_tile_ids = []

        for row_off in range(0, src.height, tile_size):
            for col_off in range(0, src.width, tile_size):
                window = Window(col_off=col_off, row_off=row_off, width=tile_size, height=tile_size)
                tile = _read_tile(src, window, tile_size)
                tile = _tile_to_float_3band(tile)
                patches = _extract_vit_patches(tile, vit_patch_size, device)

                batch_patches.append(patches)
                batch_transforms.append(src.window_transform(window))
                batch_tile_ids.append(tile_id)

                if len(batch_patches) >= batch_size:
                    flush_batch()

                tile_id += 1
                if verbose and (tile_id == 1 or tile_id % 100 == 0 or tile_id == total_tiles):
                    print(f"Processed {tile_id}/{total_tiles} tiles")

        flush_batch()
        crs = src.crs

    out_gdf = _finalise_polygons(
        rows=rows,
        crs=crs,
        label=label,
        min_area=min_area,
        min_roundness=min_roundness,
    )

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()

    if out_gdf.empty:
        _write_empty_gpkg(output_gpkg, layer, crs)
    else:
        out_gdf.to_file(output_gpkg, layer=layer, driver="GPKG")

    if verbose:
        print(f"Saved {len(out_gdf)} detections to {output_gpkg}")

    return output_gpkg
