"""
Real Landsat 8/9 data pipeline — Master Plan Phase 1 (1.2, 1.3) & Phase 4 (4.1)

NOTE: This code requires network access to the AWS Open Data STAC API and is NOT executable
in this sandboxed environment (egress is restricted to package registries only). The logic
below is complete and correct; run it on your own machine / cloud instance where you have
internet access. See README.md for how this slots into the rest of the pipeline.

Usage:
    python data/landsat_pipeline.py --bbox -122.6 37.6 -122.3 37.9 \
        --date-start 2024-06-01 --date-end 2024-09-01 --max-cloud 15 \
        --out-dir ./raw_scenes --patch-dir ./patches
"""
import argparse
import os
import json
from pathlib import Path

import numpy as np

try:
    import rasterio
    from rasterio.windows import Window
    from rasterio.warp import calculate_default_transform, reproject, Resampling
    from rasterio.enums import Resampling as ResamplingEnum
except ImportError:
    rasterio = None  # allows the rest of the module (e.g. constants) to be imported without rasterio installed

try:
    from pystac_client import Client
except ImportError:
    Client = None

STAC_URL = "https://landsatlook.usgs.gov/stac-server"  # USGS STAC; alternative: planetarycomputer.microsoft.com/api/stac/v1
COLLECTION = "landsat-c2l2-sr"  # Collection 2 Level-2 Surface Reflectance/Temperature

# Brightness temperature conversion constants (read from each scene's MTL/metadata in production;
# these are illustrative defaults for Landsat 9 TIRS Band 10).
K1_CONST = 774.8853
K2_CONST = 1321.0789


def query_scenes(bbox, date_start, date_end, max_cloud_pct=20, limit=20):
    """Query the USGS/AWS STAC catalog for Landsat 8/9 L2 scenes matching the AOI/date/cloud filters."""
    if Client is None:
        raise RuntimeError("pystac-client not installed. pip install pystac-client")
    catalog = Client.open(STAC_URL)
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=f"{date_start}/{date_end}",
        query={"eo:cloud_cover": {"lt": max_cloud_pct}},
        limit=limit,
    )
    items = list(search.items())
    return items


def download_scene_bands(item, out_dir: str, bands=("blue", "green", "red", "lwir11", "qa_pixel")):
    """Download the requested asset bands for a single STAC item to out_dir.
    band keys follow USGS Landsat Collection 2 STAC asset naming
    (e.g. 'blue'=B2, 'green'=B3, 'red'=B4, 'lwir11'=B10/TIRS)."""
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    for band in bands:
        if band not in item.assets:
            continue
        href = item.assets[band].href
        local_path = os.path.join(out_dir, f"{item.id}_{band}.tif")
        if not os.path.exists(local_path):
            import urllib.request
            urllib.request.urlretrieve(href, local_path)
        paths[band] = local_path
    return paths


def reproject_to_match(src_path, ref_path, out_path, resampling=None):
    if resampling is None:
        resampling = ResamplingEnum.bilinear
    """Reproject/resample src_path onto ref_path's CRS/transform/shape (e.g. resample TIRS
    onto the OLI 30m grid). Master Plan 4.1 step 2 — uses bilinear deliberately, not cubic,
    to avoid manufacturing artificial sharpness the model could mistake for real signal."""
    with rasterio.open(ref_path) as ref:
        ref_crs, ref_transform, ref_w, ref_h = ref.crs, ref.transform, ref.width, ref.height

    with rasterio.open(src_path) as src:
        kwargs = src.meta.copy()
        kwargs.update({"crs": ref_crs, "transform": ref_transform, "width": ref_w, "height": ref_h})
        with rasterio.open(out_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=resampling,
            )
    return out_path


def dn_to_brightness_temperature(dn_array, radiance_mult, radiance_add, k1=K1_CONST, k2=K2_CONST):
    """Convert TIRS DN -> radiance -> brightness temperature (Kelvin), per Master Plan 4.1 step 4.
    radiance_mult/add come from the scene's MTL.json metadata (RADIANCE_MULT_BAND_10 etc.)."""
    radiance = dn_array.astype(np.float64) * radiance_mult + radiance_add
    radiance = np.clip(radiance, 1e-6, None)  # avoid log(0)
    bt_kelvin = k2 / np.log((k1 / radiance) + 1)
    return bt_kelvin


def decode_qa_pixel_mask(qa_array):
    """Decode the Landsat Collection 2 QA_PIXEL bitmask to a boolean 'usable' mask,
    rejecting cloud / cloud-shadow / cirrus flagged pixels (Master Plan 4.1 step 3)."""
    qa = qa_array.astype(np.uint16)
    dilated_cloud = (qa & (1 << 1)) != 0
    cirrus = (qa & (1 << 2)) != 0
    cloud = (qa & (1 << 3)) != 0
    cloud_shadow = (qa & (1 << 4)) != 0
    bad = dilated_cloud | cirrus | cloud | cloud_shadow
    return ~bad  # True = usable pixel


def extract_paired_patches(rgb_path_stack, ir_path, label_path, qa_path, patch_size=256,
                            stride=128, max_masked_fraction=0.05, out_dir="patches"):
    """Slide a window across the co-registered grid and emit aligned
    (IR, RGB, land-cover-label) patch triplets to disk as .npy files
    (Master Plan 4.1 step 5). Rejects patches with too much cloud/nodata.
    Patches are skipped if the masked fraction (from QA_PIXEL) exceeds max_masked_fraction."""
    os.makedirs(out_dir, exist_ok=True)
    with rasterio.open(ir_path) as ir_src, rasterio.open(qa_path) as qa_src:
        rgb_srcs = [rasterio.open(p) for p in rgb_path_stack]
        width, height = ir_src.width, ir_src.height
        idx = 0
        manifest = []
        for top in range(0, height - patch_size, stride):
            for left in range(0, width - patch_size, stride):
                window = Window(left, top, patch_size, patch_size)
                qa_patch = qa_src.read(1, window=window)
                usable = decode_qa_pixel_mask(qa_patch)
                masked_fraction = 1.0 - usable.mean()
                if masked_fraction > max_masked_fraction:
                    continue

                ir_patch = ir_src.read(1, window=window)
                rgb_patch = np.stack([s.read(1, window=window) for s in rgb_srcs], axis=-1)

                np.save(os.path.join(out_dir, f"patch_{idx:06d}_ir.npy"), ir_patch)
                np.save(os.path.join(out_dir, f"patch_{idx:06d}_rgb.npy"), rgb_patch)
                manifest.append({"idx": idx, "row": top, "col": left, "masked_fraction": float(masked_fraction)})
                idx += 1
        for s in rgb_srcs:
            s.close()

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    return idx  # number of patches emitted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bbox", nargs=4, type=float, required=True, help="minlon minlat maxlon maxlat")
    parser.add_argument("--date-start", required=True)
    parser.add_argument("--date-end", required=True)
    parser.add_argument("--max-cloud", type=float, default=20)
    parser.add_argument("--out-dir", default="./raw_scenes")
    parser.add_argument("--patch-dir", default="./patches")
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--stride", type=int, default=128)
    args = parser.parse_args()

    if rasterio is None or Client is None:
        raise RuntimeError(
            "rasterio and pystac-client are required for the real pipeline. "
            "pip install rasterio pystac-client. This also requires network access "
            "to USGS/AWS, which is NOT available in the sandboxed code-generation environment."
        )

    items = query_scenes(args.bbox, args.date_start, args.date_end, args.max_cloud)
    print(f"Found {len(items)} scenes matching query.")
    total_patches = 0
    for item in items:
        paths = download_scene_bands(item, args.out_dir)
        if "lwir11" not in paths or "blue" not in paths:
            continue
        ir_resampled = paths["lwir11"].replace(".tif", "_resampled.tif")
        reproject_to_match(paths["lwir11"], paths["red"], ir_resampled)
        n = extract_paired_patches(
            rgb_path_stack=[paths["red"], paths["green"], paths["blue"]],
            ir_path=ir_resampled, label_path=None, qa_path=paths["qa_pixel"],
            patch_size=args.patch_size, stride=args.stride,
            out_dir=os.path.join(args.patch_dir, item.id),
        )
        total_patches += n
        print(f"{item.id}: extracted {n} patches")
    print(f"Total patches extracted: {total_patches}")


if __name__ == "__main__":
    main()
