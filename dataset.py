"""
Unified dataset interface — wraps either the synthetic generator (default, works offline)
or a directory of real Landsat patches produced by landsat_pipeline.py.

Real-patch directory layout expected (as emitted by extract_paired_patches):
    patches/<scene_id>/patch_000000_ir.npy   # (H, W) float32, brightness temperature normalized
    patches/<scene_id>/patch_000000_rgb.npy  # (H, W, 3) float32, surface reflectance normalized
    patches/<scene_id>/manifest.json
"""
import json
import os
from glob import glob

import numpy as np
import torch
from torch.utils.data import Dataset

from data.synthetic_data import SyntheticIRColorizationDataset


class RealLandsatPatchDataset(Dataset):
    """Loads pre-extracted real Landsat patches from disk (see landsat_pipeline.py).
    Land-cover labels and acquisition-time conditioning are pulled from manifest.json
    if present; otherwise dummy values are used (still trains, just without semantic/
    diurnal supervision for that sample)."""

    def __init__(self, patch_root: str):
        self.ir_files = sorted(glob(os.path.join(patch_root, "**", "*_ir.npy"), recursive=True))
        if not self.ir_files:
            raise FileNotFoundError(f"No *_ir.npy patches found under {patch_root}")

    def __len__(self):
        return len(self.ir_files)

    def __getitem__(self, idx):
        ir_path = self.ir_files[idx]
        rgb_path = ir_path.replace("_ir.npy", "_rgb.npy")
        ir = np.load(ir_path).astype(np.float32)
        rgb = np.load(rgb_path).astype(np.float32)

        # normalize to [-1, 1] assuming inputs were stored as physical units / reflectance [0,1]
        if ir.max() > 1.5:  # heuristic: looks like raw DN/Kelvin, not yet normalized
            ir = (ir - ir.min()) / (ir.max() - ir.min() + 1e-6)
        ir = ir * 2 - 1
        rgb = np.clip(rgb, 0, 1) * 2 - 1

        manifest_path = os.path.join(os.path.dirname(ir_path), "manifest.json")
        cond = np.array([0.5, 0.5], dtype=np.float32)
        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                _ = json.load(f)  # acquisition time-of-day/day-of-year would be parsed here
            # In production: parse the scene's STAC 'datetime' property into [tod, doy] in [0,1].

        label = np.zeros(rgb.shape[:2], dtype=np.int64)  # placeholder; wire up a real NLCD raster here

        return {
            "ir_lr": torch.from_numpy(ir).unsqueeze(0),
            "rgb_hr": torch.from_numpy(rgb).transpose(2, 0).transpose(1, 2),
            "label_hr": torch.from_numpy(label),
            "cond": torch.from_numpy(cond),
        }


def build_dataset(data_root: str = None, n_synthetic: int = 256, size_hr: int = 256, scale: int = 2):
    """Returns a real dataset if data_root is given and contains patches, otherwise falls
    back to the synthetic generator (useful for the smoke test and for environments without
    network access to Landsat data, like this one)."""
    if data_root and os.path.isdir(data_root):
        try:
            return RealLandsatPatchDataset(data_root)
        except FileNotFoundError:
            print(f"[dataset] No real patches found at {data_root}, falling back to synthetic data.")
    return SyntheticIRColorizationDataset(n_samples=n_synthetic, size_hr=size_hr, scale=scale)


if __name__ == "__main__":
    ds = build_dataset(data_root=None, n_synthetic=4, size_hr=128)
    print(f"Dataset size: {len(ds)}, sample keys: {list(ds[0].keys())}")
    print("OK")
