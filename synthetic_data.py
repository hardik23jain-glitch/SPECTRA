"""
Synthetic paired-data generator — stands in for real Landsat downloads in this sandbox
(no network access to USGS EarthExplorer / AWS Landsat S3 here, see README).

Generates procedural "fake satellite scenes" with plausible land-cover blobs (water,
forest, bare soil, built-up, road) and renders:
  - a low-res, blurred, single-channel "IR" view (simulating the ~100m-equivalent TIRS signal)
  - a high-res 3-channel "RGB" ground truth, color-coded by land-cover class with added texture
  - a per-pixel land-cover label map (for the semantic loss)

This lets the entire training/inference/metrics pipeline be exercised end-to-end without
internet access. Swap this module for `landsat_pipeline.py` once you have real downloaded data.
"""
import numpy as np
import torch
from scipy.ndimage import gaussian_filter

CLASS_COLORS = {
    0: (30, 60, 200),    # water -> blue
    1: (30, 140, 40),    # vegetation -> green
    2: (170, 140, 90),   # bare soil -> tan
    3: (120, 120, 120),  # built-up -> gray
    4: (60, 60, 60),     # road -> dark gray
}
# Rough relative "thermal signature" base value per class (purely illustrative, not physical)
CLASS_THERMAL = {0: 0.15, 1: 0.35, 2: 0.55, 3: 0.75, 4: 0.85}


def _random_blobs(size, n_blobs=6, rng=None):
    rng = rng or np.random.default_rng()
    label = np.zeros((size, size), dtype=np.int64)
    for _ in range(n_blobs):
        cls = rng.integers(0, len(CLASS_COLORS))
        cy, cx = rng.integers(0, size, size=2)
        r = rng.integers(size // 8, size // 3)
        yy, xx = np.ogrid[:size, :size]
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        label[mask] = cls
    return label


def generate_patch(size_hr: int = 256, scale: int = 2, seed: int = None):
    """Returns (ir_lr, rgb_hr, label_hr, cond) as numpy arrays.
    ir_lr: (size_hr//scale, size_hr//scale) float32 in [-1,1]
    rgb_hr: (size_hr, size_hr, 3) float32 in [-1,1]
    label_hr: (size_hr, size_hr) int64 class map
    cond: (2,) float32 [time_of_day, day_of_year] both in [0,1]
    """
    rng = np.random.default_rng(seed)
    label_hr = _random_blobs(size_hr, n_blobs=rng.integers(4, 10), rng=rng)

    rgb_hr = np.zeros((size_hr, size_hr, 3), dtype=np.float32)
    for cls, color in CLASS_COLORS.items():
        mask = label_hr == cls
        rgb_hr[mask] = np.array(color, dtype=np.float32) / 255.0
    # add per-pixel texture noise so it's not flat blobs
    rgb_hr = rgb_hr + rng.normal(0, 0.03, rgb_hr.shape).astype(np.float32)
    rgb_hr = np.clip(rgb_hr, 0, 1)

    # build a thermal field correlated with class but with diurnal/seasonal modulation
    cond = rng.random(2).astype(np.float32)  # [time_of_day, day_of_year]
    diurnal_shift = 0.15 * np.sin(2 * np.pi * cond[0])      # day/night swing
    seasonal_shift = 0.1 * np.cos(2 * np.pi * cond[1])      # winter/summer swing
    thermal = np.zeros((size_hr, size_hr), dtype=np.float32)
    for cls, base in CLASS_THERMAL.items():
        thermal[label_hr == cls] = base
    thermal = thermal + diurnal_shift + seasonal_shift
    thermal = thermal + rng.normal(0, 0.02, thermal.shape).astype(np.float32)
    thermal = np.clip(thermal, 0, 1)

    # simulate the real ~100m-effective resolution: heavy blur + downsample (Master Plan 1.1 / 2.1)
    thermal_blurred = gaussian_filter(thermal, sigma=size_hr / 64)
    lr_size = size_hr // scale
    ir_lr = thermal_blurred[::scale, ::scale][:lr_size, :lr_size]
    ir_lr = ir_lr * 2 - 1  # -> [-1, 1]

    rgb_hr = rgb_hr * 2 - 1  # -> [-1, 1]
    return ir_lr.astype(np.float32), rgb_hr.astype(np.float32), label_hr, cond


class SyntheticIRColorizationDataset(torch.utils.data.Dataset):
    def __init__(self, n_samples: int = 256, size_hr: int = 256, scale: int = 2, seed_offset: int = 0):
        self.n_samples = n_samples
        self.size_hr = size_hr
        self.scale = scale
        self.seed_offset = seed_offset

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        ir_lr, rgb_hr, label_hr, cond = generate_patch(
            self.size_hr, self.scale, seed=idx + self.seed_offset)
        ir_lr_t = torch.from_numpy(ir_lr).unsqueeze(0)              # (1, h, w)
        rgb_hr_t = torch.from_numpy(rgb_hr).permute(2, 0, 1)        # (3, H, W)
        label_hr_t = torch.from_numpy(label_hr)                     # (H, W)
        cond_t = torch.from_numpy(cond)                             # (2,)
        return {"ir_lr": ir_lr_t, "rgb_hr": rgb_hr_t, "label_hr": label_hr_t, "cond": cond_t}


if __name__ == "__main__":
    ds = SyntheticIRColorizationDataset(n_samples=4, size_hr=128, scale=2)
    sample = ds[0]
    for k, v in sample.items():
        print(k, v.shape, v.dtype)
    print("OK")
