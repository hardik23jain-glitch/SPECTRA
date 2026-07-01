"""
End-to-end smoke test — proves the whole pipeline (data -> train -> infer -> export -> metrics)
runs correctly together, using synthetic data since this sandbox has no internet access to
real Landsat scenes. Run from the project root: python -m scripts.smoke_test
"""
import os
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.synthetic_data import SyntheticIRColorizationDataset
from scripts.train import train_stage1_sr, train_stage2_colorization
from scripts.inference import run_inference
from metrics.metrics import psnr, ssim, FIDScorer


def main():
    print("=" * 60)
    print("SMOKE TEST: IR Colorization & Enhancement Pipeline")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    size_hr = 64  # small for a fast CPU smoke test; use 256 for real training
    scale = 2

    ds = SyntheticIRColorizationDataset(n_samples=16, size_hr=size_hr, scale=scale)
    loader = DataLoader(ds, batch_size=4, shuffle=True, drop_last=True)
    print(f"\n[1/5] Synthetic dataset ready: {len(ds)} samples, patch size {size_hr}")

    print("\n[2/5] Training Stage 1 (SR) for 1 epoch...")
    sr_gen = train_stage1_sr(loader, device, epochs=1, scale=scale)

    print("\n[3/5] Training Stage 2 (Colorization) for 1 epoch...")
    color_gen, color_disc = train_stage2_colorization(loader, device, epochs=1)

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(sr_gen.state_dict(), "checkpoints/stage1_sr.pt")
    torch.save(color_gen.state_dict(), "checkpoints/stage2_color.pt")
    print("Checkpoints saved to checkpoints/")

    print("\n[4/5] Running inference on a single held-out synthetic IR patch...")
    sample = ds[999]  # different seed than any training index
    ir_np = sample["ir_lr"].squeeze(0).numpy()
    rgb_out = run_inference(ir_np, "checkpoints/stage1_sr.pt", "checkpoints/stage2_color.pt",
                             cond=tuple(sample["cond"].numpy()), device=device, scale=scale)
    os.makedirs("outputs_smoke", exist_ok=True)
    from PIL import Image
    Image.fromarray(rgb_out).save("outputs_smoke/sample_colorized.png")
    print(f"Inference output shape: {rgb_out.shape}, saved to outputs_smoke/sample_colorized.png")

    print("\n[5/5] Computing evaluation metrics on a batch...")
    batch = next(iter(loader))
    ir_lr = batch["ir_lr"].to(device)
    rgb_hr = batch["rgb_hr"].to(device)
    cond = batch["cond"].to(device)
    with torch.no_grad():
        sr_out = sr_gen(ir_lr)
        if sr_out.shape[-2:] != rgb_hr.shape[-2:]:
            sr_out = torch.nn.functional.interpolate(sr_out, size=rgb_hr.shape[-2:], mode="bilinear")
        fake_rgb = color_gen(sr_out, cond)

    p = psnr(fake_rgb, rgb_hr)
    s = ssim(fake_rgb, rgb_hr)
    fid_scorer = FIDScorer(device=device)
    f = fid_scorer.score(rgb_hr, fake_rgb)
    print(f"PSNR: {p:.2f} dB | SSIM: {s:.3f} | FID: {f:.2f} "
          f"(real ImageNet Inception weights: {fid_scorer._real_inception})")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED — pipeline runs end-to-end.")
    print("Note: metrics will be weak/poor after only 1 epoch on tiny synthetic data —")
    print("that's expected. This test verifies wiring/shapes/gradients, not model quality.")
    print("=" * 60)


if __name__ == "__main__":
    main()
