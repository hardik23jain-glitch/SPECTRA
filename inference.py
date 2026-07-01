"""
Inference — Master Plan Phase 4: chain Stage 1 (SR) -> Stage 2 (Colorization) on a single
raw IR patch and save the result as a PNG.

Usage:
    python -m scripts.inference --sr-checkpoint checkpoints/stage1_sr.pt \
        --color-checkpoint checkpoints/stage2_color.pt --input patch.npy --output colorized.png
"""
import argparse

import numpy as np
import torch
from PIL import Image

from models.sr_model import ESRGANGenerator
from models.colorization_model import UNetColorizationGenerator


def run_inference(ir_array: np.ndarray, sr_ckpt: str, color_ckpt: str,
                   cond=(0.5, 0.5), device: str = "cpu", scale: int = 2):
    """ir_array: (H, W) float32, normalized to [-1, 1]."""
    sr_gen = ESRGANGenerator(scale=scale, num_rrdb=4).to(device)
    sr_gen.load_state_dict(torch.load(sr_ckpt, map_location=device))
    sr_gen.eval()

    color_gen = UNetColorizationGenerator().to(device)
    color_gen.load_state_dict(torch.load(color_ckpt, map_location=device))
    color_gen.eval()

    ir_t = torch.from_numpy(ir_array).float().unsqueeze(0).unsqueeze(0).to(device)
    cond_t = torch.tensor([cond], dtype=torch.float32, device=device)

    with torch.no_grad():
        sr_out = sr_gen(ir_t)
        rgb_out = color_gen(sr_out, cond_t)

    rgb_np = rgb_out.squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb_np = np.clip((rgb_np + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return rgb_np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sr-checkpoint", required=True)
    parser.add_argument("--color-checkpoint", required=True)
    parser.add_argument("--input", required=True, help=".npy file containing a (H,W) IR patch in [-1,1]")
    parser.add_argument("--output", required=True, help="output PNG path")
    parser.add_argument("--time-of-day", type=float, default=0.5)
    parser.add_argument("--day-of-year", type=float, default=0.5)
    args = parser.parse_args()

    ir_array = np.load(args.input).astype(np.float32)
    rgb_out = run_inference(ir_array, args.sr_checkpoint, args.color_checkpoint,
                             cond=(args.time_of_day, args.day_of_year))
    Image.fromarray(rgb_out).save(args.output)
    print(f"Saved colorized output to {args.output}")


if __name__ == "__main__":
    main()
