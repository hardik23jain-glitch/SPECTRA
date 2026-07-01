"""
Training script — Master Plan Phase 4.2 (two stages trained separately, then jointly fine-tuned).

Run from the project root, e.g.:
    python -m scripts.train --stage both --epochs 2 --data-root None --size 64 --n-synthetic 32
"""
import argparse
import os

import torch
from torch.utils.data import DataLoader

from data.dataset import build_dataset
from models.sr_model import ESRGANGenerator
from models.colorization_model import UNetColorizationGenerator, PatchGANDiscriminator
from losses.losses import CompoundGeneratorLoss, relativistic_adversarial_loss
from metrics.metrics import psnr, ssim


def train_stage1_sr(loader, device, epochs, lr=2e-4, scale=2):
    """Pretrain the SR generator with pixel + adversarial loss against the high-res IR
    proxy (here, the ground-truth RGB's luminance channel stands in for a co-registered
    panchromatic band, since synthetic data has no separate pan band)."""
    gen = ESRGANGenerator(scale=scale, num_rrdb=4).to(device)
    opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.9, 0.999))

    for epoch in range(epochs):
        running = 0.0
        for batch in loader:
            ir_lr = batch["ir_lr"].to(device)
            rgb_hr = batch["rgb_hr"].to(device)
            # Use luminance of the HR RGB as the "real sharp edge" supervision target,
            # standing in for the co-registered panchromatic band described in Master Plan 4.2.
            target_luma = (0.299 * rgb_hr[:, 0] + 0.587 * rgb_hr[:, 1] + 0.114 * rgb_hr[:, 2]).unsqueeze(1)

            opt.zero_grad()
            pred = gen(ir_lr)
            if pred.shape[-2:] != target_luma.shape[-2:]:
                target_luma = torch.nn.functional.interpolate(target_luma, size=pred.shape[-2:], mode="bilinear")
            loss = torch.nn.functional.l1_loss(pred, target_luma)
            loss.backward()
            opt.step()
            running += loss.item()
        print(f"[Stage1-SR] epoch {epoch+1}/{epochs} - L1 loss: {running/len(loader):.4f}")
    return gen


def train_stage2_colorization(loader, device, epochs, lr=2e-4):
    """Pretrain the colorization generator + PatchGAN discriminator with the full
    compound loss (adv + L1 + perceptual + semantic)."""
    gen = UNetColorizationGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)
    loss_fn = CompoundGeneratorLoss().to(device)

    opt_g = torch.optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        running_g, running_d = 0.0, 0.0
        last_metrics = {}
        for batch in loader:
            ir_hr = batch["ir_lr"].to(device)  # already upsampled by stage1 in the joint pipeline;
            # for standalone stage2 pretraining we just upsample the raw IR to match RGB size.
            rgb_hr = batch["rgb_hr"].to(device)
            cond = batch["cond"].to(device)

            if ir_hr.shape[-2:] != rgb_hr.shape[-2:]:
                ir_hr = torch.nn.functional.interpolate(ir_hr, size=rgb_hr.shape[-2:], mode="bilinear")

            # --- Discriminator step ---
            opt_d.zero_grad()
            with torch.no_grad():
                fake_rgb = gen(ir_hr, cond)
            d_real = disc(ir_hr, rgb_hr)
            d_fake = disc(ir_hr, fake_rgb)
            d_loss = relativistic_adversarial_loss(d_real, d_fake, for_generator=False)
            d_loss.backward()
            opt_d.step()

            # --- Generator step ---
            opt_g.zero_grad()
            fake_rgb = gen(ir_hr, cond)
            d_real = disc(ir_hr, rgb_hr)
            d_fake = disc(ir_hr, fake_rgb)
            g_loss, parts = loss_fn(fake_rgb, rgb_hr, d_real, d_fake)
            g_loss.backward()
            opt_g.step()

            running_g += g_loss.item()
            running_d += d_loss.item()
            last_metrics = parts

        avg_psnr = psnr(fake_rgb, rgb_hr)
        avg_ssim = ssim(fake_rgb, rgb_hr)
        print(f"[Stage2-Color] epoch {epoch+1}/{epochs} - G: {running_g/len(loader):.3f} "
              f"D: {running_d/len(loader):.3f} | last-batch components: {last_metrics} "
              f"| PSNR: {avg_psnr:.2f} SSIM: {avg_ssim:.3f}")
    return gen, disc


def train_joint(sr_gen, color_gen, color_disc, loader, device, epochs, lr=1e-5):
    """Joint fine-tune: chain SR -> colorization, backprop through both with a small LR
    so the SR module's notion of 'useful sharpening' adapts to what helps colorization,
    per Master Plan 4.2."""
    loss_fn = CompoundGeneratorLoss().to(device)
    params = list(sr_gen.parameters()) + list(color_gen.parameters())
    opt = torch.optim.Adam(params, lr=lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(color_disc.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        running = 0.0
        for batch in loader:
            ir_lr = batch["ir_lr"].to(device)
            rgb_hr = batch["rgb_hr"].to(device)
            cond = batch["cond"].to(device)

            sr_out = sr_gen(ir_lr)
            if sr_out.shape[-2:] != rgb_hr.shape[-2:]:
                sr_out = torch.nn.functional.interpolate(sr_out, size=rgb_hr.shape[-2:], mode="bilinear")

            opt_d.zero_grad()
            with torch.no_grad():
                fake_rgb = color_gen(sr_out, cond)
            d_real = color_disc(sr_out, rgb_hr)
            d_fake = color_disc(sr_out, fake_rgb)
            d_loss = relativistic_adversarial_loss(d_real, d_fake, for_generator=False)
            d_loss.backward()
            opt_d.step()

            opt.zero_grad()
            sr_out = sr_gen(ir_lr)
            if sr_out.shape[-2:] != rgb_hr.shape[-2:]:
                sr_out = torch.nn.functional.interpolate(sr_out, size=rgb_hr.shape[-2:], mode="bilinear")
            fake_rgb = color_gen(sr_out, cond)
            d_real = color_disc(sr_out, rgb_hr)
            d_fake = color_disc(sr_out, fake_rgb)
            g_loss, _ = loss_fn(fake_rgb, rgb_hr, d_real, d_fake)
            g_loss.backward()
            opt.step()
            running += g_loss.item()
        print(f"[Joint fine-tune] epoch {epoch+1}/{epochs} - loss: {running/len(loader):.3f}")
    return sr_gen, color_gen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=None, help="Path to real Landsat patches; omit for synthetic data")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--size", type=int, default=64, help="patch size (use 256 for real training)")
    parser.add_argument("--n-synthetic", type=int, default=32)
    parser.add_argument("--stage", choices=["sr", "color", "joint", "both"], default="both")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = build_dataset(data_root=args.data_root, n_synthetic=args.n_synthetic, size_hr=args.size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    sr_gen = color_gen = color_disc = None
    if args.stage in ("sr", "both", "joint"):
        sr_gen = train_stage1_sr(loader, device, args.epochs)
        torch.save(sr_gen.state_dict(), os.path.join(args.checkpoint_dir, "stage1_sr.pt"))

    if args.stage in ("color", "both", "joint"):
        color_gen, color_disc = train_stage2_colorization(loader, device, args.epochs)
        torch.save(color_gen.state_dict(), os.path.join(args.checkpoint_dir, "stage2_color.pt"))

    if args.stage == "both" and sr_gen is not None and color_gen is not None:
        sr_gen, color_gen = train_joint(sr_gen, color_gen, color_disc, loader, device, max(1, args.epochs // 2))
        torch.save(sr_gen.state_dict(), os.path.join(args.checkpoint_dir, "stage1_sr_joint.pt"))
        torch.save(color_gen.state_dict(), os.path.join(args.checkpoint_dir, "stage2_color_joint.pt"))

    print("Training complete. Checkpoints saved to", args.checkpoint_dir)


if __name__ == "__main__":
    main()
