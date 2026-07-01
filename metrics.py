"""
Evaluation Metrics — Master Plan Phase 1 (3.1) / Phase 4

All functions accept numpy arrays or torch tensors in [0, 1] or [-1, 1] (auto-detected)
shaped (H, W, C) or (B, C, H, W).
"""
import numpy as np
import torch
from skimage.metrics import structural_similarity as sk_ssim
from scipy import linalg


def _to_numpy_01(x):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    x = x.astype(np.float64)
    if x.min() < -1e-3:  # looks like [-1, 1] range
        x = (x + 1) / 2
    return np.clip(x, 0, 1)


def psnr(generated, ground_truth, max_val: float = 1.0) -> float:
    """PSNR = 10 * log10(MAX_I^2 / MSE)"""
    g = _to_numpy_01(generated)
    r = _to_numpy_01(ground_truth)
    mse = np.mean((g - r) ** 2)
    if mse == 0:
        return float("inf")
    return 10 * np.log10((max_val ** 2) / mse)


def ssim(generated, ground_truth) -> float:
    """Structural Similarity Index, averaged over channels if multi-channel."""
    g = _to_numpy_01(generated)
    r = _to_numpy_01(ground_truth)
    if g.ndim == 4:  # batch -> average per-sample
        return float(np.mean([ssim(g[i], r[i]) for i in range(g.shape[0])]))
    if g.ndim == 3 and g.shape[0] in (1, 3):  # CHW -> HWC
        g = np.transpose(g, (1, 2, 0))
        r = np.transpose(r, (1, 2, 0))
    channel_axis = -1 if g.ndim == 3 and g.shape[-1] in (1, 3) else None
    return float(sk_ssim(g, r, data_range=1.0, channel_axis=channel_axis))


def _fid_from_features(feats_real: np.ndarray, feats_fake: np.ndarray) -> float:
    mu_r, mu_g = feats_real.mean(axis=0), feats_fake.mean(axis=0)
    sigma_r = np.cov(feats_real, rowvar=False)
    sigma_g = np.cov(feats_fake, rowvar=False)

    diff = mu_r - mu_g
    covmean, _ = linalg.sqrtm(sigma_r @ sigma_g, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff @ diff + np.trace(sigma_r + sigma_g - 2 * covmean)
    return float(fid)


class FIDScorer:
    """Wraps an Inception-v3 feature extractor (real, ImageNet-pretrained) to compute FID.
    Requires internet access on first use to download torchvision's inception weights;
    falls back to a lightweight random-projection feature extractor (still a valid, if
    weaker, FID-style distributional distance) when offline, so the pipeline still runs."""

    def __init__(self, device: str = "cpu"):
        self.device = device
        try:
            from torchvision.models import inception_v3, Inception_V3_Weights
            net = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1, aux_logits=True)
            net.fc = torch.nn.Identity()
            net.eval().to(device)
            self.model = net
            self.input_size = 299
            self._real_inception = True
        except Exception:
            # Offline fallback: fixed random conv projector -> still produces a meaningful
            # Gaussian-distribution distance between real/fake batches, just not the
            # standard literature-comparable Inception-feature FID.
            torch.manual_seed(0)
            self.model = torch.nn.Sequential(
                torch.nn.Conv2d(3, 16, 3, 2, 1), torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 3, 2, 1), torch.nn.ReLU(),
                torch.nn.AdaptiveAvgPool2d(1),
            ).eval().to(device)
            self.input_size = 64
            self._real_inception = False

    @torch.no_grad()
    def _features(self, images: torch.Tensor) -> np.ndarray:
        # images: (B, 3, H, W) in [-1, 1] or [0, 1]
        if images.min() < -1e-3:
            images = (images + 1) / 2
        images = torch.nn.functional.interpolate(
            images, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        feats = self.model(images.to(self.device))
        if isinstance(feats, tuple):  # inception aux output during eval shouldn't occur, but be safe
            feats = feats[0]
        return feats.view(feats.size(0), -1).cpu().numpy()

    def score(self, real_batch: torch.Tensor, fake_batch: torch.Tensor) -> float:
        f_real = self._features(real_batch)
        f_fake = self._features(fake_batch)
        return _fid_from_features(f_real, f_fake)


if __name__ == "__main__":
    a = torch.rand(4, 3, 64, 64)
    b = a + torch.randn_like(a) * 0.05
    print("PSNR:", psnr(b, a))
    print("SSIM:", ssim(b, a))
    fid_scorer = FIDScorer()
    print("FID (real_inception=%s):" % fid_scorer._real_inception, fid_scorer.score(a, b))
    print("OK")
