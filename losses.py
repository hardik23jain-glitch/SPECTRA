"""
Compound Loss Function — Master Plan Phase 4.3

L_total = lambda_adv * L_adv + lambda_L1 * L_L1 + lambda_perc * L_perceptual + lambda_sem * L_semantic

- L_adv: Relativistic-average GAN loss (RaGAN), more stable than vanilla GAN loss and what
         ESRGAN/real production pipelines use.
- L_L1: pixel reconstruction, anchors low-frequency color/structure.
- L_perceptual: VGG16 feature-space distance, anchors texture/structure (downloads ImageNet
         VGG16 weights from torchvision on first use — needs internet on whichever machine
         actually trains the model).
- L_semantic: cross-entropy against a frozen land-cover classifier's predicted class map,
         directly penalizing hallucinated/semantically-wrong colorization (e.g., painting a
         parking lot blue). A small trainable stand-in classifier (SemanticClassifierStub) is
         provided since we don't have network access to a real pretrained NLCD model here —
         swap `classifier` for a real frozen pretrained model in production.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticClassifierStub(nn.Module):
    """Tiny CNN land-cover classifier stand-in (e.g. classes: water, vegetation, bare-soil,
    built-up, road). In production, replace with a frozen model pretrained on NLCD/EuroSAT/
    DeepGlobe labels, as outlined in Master Plan 2.3. Kept trainable here so the smoke test
    can run without external pretrained weights."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, 3, 1, 1),  # per-pixel-block class logits (downsampled map)
        )

    def forward(self, rgb):
        return self.net(rgb)  # (B, num_classes, H/4, W/4)


class VGGPerceptualLoss(nn.Module):
    """L2 distance between VGG16 relu2_2 / relu3_3 activations of generated vs. real RGB."""

    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
            self._available = True
        except Exception:
            # No internet to fetch pretrained weights in this sandbox -> fall back gracefully.
            vgg = None
            self._available = False
        if self._available:
            self.slice2_2 = nn.Sequential(*list(vgg.children())[:9]).eval()   # up to relu2_2
            self.slice3_3 = nn.Sequential(*list(vgg.children())[:16]).eval()  # up to relu3_3
            for p in self.parameters():
                p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _norm(self, x):
        # x assumed in [-1, 1] -> [0, 1] -> ImageNet normalization
        x = (x + 1) / 2
        return (x - self.mean) / self.std

    def forward(self, fake_rgb, real_rgb):
        if not self._available:
            # Graceful fallback: plain pixel-space L2 as a stand-in so training still runs
            # end-to-end without internet access to download VGG weights.
            return F.mse_loss(fake_rgb, real_rgb)
        f, r = self._norm(fake_rgb), self._norm(real_rgb)
        loss = F.mse_loss(self.slice2_2(f), self.slice2_2(r))
        loss = loss + F.mse_loss(self.slice3_3(f), self.slice3_3(r))
        return loss


def relativistic_adversarial_loss(d_real_logits, d_fake_logits, for_generator: bool):
    """RaGAN loss as used in ESRGAN. D predicts whether real is more realistic than fake
    (and vice versa) rather than an absolute real/fake label."""
    real_mean = d_real_logits.mean(dim=0, keepdim=True)
    fake_mean = d_fake_logits.mean(dim=0, keepdim=True)

    if for_generator:
        loss_real = F.binary_cross_entropy_with_logits(
            d_real_logits - fake_mean, torch.zeros_like(d_real_logits))
        loss_fake = F.binary_cross_entropy_with_logits(
            d_fake_logits - real_mean, torch.ones_like(d_fake_logits))
    else:
        loss_real = F.binary_cross_entropy_with_logits(
            d_real_logits - fake_mean, torch.ones_like(d_real_logits))
        loss_fake = F.binary_cross_entropy_with_logits(
            d_fake_logits - real_mean, torch.zeros_like(d_fake_logits))
    return (loss_real + loss_fake) / 2


class CompoundGeneratorLoss(nn.Module):
    def __init__(self, lambda_adv=1.0, lambda_l1=100.0, lambda_perc=10.0, lambda_sem=10.0,
                 semantic_classifier: nn.Module = None):
        super().__init__()
        self.lambda_adv = lambda_adv
        self.lambda_l1 = lambda_l1
        self.lambda_perc = lambda_perc
        self.lambda_sem = lambda_sem
        self.perceptual = VGGPerceptualLoss()
        self.classifier = semantic_classifier if semantic_classifier is not None else SemanticClassifierStub()

    def forward(self, fake_rgb, real_rgb, d_real_logits, d_fake_logits):
        l_adv = relativistic_adversarial_loss(d_real_logits, d_fake_logits, for_generator=True)
        l_l1 = F.l1_loss(fake_rgb, real_rgb)
        l_perc = self.perceptual(fake_rgb, real_rgb)

        with torch.no_grad():
            real_class_map = self.classifier(real_rgb).argmax(dim=1)  # pseudo-labels from GT
        fake_class_logits = self.classifier(fake_rgb)
        l_sem = F.cross_entropy(fake_class_logits, real_class_map)

        total = (self.lambda_adv * l_adv + self.lambda_l1 * l_l1 +
                 self.lambda_perc * l_perc + self.lambda_sem * l_sem)
        return total, {"adv": l_adv.item(), "l1": l_l1.item(),
                        "perceptual": l_perc.item(), "semantic": l_sem.item(), "total": total.item()}


if __name__ == "__main__":
    fake = torch.rand(2, 3, 64, 64) * 2 - 1
    real = torch.rand(2, 3, 64, 64) * 2 - 1
    d_real = torch.randn(2, 1, 8, 8)
    d_fake = torch.randn(2, 1, 8, 8)
    loss_fn = CompoundGeneratorLoss()
    total, parts = loss_fn(fake, real, d_real, d_fake)
    print("Compound loss components:", parts)
    print("OK")
