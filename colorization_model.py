"""
Stage 2 — Colorization Module
Pix2PixHD-style U-Net generator (IR -> RGB) conditioned on auxiliary scalar metadata
(time-of-day, day-of-year) to address the diurnal/seasonal thermal-ambiguity problem
(Master Plan Phase 2, 2.2), plus a multi-scale PatchGAN discriminator.

Generator input:  (B, 1, H, W) enhanced IR  +  (B, 2) conditioning vector [sin/cos encoded time]
Generator output: (B, 3, H, W) RGB in [-1, 1]
Discriminator input: (B, 4, H, W) concat(IR, RGB) -> patch realism map
"""
import torch
import torch.nn as nn


def conv_block(in_c, out_c, down=True, use_norm=True, dropout=False):
    layers = []
    if down:
        layers.append(nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not use_norm))
    else:
        layers.append(nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=not use_norm))
    if use_norm:
        layers.append(nn.InstanceNorm2d(out_c, affine=True))  # InstanceNorm: per-sample stats,
        # more robust than BatchNorm to the cross-scene brightness shifts described in Phase 2.
    layers.append(nn.LeakyReLU(0.2, inplace=True) if down else nn.ReLU(inplace=True))
    if dropout:
        layers.append(nn.Dropout(0.5))
    return nn.Sequential(*layers)


class ConditioningEmbed(nn.Module):
    """Encodes [time_of_day, day_of_year] (each in [0,1]) into a small spatial feature map
    that gets concatenated with the bottleneck features, per Master Plan 4.2."""

    def __init__(self, out_channels: int, spatial: int = 8):
        super().__init__()
        self.spatial = spatial
        self.out_channels = out_channels
        # sin/cos cyclical encoding (2 inputs -> 4 cyclical features) -> MLP -> spatial map
        self.mlp = nn.Sequential(
            nn.Linear(4, 64), nn.ReLU(inplace=True),
            nn.Linear(64, out_channels * spatial * spatial),
        )

    def forward(self, cond: torch.Tensor):
        # cond: (B, 2) raw [0,1] values for time_of_day, day_of_year
        tod, doy = cond[:, 0], cond[:, 1]
        cyc = torch.stack([
            torch.sin(2 * torch.pi * tod), torch.cos(2 * torch.pi * tod),
            torch.sin(2 * torch.pi * doy), torch.cos(2 * torch.pi * doy),
        ], dim=1)
        feat = self.mlp(cyc)
        return feat.view(-1, self.out_channels, self.spatial, self.spatial)


class UNetColorizationGenerator(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3, base: int = 64):
        super().__init__()
        # Encoder
        self.down1 = conv_block(in_channels, base, down=True, use_norm=False)        # H/2
        self.down2 = conv_block(base, base * 2, down=True)                            # H/4
        self.down3 = conv_block(base * 2, base * 4, down=True)                        # H/8
        self.down4 = conv_block(base * 4, base * 8, down=True)                        # H/16
        self.down5 = conv_block(base * 8, base * 8, down=True)                        # H/32 (bottleneck)

        self.cond_embed = ConditioningEmbed(out_channels=base * 8, spatial=1)

        # Decoder (skip connections double channel counts on the way back up)
        self.up1 = conv_block(base * 8 + base * 8, base * 8, down=False, dropout=True)
        self.up2 = conv_block(base * 8 + base * 8, base * 4, down=False, dropout=True)
        self.up3 = conv_block(base * 4 + base * 4, base * 2, down=False)
        self.up4 = conv_block(base * 2 + base * 2, base, down=False)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(base + base, out_channels, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x, cond):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        cond_map = self.cond_embed(cond)                       # (B, base*8, 1, 1)
        cond_map = cond_map.expand(-1, -1, d5.shape[2], d5.shape[3])
        bottleneck = d5 + cond_map                              # additive conditioning injection

        u1 = self.up1(torch.cat([bottleneck, d5], 1))
        u2 = self.up2(torch.cat([u1, d4], 1))
        u3 = self.up3(torch.cat([u2, d3], 1))
        u4 = self.up4(torch.cat([u3, d2], 1))
        out = self.final(torch.cat([u4, d1], 1))
        return out


class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN: classifies overlapping patches as real/fake rather than the whole image,
    which is what gives Pix2Pix its sharp high-frequency output (Master Plan 2.2)."""

    def __init__(self, in_channels: int = 4, base: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, base, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base, base * 2, 4, 2, 1), nn.InstanceNorm2d(base * 2), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 2, base * 4, 4, 2, 1), nn.InstanceNorm2d(base * 4), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 4, base * 8, 4, 1, 1), nn.InstanceNorm2d(base * 8), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base * 8, 1, 4, 1, 1),  # raw logits, per-patch
        )

    def forward(self, ir, rgb):
        x = torch.cat([ir, rgb], dim=1)
        return self.model(x)


if __name__ == "__main__":
    g = UNetColorizationGenerator()
    d = PatchGANDiscriminator()
    ir = torch.randn(2, 1, 256, 256)
    cond = torch.rand(2, 2)
    rgb = g(ir, cond)
    print("Generator:", ir.shape, "+ cond", cond.shape, "-> RGB", rgb.shape)
    assert rgb.shape == (2, 3, 256, 256)

    patch_out = d(ir, rgb)
    print("Discriminator patch map:", patch_out.shape)
    print("OK")
