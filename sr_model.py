"""
Stage 1 — Super-Resolution Module
ESRGAN-style generator: Residual-in-Residual Dense Blocks (RRDB), no BatchNorm
(BN is deliberately omitted — see Master Plan Phase 1, 2.1: BN introduces artifacts
under the seasonally-shifting brightness-temperature statistics of IR input).

Input:  (B, 1, H, W)   single-channel IR (brightness temperature, normalized to [-1, 1])
Output: (B, 1, H*scale, W*scale)  sharpened / super-resolved IR
"""
import torch
import torch.nn as nn


class DenseBlock(nn.Module):
    """A single 5-conv densely-connected block (the 'D' in RRDB)."""

    def __init__(self, channels: int, growth: int = 32):
        super().__init__()
        self.c1 = nn.Conv2d(channels, growth, 3, 1, 1)
        self.c2 = nn.Conv2d(channels + growth, growth, 3, 1, 1)
        self.c3 = nn.Conv2d(channels + 2 * growth, growth, 3, 1, 1)
        self.c4 = nn.Conv2d(channels + 3 * growth, growth, 3, 1, 1)
        self.c5 = nn.Conv2d(channels + 4 * growth, channels, 3, 1, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.act(self.c1(x))
        x2 = self.act(self.c2(torch.cat([x, x1], 1)))
        x3 = self.act(self.c3(torch.cat([x, x1, x2], 1)))
        x4 = self.act(self.c4(torch.cat([x, x1, x2, x3], 1)))
        x5 = self.c5(torch.cat([x, x1, x2, x3, x4], 1))
        return x + 0.2 * x5  # residual scaling stabilizes deep stacks


class RRDB(nn.Module):
    """Residual-in-Residual Dense Block: 3 stacked DenseBlocks + outer residual."""

    def __init__(self, channels: int, growth: int = 32):
        super().__init__()
        self.db1 = DenseBlock(channels, growth)
        self.db2 = DenseBlock(channels, growth)
        self.db3 = DenseBlock(channels, growth)

    def forward(self, x):
        out = self.db1(x)
        out = self.db2(out)
        out = self.db3(out)
        return x + 0.2 * out


class PixelShuffleUpsample(nn.Module):
    """Sub-pixel convolution upsampling — avoids checkerboard artifacts of transposed conv."""

    def __init__(self, channels: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * scale * scale, 3, 1, 1)
        self.shuffle = nn.PixelShuffle(scale)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.shuffle(self.conv(x)))


class ESRGANGenerator(nn.Module):
    """
    Full Stage 1 SR generator.

    Args:
        in_channels: 1 for single-band IR input
        base_channels: feature width
        num_rrdb: how many RRDB blocks to stack (16-23 in the original paper; we default
                   smaller for tractable training on modest hardware, per Master Plan 3.2)
        scale: upsampling factor (2 or 4). Use scale=1 (no PixelShuffle stages) if you only
               want sharpening without changing pixel dimensions.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64,
                 num_rrdb: int = 8, growth: int = 32, scale: int = 2):
        super().__init__()
        assert scale in (1, 2, 4)
        self.head = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.body = nn.Sequential(*[RRDB(base_channels, growth) for _ in range(num_rrdb)])
        self.body_conv = nn.Conv2d(base_channels, base_channels, 3, 1, 1)

        ups = []
        n_up = {1: 0, 2: 1, 4: 2}[scale]
        for _ in range(n_up):
            ups.append(PixelShuffleUpsample(base_channels, scale=2))
        self.upsample = nn.Sequential(*ups)

        self.tail = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, in_channels, 3, 1, 1),
            nn.Tanh(),  # output normalized to [-1, 1], matches input normalization convention
        )

    def forward(self, x):
        feat = self.head(x)
        body_out = self.body_conv(self.body(feat))
        feat = feat + body_out  # global residual
        feat = self.upsample(feat)
        return self.tail(feat)


if __name__ == "__main__":
    # quick shape check
    m = ESRGANGenerator(scale=2, num_rrdb=4)
    x = torch.randn(2, 1, 128, 128)
    y = m(x)
    print("SR input:", x.shape, "-> output:", y.shape)
    assert y.shape == (2, 1, 256, 256)
    print("OK")
