"""FUnIE-GAN generator that exposes multi-scale intermediate features."""
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, bn=True):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if bn:
            layers.append(nn.BatchNorm2d(out_size, momentum=0.8))
        layers.append(nn.LeakyReLU(0.2))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size, momentum=0.8),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip_input):
        x = self.model(x)
        return torch.cat((x, skip_input), dim=1)


class GeneratorFunieGANFeatures(nn.Module):
    """Weight-compatible FUnIE-GAN generator with optional feature output.

    Calling ``forward(x)`` preserves the original API and returns only the
    enhanced image. Calling ``forward(x, return_features=True)`` returns
    ``(enhanced, features)``, where features is ordered from shallow to deep.
    """

    encoder_channels = OrderedDict(
        d1=32,
        d2=128,
        d3=256,
        d4=256,
        d5=256,
    )
    decoder_channels = OrderedDict(
        u1=512,
        u2=512,
        u3=256,
        u4=64,
    )

    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Names and shapes intentionally match the original implementation so
        # its generator state_dict can be loaded with strict=True.
        self.down1 = UNetDown(in_channels, 32, bn=False)
        self.down2 = UNetDown(32, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 256)
        self.down5 = UNetDown(256, 256, bn=False)
        self.up1 = UNetUp(256, 256)
        self.up2 = UNetUp(512, 256)
        self.up3 = UNetUp(512, 128)
        self.up4 = UNetUp(256, 32)
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(64, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x, return_features=False):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        enhanced = self.final(u4)

        if not return_features:
            return enhanced

        features = {
            "encoder": OrderedDict(d1=d1, d2=d2, d3=d3, d4=d4, d5=d5),
            "decoder": OrderedDict(u1=u1, u2=u2, u3=u3, u4=u4),
        }
        return enhanced, features


def load_funiegan_weights(model, checkpoint, map_location="cpu"):
    """Load original, DataParallel, or wrapped FUnIE-GAN checkpoints."""
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location=map_location)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint format: {checkpoint}")
    state = {
        key.removeprefix("module.").removeprefix("generator."): value
        for key, value in state.items()
    }
    model.load_state_dict(state, strict=True)
    return model
