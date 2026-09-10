"""Feature-level fusion of FUnIE-GAN and CrackFormer-II."""
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .crackformer import crackformer
from .funiegan_features import GeneratorFunieGANFeatures, load_funiegan_weights


class FeatureFusionBlock(nn.Module):
    """Inject an enhancement feature through a gated residual adapter."""

    def __init__(self, crack_channels, funie_channels):
        super().__init__()
        groups = max(1, min(32, crack_channels // 4))
        self.adapter = nn.Sequential(
            nn.Conv2d(funie_channels, crack_channels, 1, bias=False),
            nn.GroupNorm(groups, crack_channels),
            nn.GELU(),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(crack_channels * 2, 1, 1),
            nn.Sigmoid(),
        )
        self.strength = nn.Parameter(torch.tensor(0.1))

    def forward(self, crack_feature, funie_feature):
        if funie_feature.shape[-2:] != crack_feature.shape[-2:]:
            funie_feature = F.interpolate(
                funie_feature, size=crack_feature.shape[-2:],
                mode="bilinear", align_corners=False,
            )
        adapted = self.adapter(funie_feature)
        gate = self.gate(torch.cat((crack_feature, adapted), dim=1))
        return crack_feature + torch.tanh(self.strength) * gate * adapted


class FunieCrackFusion(nn.Module):
    """End-to-end multi-scale feature fusion network."""

    def __init__(self):
        super().__init__()
        self.funie = GeneratorFunieGANFeatures()
        self.crack = crackformer()
        self.fusions = nn.ModuleDict(OrderedDict(
            stage1=FeatureFusionBlock(64, 64),
            stage2=FeatureFusionBlock(128, 256),
            stage3=FeatureFusionBlock(256, 512),
            stage4=FeatureFusionBlock(512, 512),
            stage5=FeatureFusionBlock(512, 256),
        ))

    def forward(self, inputs, return_features=False):
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError("Expected input shape (batch, 3, height, width)")
        if inputs.shape[-2] % 32 or inputs.shape[-1] % 32:
            raise ValueError("Input height and width must be divisible by 32")

        enhanced, funie_features = self.funie(inputs, return_features=True)
        enc, dec = funie_features["encoder"], funie_features["decoder"]

        out, i1, s1, a11, a12 = self.crack.down1(enhanced)
        out = self.fusions["stage1"](out, dec["u4"])
        out, i2, s2, a21, a22 = self.crack.down2(out)
        out = self.fusions["stage2"](out, dec["u3"])
        out, i3, s3, a31, a32, a33 = self.crack.down3(out)
        out = self.fusions["stage3"](out, dec["u2"])
        out, i4, s4, a41, a42, a43 = self.crack.down4(out)
        out = self.fusions["stage4"](out, dec["u1"])
        out, i5, s5, a51, a52, a53 = self.crack.down5(out)
        out = self.fusions["stage5"](out, enc["d5"])

        a54, a55, up5 = self.crack.up5(out, i5, s5)
        a44, a45, up4 = self.crack.up4(up5, i4, s4)
        a34, a35, up3 = self.crack.up3(up4, i3, s3)
        a23, up2 = self.crack.up2(up3, i2, s2)
        a13, up1 = self.crack.up1(up2, i1, s1)

        att1 = self.crack.LABlock_1([a11, a13])
        att2 = self.crack.LABlock_2([a21, a23])
        att3 = self.crack.LABlock_3([a31, a32, a34, a35])
        att4 = self.crack.LABlock_4([a41, a42, a44, a45])
        att5 = self.crack.LABlock_5([a51, a52, a54, a55])

        size = inputs.shape[-2:]
        f5 = self.crack.fuse5(a53, up5, size, att5)
        f4 = self.crack.fuse4(a43, up4, size, att4)
        f3 = self.crack.fuse3(a33, up3, size, att3)
        f2 = self.crack.fuse2(a22, up2, size, att2)
        f1 = self.crack.fuse1(a12, up1, size, att1)
        logits = self.crack.final(torch.cat((f5, f4, f3, f2, f1), dim=1))

        result = {"enhanced": enhanced, "logits": logits,
                  "side_outputs": (f5, f4, f3, f2, f1)}
        if return_features:
            result["funie_features"] = funie_features
        return result

    def load_pretrained(self, funie_checkpoint, crack_checkpoint, map_location="cpu"):
        load_funiegan_weights(self.funie, funie_checkpoint, map_location)
        state = torch.load(Path(crack_checkpoint), map_location=map_location)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        state = {k.removeprefix("module."): v for k, v in state.items()}
        self.crack.load_state_dict(state, strict=True)
        return self
