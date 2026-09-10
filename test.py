"""Verify FUnIE-GAN checkpoint compatibility and feature-map shapes."""
import argparse
from pathlib import Path

import torch

from nets import GeneratorFunieGANFeatures, load_funiegan_weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GeneratorFunieGANFeatures()
    load_funiegan_weights(model, args.checkpoint, map_location=device)
    model.to(device).eval()

    sample = torch.randn(1, 3, args.height, args.width, device=device)
    with torch.no_grad():
        enhanced, features = model(sample, return_features=True)

    print(f"device: {device}")
    print(f"enhanced: {tuple(enhanced.shape)}")
    for group, values in features.items():
        for name, value in values.items():
            print(f"{group}.{name}: {tuple(value.shape)}")


if __name__ == "__main__":
    main()
