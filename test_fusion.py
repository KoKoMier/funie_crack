"""Load both pretrained networks and verify feature-level fused inference."""
import argparse
from pathlib import Path

import torch

from nets import FunieCrackFusion


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--funie", default=str(root / "models" / "funie_generator.pth"))
    parser.add_argument("--crack", default=str(root / "models" / "crackformer_crack537.pth"))
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FunieCrackFusion()
    model.load_pretrained(args.funie, args.crack, map_location=device)
    model.to(device).eval()
    sample = torch.randn(1, 3, args.height, args.width, device=device)
    with torch.no_grad():
        result = model(sample, return_features=True)

    print(f"device: {device}")
    print(f"parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"enhanced: {tuple(result['enhanced'].shape)}")
    print(f"logits: {tuple(result['logits'].shape)}")
    print(f"side outputs: {[tuple(x.shape) for x in result['side_outputs']]}")
    print(f"finite logits: {torch.isfinite(result['logits']).all().item()}")
    for name, block in model.fusions.items():
        print(f"{name} strength: {block.strength.item():.4f}")


if __name__ == "__main__":
    main()
