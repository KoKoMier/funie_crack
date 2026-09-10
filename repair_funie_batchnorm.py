"""Restore pretrained FUnIE BatchNorm statistics in a fusion checkpoint."""
import argparse
from pathlib import Path

import torch


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(root / "checkpoints" / "fusion" / "best.pth"))
    parser.add_argument("--funie", default=str(root / "models" / "funie_generator.pth"))
    parser.add_argument("--output", default=str(root / "checkpoints" / "fusion" / "best_bn_fixed.pth"))
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if "model_state" not in checkpoint:
        raise ValueError("Expected a complete fusion training checkpoint")
    pretrained = torch.load(args.funie, map_location="cpu")
    restored = []
    for key, value in pretrained.items():
        if key.endswith(("running_mean", "running_var", "num_batches_tracked")):
            fusion_key = f"funie.{key}"
            if fusion_key not in checkpoint["model_state"]:
                raise KeyError(fusion_key)
            checkpoint["model_state"][fusion_key] = value.clone()
            restored.append(fusion_key)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    print(f"Restored {len(restored)} BatchNorm buffers")
    print(f"Saved: {output.resolve()}")


if __name__ == "__main__":
    main()
