"""Inference for the trained FUnIE-GAN/CrackFormer-II fusion model."""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nets import FunieCrackFusion


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def find_images(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        if input_path.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image format: {input_path}")
        return [input_path]
    if input_path.is_dir():
        images = sorted(path for path in input_path.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        if not images:
            raise RuntimeError(f"No images found in: {input_path}")
        return images
    raise FileNotFoundError(input_path)


def load_fusion_checkpoint(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state = checkpoint["model_state"]
        config = checkpoint.get("config", {})
        epoch = checkpoint.get("epoch", "unknown")
        best_f1 = checkpoint.get("best_f1", "unknown")
    else:
        state = checkpoint
        config, epoch, best_f1 = {}, "unknown", "unknown"
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return config, epoch, best_f1


def preprocess(image, height, width, device):
    resized = image.resize((width, height), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32).copy() / 127.5 - 1.0
    return torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to(device)


def enhanced_to_image(tensor, original_size):
    array = tensor[0].permute(1, 2, 0).detach().cpu().numpy()
    array = np.clip((array + 1.0) * 127.5, 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGB").resize(original_size, Image.Resampling.BICUBIC)


def probability_to_image(logits, original_size):
    probability = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
    probability = np.clip(probability * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(probability, mode="L").resize(original_size, Image.Resampling.BILINEAR)


def make_overlay(original, probability, threshold, alpha=0.55):
    base = np.asarray(original.convert("RGB"), dtype=np.float32).copy()
    mask = np.asarray(probability, dtype=np.uint8) >= round(threshold * 255)
    red = np.zeros_like(base)
    red[..., 0] = 255
    base[mask] = (1.0 - alpha) * base[mask] + alpha * red[mask]
    overlay = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), mode="RGB")
    return overlay, mask


def main():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="an image or a directory")
    parser.add_argument("--output_dir", default=str(project_root / "outputs" / "inference"))
    parser.add_argument("--checkpoint", default=str(project_root / "checkpoints" / "fusion" / "best.pth"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--height", type=int, default=0, help="0 uses checkpoint config")
    parser.add_argument("--width", type=int, default=0, help="0 uses checkpoint config")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        parser.error("--threshold must be between 0 and 1")

    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(device_name)

    model = FunieCrackFusion().to(device)
    config, epoch, best_f1 = load_fusion_checkpoint(model, args.checkpoint, device)
    model.eval()
    height = args.height or config.get("image_height", 256)
    width = args.width or config.get("image_width", 256)
    if height % 32 or width % 32:
        raise ValueError("Inference height and width must be divisible by 32")

    output_root = Path(args.output_dir)
    directories = {name: output_root / name
                   for name in ("enhanced", "probability", "mask", "overlay")}
    for directory in directories.values():
        directory.mkdir(parents=True, exist_ok=True)

    images = find_images(args.input)
    elapsed = []
    print(f"Device: {device}; checkpoint epoch: {epoch}; best F1: {best_f1}")
    print(f"Images: {len(images)}; network size: {height}x{width}; threshold: {args.threshold}")
    with torch.no_grad():
        for image_path in images:
            original = Image.open(image_path).convert("RGB")
            tensor = preprocess(original, height, width, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = model(tensor)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed.append(time.perf_counter() - start)

            enhanced = enhanced_to_image(result["enhanced"], original.size)
            probability = probability_to_image(result["logits"], original.size)
            overlay, mask_array = make_overlay(original, probability, args.threshold)
            mask = Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L")
            filename = f"{image_path.stem}.png"
            enhanced.save(directories["enhanced"] / filename)
            probability.save(directories["probability"] / filename)
            mask.save(directories["mask"] / filename)
            overlay.save(directories["overlay"] / filename)
            print(f"Processed: {image_path.name}")

    mean_time = sum(elapsed) / len(elapsed)
    print(f"Done: {output_root.resolve()}")
    print(f"Mean network time: {mean_time:.4f} s/image ({1.0 / mean_time:.2f} FPS)")


if __name__ == "__main__":
    main()
