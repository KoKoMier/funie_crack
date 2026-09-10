"""Jointly train the FUnIE-GAN/CrackFormer-II feature-fusion network."""
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from nets import FunieCrackFusion
from utils import FusionTripletDataset, grouped_train_val_split, joint_loss, read_manifest


def resolve_path(root, value):
    path = Path(value)
    return path if path.is_absolute() else root / path


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_funie_trainable(model, trainable):
    for parameter in model.funie.parameters():
        parameter.requires_grad_(trainable)
    model.funie.train(trainable)


def freeze_batchnorm_stats(module):
    for child in module.modules():
        if isinstance(child, torch.nn.modules.batchnorm._BatchNorm):
            child.eval()


def build_optimizer(model, config):
    return torch.optim.AdamW([
        {"params": model.funie.parameters(), "lr": config["lr_funie"]},
        {"params": model.crack.parameters(), "lr": config["lr_crack"]},
        {"params": model.fusions.parameters(), "lr": config["lr_fusion"]},
    ], weight_decay=config["weight_decay"])


def metrics_from_counts(tp, fp, fn):
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return precision, recall, f1


def run_epoch(model, loader, device, weights, optimizer=None, grad_clip=1.0, max_batches=None):
    training = optimizer is not None
    totals = {key: 0.0 for key in ("loss", "seg", "side", "enh", "edge")}
    tp = fp = fn = 0
    sample_count = 0
    for batch_index, batch in enumerate(loader, 1):
        degraded = batch["degraded"].to(device, non_blocking=True)
        reference = batch["reference"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            result = model(degraded)
            loss, parts = joint_loss(result, reference, label, weights)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at batch {batch_index}; AMP must remain disabled")
            if training:
                loss.backward()
                clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        batch_size = degraded.size(0)
        sample_count += batch_size
        totals["loss"] += loss.item() * batch_size
        for name, value in parts.items():
            totals[name] += value.item() * batch_size
        prediction = torch.sigmoid(result["logits"]) >= 0.5
        truth = label >= 0.5
        tp += (prediction & truth).sum().item()
        fp += (prediction & ~truth).sum().item()
        fn += (~prediction & truth).sum().item()
        if max_batches is not None and batch_index >= max_batches:
            break

    count = sample_count
    averages = {name: value / count for name, value in totals.items()}
    averages["precision"], averages["recall"], averages["f1"] = metrics_from_counts(tp, fp, fn)
    return averages


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_f1, config):
    torch.save({
        "epoch": epoch,
        "best_f1": best_f1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "config": config,
    }, path)


def main():
    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(project_root / "configs" / "train.yaml"))
    parser.add_argument("--resume", default="")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8-sig") as stream:
        config = yaml.safe_load(stream)

    seed_everything(config["seed"])
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(config["device"])
    data_root = resolve_path(project_root, config["data_root"])
    rows = read_manifest(data_root, config["manifest"])
    train_rows, val_rows = grouped_train_val_split(rows, config["val_ratio"], config["seed"])
    size = (config["image_height"], config["image_width"])
    train_data = FusionTripletDataset(data_root, train_rows, size, augment=True)
    val_data = FusionTripletDataset(data_root, val_rows, size, augment=False)
    train_loader = DataLoader(train_data, batch_size=config["batch_size"], shuffle=True,
                              num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_data, batch_size=config["batch_size"], shuffle=False,
                            num_workers=config["num_workers"], pin_memory=device.type == "cuda")

    model = FunieCrackFusion().to(device)
    optimizer = build_optimizer(model, config)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    output_dir = resolve_path(project_root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    start_epoch, best_f1 = 1, -1.0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint["epoch"] + 1
        best_f1 = checkpoint["best_f1"]
        print(f"Resumed: {args.resume}")
    else:
        model.load_pretrained(
            resolve_path(project_root, config["funie_checkpoint"]),
            resolve_path(project_root, config["crack_checkpoint"]),
            map_location=device,
        )
        print("Loaded both pretrained networks")

    print(f"Device: {device}; train={len(train_data)}; val={len(val_data)}; size={size}")
    if args.dry_run:
        set_funie_trainable(model, False)
        model.train()
        model.funie.eval()
        stats = run_epoch(model, train_loader, device, config["loss_weights"],
                          optimizer, config["grad_clip"], max_batches=1)
        print("Dry run passed:", " ".join(f"{k}={v:.4f}" for k, v in stats.items()))
        return

    log_path = output_dir / "history.csv"
    log_exists = log_path.exists() and start_epoch > 1
    for epoch in range(start_epoch, config["epochs"] + 1):
        model.train()
        frozen = epoch <= config["freeze_funie_epochs"]
        set_funie_trainable(model, not frozen)
        if config.get("freeze_funie_batchnorm", True):
            freeze_batchnorm_stats(model.funie)
        train_stats = run_epoch(model, train_loader, device, config["loss_weights"],
                                optimizer, config["grad_clip"])
        model.eval()
        with torch.no_grad():
            val_stats = run_epoch(model, val_loader, device, config["loss_weights"])
        scheduler.step()

        print(
            f"Epoch {epoch:03d}/{config['epochs']} funie={'frozen' if frozen else 'train'} "
            f"train_loss={train_stats['loss']:.4f} val_loss={val_stats['loss']:.4f} "
            f"P={val_stats['precision']:.4f} R={val_stats['recall']:.4f} F1={val_stats['f1']:.4f}"
        )
        improved = val_stats["f1"] > best_f1
        best_f1 = max(best_f1, val_stats["f1"])
        save_checkpoint(output_dir / "last.pth", model, optimizer, scheduler,
                        epoch, best_f1, config)
        if improved:
            save_checkpoint(output_dir / "best.pth", model, optimizer, scheduler,
                            epoch, best_f1, config)
        if epoch % config["save_every"] == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:03d}.pth", model, optimizer,
                            scheduler, epoch, best_f1, config)

        row = {"epoch": epoch, "funie_frozen": frozen, "best_f1": best_f1}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"val_{k}": v for k, v in val_stats.items()})
        with log_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not log_exists:
                writer.writeheader()
                log_exists = True
            writer.writerow(row)


if __name__ == "__main__":
    main()
