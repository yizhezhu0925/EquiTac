#!/usr/bin/env python3

import argparse
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from .dataset import NormalDataset, NormalMapGenerator
from .model import E2DirectionIrrep


def train(cfg: dict):
    seed = cfg["training"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Paths are relative to the repo root (parent of this package)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def abspath(p):
        return p if os.path.isabs(p) else os.path.join(repo_root, p)

    normal_gen = NormalMapGenerator(
        model_pth=abspath(cfg["sensor"]["calib_model_path"]),
        config_yaml=abspath(cfg["sensor"]["config_path"]),
        bg_image_path=abspath(cfg["sensor"]["bg_image_path"]),
        device="cpu",
    )

    dataset = NormalDataset(
        data_dir=abspath(cfg["data"]["data_dir"]),
        img_size=cfg["data"]["img_size"],
        aug_mode=cfg["augmentation"]["aug_mode"],
        normal_gen=normal_gen,
    )

    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
    )

    N_group = cfg["model"]["N"]
    img_size = cfg["data"]["img_size"]
    aug_mode = cfg["augmentation"]["aug_mode"]
    loss_type = cfg["training"]["loss_type"]

    model = E2DirectionIrrep(N=N_group).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])
    try:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5, verbose=True, mode="min"
        )
    except TypeError:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )

    save_dir = abspath(cfg["training"]["save_dir"])
    os.makedirs(save_dir, exist_ok=True)

    model_name = f"normal_irrep_N{N_group}_img{img_size}_{loss_type}_{aug_mode}_best.pth"
    best_path = os.path.join(save_dir, model_name)

    epochs = cfg["training"]["epochs"]
    best_loss = float("inf")
    best_epoch = 0
    losses = []

    print(f"\n{'='*60}")
    print(f"E2 Orientation Model  |  aug_mode={aug_mode}  |  N={N_group}")
    print(f"Dataset: {len(dataset)} samples  |  Batch: {cfg['training']['batch_size']}")
    print(f"Loss: {loss_type}  |  LR: {cfg['training']['lr']}")
    print(f"Save: {best_path}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0

        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            v_unit, _ = model(images)
            if loss_type == "mse":
                loss = F.mse_loss(v_unit, targets)
            else:
                raise ValueError(f"Unknown loss_type: {loss_type}")
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        losses.append(avg_loss)
        scheduler.step(avg_loss)

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs}  |  Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "train_loss": avg_loss,
                    "N": N_group,
                    "img_size": img_size,
                    "aug_mode": aug_mode,
                    "loss_type": loss_type,
                },
                best_path,
            )
            print(f"  -> Saved best model (epoch {epoch}, loss {avg_loss:.6f})")

    print(f"\nBest loss: {best_loss:.6f} at epoch {best_epoch}")

    plt.figure(figsize=(10, 6))
    plt.plot(losses, linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(f"Train Loss ({loss_type.upper()})")
    plt.title(f"Training Loss  |  aug_mode={aug_mode}")
    plt.grid(True, alpha=0.3)
    min_idx = int(np.argmin(losses))
    plt.plot(min_idx + 1, losses[min_idx], "ro", markersize=10)
    curve_path = os.path.join(save_dir, f"loss_curve_{aug_mode}.png")
    plt.savefig(curve_path, dpi=160, bbox_inches="tight")
    print(f"Loss curve saved: {curve_path}")


def main():
    parser = argparse.ArgumentParser(description="Train E2 orientation model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/train.yaml",
        help="Path to training config YAML (default: configs/train.yaml)",
    )
    parser.add_argument(
        "--aug_mode",
        type=str,
        choices=["none", "8dir", "full"],
        default="full",
        help="Override aug_mode from config. Options: none | 8dir | full",
    )
    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    config_path = args.config if os.path.isabs(args.config) else os.path.join(repo_root, args.config)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    if args.aug_mode is not None:
        cfg["augmentation"]["aug_mode"] = args.aug_mode

    train(cfg)


if __name__ == "__main__":
    main()
