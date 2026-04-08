"""
Dataset and normal map utilities for orientation training.
"""

import math
import os
import random

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import Dataset


class NormalMapGenerator:
    """Converts raw GelSight tactile images into surface normal maps."""

    def __init__(self, model_pth: str, config_yaml: str, bg_image_path: str, device: str = "cpu"):
        from gs_sdk.gs_reconstruct import Reconstructor
        from utils import erode_contact_mask, gxy2normal

        self._erode = erode_contact_mask
        self._gxy2normal = gxy2normal

        with open(config_yaml, "r") as f:
            cfg = yaml.safe_load(f)
        self.ppmm = float(cfg["ppmm"])
        self.imgh = int(cfg.get("imgh", 0)) or None
        self.imgw = int(cfg.get("imgw", 0)) or None

        bg = cv2.imread(bg_image_path)
        if bg is None:
            raise FileNotFoundError(f"Background image not found: {bg_image_path}")

        if self.imgh is None or self.imgw is None:
            self.imgh, self.imgw = bg.shape[:2]

        if (bg.shape[0], bg.shape[1]) != (self.imgh, self.imgw):
            bg = cv2.resize(bg, (self.imgw, self.imgh), interpolation=cv2.INTER_AREA)

        self.recon = Reconstructor(model_pth, device=device)
        self.recon.load_bg(bg)
        self._cache: dict = {}

    def get_base_normal(self, img_path: str) -> np.ndarray:
        if img_path in self._cache:
            return self._cache[img_path]
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(img_path)
        if (img.shape[0], img.shape[1]) != (self.imgh, self.imgw):
            img = cv2.resize(img, (self.imgw, self.imgh), interpolation=cv2.INTER_AREA)
        G, _, C = self.recon.get_surface_info(img, self.ppmm)
        C = self._erode(C)
        N = self._gxy2normal(G)
        self._cache[img_path] = N.astype(np.float32)
        return self._cache[img_path]


    @staticmethod
    def rotate_field_and_vectors(N: np.ndarray, angle_deg: float) -> np.ndarray:
        h, w, _ = N.shape
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle_deg, 1.0)

        # Warp each channel; use (0,0,1) as border normal (flat surface)
        Nx = cv2.warpAffine(N[:, :, 0], M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        Ny = cv2.warpAffine(N[:, :, 1], M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        Nz = cv2.warpAffine(N[:, :, 2], M, (w, h), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=1.0)

        # Mask: 1 where original content exists, 0 for border
        mask = cv2.warpAffine(np.ones((h, w), np.float32), M, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        mask = (mask > 0.5).astype(np.float32)

        # Rotate the (Nx, Ny) vectors by angle_deg
        rad = math.radians(angle_deg)
        c, s = math.cos(rad), math.sin(rad)
        Nx2 = c * Nx - s * Ny
        Ny2 = s * Nx + c * Ny
        Nz2 = Nz

        # Apply mask: rotated content in foreground, (0,0,1) in border
        Nx_out = Nx2 * mask + 0.0 * (1.0 - mask)
        Ny_out = Ny2 * mask + 0.0 * (1.0 - mask)
        Nz_out = Nz2 * mask + 1.0 * (1.0 - mask)

        N_stack = np.stack([Nx_out, Ny_out, Nz_out], axis=-1)
        norm = np.linalg.norm(N_stack, axis=-1, keepdims=True) + 1e-8
        return np.clip(N_stack / norm, -1.0, 1.0).astype(np.float32)


# 8 discrete angles for "8dir" augmentation mode (multiples of 45°, full circle)
_8DIR_ANGLES = [k * 45.0 - 180.0 for k in range(8)]  # [-180, -135, ..., 135]


class NormalDataset(Dataset):
    """
    Dataset of GelSight normal maps with rotational augmentation.

    aug_mode options
    ----------------
    "none"  : no rotation; label is always 0°
    "8dir"  : rotate by a random multiple of 45° (8 discrete directions)
    "full"  : rotate by a uniformly random angle in [-180°, 180°]
    """

    def __init__(
        self,
        data_dir: str,
        img_size: int,
        aug_mode: str,
        normal_gen: NormalMapGenerator,
    ):
        self.paths = sorted(
            [os.path.join(data_dir, f) for f in os.listdir(data_dir)
             if f.lower().endswith(".png")]
        )
        if not self.paths:
            raise RuntimeError(f"No PNG files found in {data_dir}")

        assert aug_mode in ("none", "8dir", "full"), \
            f"aug_mode must be 'none', '8dir', or 'full', got '{aug_mode}'"

        self.img_size = img_size
        self.aug_mode = aug_mode
        self.normal_gen = normal_gen

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]

        if self.aug_mode == "none":
            theta_deg = 0.0
        elif self.aug_mode == "8dir":
            theta_deg = random.choice(_8DIR_ANGLES)
        else:  # full
            theta_deg = random.uniform(-90.0, 90.0)

        N0 = self.normal_gen.get_base_normal(img_path)
        N0 = cv2.resize(N0, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        N = self.normal_gen.rotate_field_and_vectors(N0, theta_deg) if theta_deg != 0.0 else N0

        img = torch.from_numpy(N).permute(2, 0, 1).float()
        theta_rad = math.radians(theta_deg)
        target = torch.tensor(
            [math.cos(2 * theta_rad), math.sin(2 * theta_rad)], dtype=torch.float32
        )
        return img, target
