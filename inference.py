#!/usr/bin/env python3
"""
Real-time orientation inference from a GelSight tactile sensor.

Usage:
    python inference.py --config configs/inference.yaml

Controls:
    k  - capture and save current frame
    q / ESC - quit
"""

import argparse
import math
import os
import cv2
import numpy as np
import torch
import yaml

from gs_sdk.gs_device import Camera, FastCamera
from gs_sdk.gs_reconstruct import Reconstructor
from utils import erode_contact_mask, gxy2normal
from train.model import E2DirectionIrrep


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class NormalIrrepPredictor:
    def __init__(self, cfg: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        base = os.path.dirname(os.path.abspath(__file__))

        def abspath(p):
            return p if os.path.isabs(p) else os.path.join(base, p)

        self.img_size = cfg["model"]["img_size"]
        self.N = cfg["model"]["N"]

        with open(abspath(cfg["sensor"]["config_path"]), "r") as f:
            cam_cfg = yaml.safe_load(f)
        self.ppmm = float(cam_cfg["ppmm"])
        self.imgh = int(cam_cfg["imgh"])
        self.imgw = int(cam_cfg["imgw"])

        self.reconstructor = Reconstructor(
            abspath(cfg["sensor"]["calib_model_path"]), device="cpu"
        )
        bg = cv2.imread(abspath(cfg["sensor"]["bg_image_path"]))
        if bg is None:
            raise FileNotFoundError(
                f"Background image not found: {cfg['sensor']['bg_image_path']}"
            )
        if (bg.shape[0], bg.shape[1]) != (self.imgh, self.imgw):
            bg = cv2.resize(bg, (self.imgw, self.imgh), interpolation=cv2.INTER_AREA)
        self.reconstructor.load_bg(bg)

        self.model = self._load_model(abspath(cfg["model"]["checkpoint_path"]))

    def _load_model(self, model_path: str) -> torch.nn.Module:
        model = E2DirectionIrrep(N=self.N).to(self.device)
        with torch.no_grad():
            dummy = torch.randn(1, 3, self.img_size, self.img_size, device=self.device)
            _ = model(dummy)
        checkpoint = torch.load(model_path, map_location=self.device)
        state_dict = checkpoint.get("model_state", checkpoint)
        model_state = model.state_dict()
        for name, param in state_dict.items():
            if name in model_state and param.shape == model_state[name].shape:
                model_state[name].copy_(param)
        model.eval()
        return model

    def process_frame(self, tactile_image: np.ndarray):
        G, _, C = self.reconstructor.get_surface_info(tactile_image, self.ppmm)
        C = erode_contact_mask(C)
        N = gxy2normal(G)

        contact_area = float(np.sum(C))
        has_contact = contact_area > 500
        results = {"has_contact": has_contact, "contact_area": contact_area}

        if not has_contact:
            return results, N

        N_resized = cv2.resize(N, (self.img_size, self.img_size))
        tensor = torch.from_numpy(N_resized).float().permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            v_unit, v_raw = self.model(tensor)
            v_unit = v_unit.squeeze(0).cpu().numpy()
            v_raw = v_raw.squeeze(0).cpu().numpy()

        theta2 = math.atan2(v_raw[1], v_raw[0])
        theta = 0.5 * theta2
        angle_deg = math.degrees(theta)
        confidence = float(np.linalg.norm(v_raw))

        results.update(
            {
                "angle_deg": float(angle_deg),
                "confidence": confidence,
                "vector_unit": v_unit.tolist(),
            }
        )
        return results, N

    def visualize(self, tactile_image: np.ndarray, normal_map: np.ndarray, results: dict):
        panel_size = (360, 360)
        font = cv2.FONT_HERSHEY_SIMPLEX

        tactile_panel = cv2.resize(tactile_image, panel_size).copy()
        normal_vis = ((normal_map + 1.0) * 127.5).astype(np.uint8)
        normal_panel = cv2.resize(cv2.cvtColor(normal_vis, cv2.COLOR_RGB2BGR), panel_size).copy()

        cv2.putText(tactile_panel, "Tactile RGB", (10, 28), font, 0.7, (0, 255, 0), 2)
        cv2.putText(normal_panel, "Normal Map", (10, 28), font, 0.7, (0, 255, 0), 2)

        if results.get("has_contact", False):
            angle = results["angle_deg"]
            conf = results["confidence"]
            center = (panel_size[0] // 2, panel_size[1] // 2)
            radius = 80
            end_x = int(center[0] + radius * math.cos(math.radians(angle)))
            end_y = int(center[1] + radius * math.sin(math.radians(angle)))
            cv2.arrowedLine(tactile_panel, center, (end_x, end_y), (255, 255, 0), 6, tipLength=0.2)
            cv2.putText(tactile_panel, f"Angle: {angle:.1f} deg",
                        (10, panel_size[1] - 40), font, 0.6, (0, 255, 255), 2)
            cv2.putText(tactile_panel, f"Conf: {conf:.3f}",
                        (10, panel_size[1] - 15), font, 0.58, (0, 255, 255), 2)
        else:
            cv2.putText(tactile_panel, "No Contact",
                        (10, panel_size[1] - 25), font, 0.7, (0, 0, 255), 2)

        combined = np.zeros((panel_size[1], panel_size[0] * 2, 3), dtype=np.uint8)
        combined[:, : panel_size[0]] = tactile_panel
        combined[:, panel_size[0] :] = normal_panel
        cv2.putText(combined, "E2 Irrep Orientation", (10, 25), font, 0.6, (255, 255, 0), 1)
        return combined


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Real-time E2 orientation inference")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/inference.yaml",
        help="Path to inference config YAML",
    )
    args = parser.parse_args()

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_path)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    base = os.path.dirname(os.path.abspath(__file__))

    def abspath(p):
        return p if os.path.isabs(p) else os.path.join(base, p)

    predictor = NormalIrrepPredictor(cfg)

    with open(abspath(cfg["sensor"]["config_path"]), "r") as f:
        gel_cfg = yaml.safe_load(f)

    streamer = cfg["sensor"].get("streamer", "opencv")
    device_name = gel_cfg["device_name"]
    imgh, imgw = int(gel_cfg["imgh"]), int(gel_cfg["imgw"])

    if streamer == "opencv":
        camera = Camera(device_name, imgh, imgw)
    else:
        camera = FastCamera(
            device_name, imgh, imgw,
            int(gel_cfg["raw_imgh"]), int(gel_cfg["raw_imgw"]),
            gel_cfg["framerate"],
        )
    camera.connect()

    print("\n" + "=" * 60)
    print("E2 Irrep Orientation Inference")
    print("=" * 60)
    print("  q - Quit")
    print("=" * 60 + "\n")

    try:
        while True:
            tactile_image = camera.get_image()
            try:
                results, normal_map = predictor.process_frame(tactile_image)
                display = predictor.visualize(tactile_image, normal_map, results)
                cv2.imshow("E2 Irrep Orientation", display)
            except Exception as exc:
                print(f"Processing error: {exc}")
                cv2.imshow("E2 Irrep Orientation", tactile_image)

            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
