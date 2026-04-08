#!/usr/bin/env python3
"""
Collect images from a GelSight tactile sensor.

Modes
-----
bg   : Capture a single background image (no object on sensor).
       Saved as gelsight/background.png. Press k to confirm, auto-exits after saving.

data : Collect training images. Press k to save each frame, q/ESC to quit.
       Images are saved as 000000.png, 000001.png, ... under --save_dir.

Usage:
    python collect_data/collect_data.py --mode bg
    python collect_data/collect_data.py --mode data --save_dir data/my_object
"""

import argparse
import os

import cv2
import yaml
from gs_sdk.gs_device import Camera, FastCamera



def main():
    parser = argparse.ArgumentParser(
        description="Collect GelSight images."
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["bg", "data"],
        default="data",
        help="'bg' to capture background, 'data' to collect training images (default: data)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sensor.yaml",
        help="Path to sensor config YAML (default: configs/sensor.yaml)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data",
        help="[data mode only] Directory to save captured images (default: data/)",
    )
    parser.add_argument(
        "--streamer",
        type=str,
        choices=["opencv", "ffmpeg"],
        default="opencv",
        help="Camera backend: 'opencv' or 'ffmpeg'",
    )
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(project_root, config_path)

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    device_name = cfg["device_name"]
    imgh = int(cfg["imgh"])
    imgw = int(cfg["imgw"])

    if args.streamer == "opencv":
        device = Camera(device_name, imgh, imgw)
    else:
        device = FastCamera(
            device_name, imgh, imgw,
            int(cfg["raw_imgh"]), int(cfg["raw_imgw"]),
            cfg["framerate"],
        )

    device.connect()
    print("Connected to GelSight sensor.")

    # ------------------------------------------------------------------
    # Background mode
    # ------------------------------------------------------------------
    if args.mode == "bg":
        gelsight_dir = os.path.join(project_root, "gelsight")
        os.makedirs(gelsight_dir, exist_ok=True)
        save_path = os.path.join(gelsight_dir, "background.png")

        print("=" * 50)
        print("  Background capture mode")
        print("  Make sure NO object is on the sensor.")
        print("  k       - confirm and save background")
        print("  q / ESC - quit without saving")
        print("=" * 50)

        while True:
            image = device.get_image()
            cv2.imshow("Background Capture", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("k"):
                cv2.imwrite(save_path, image)
                print(f"\nBackground saved: {save_path}")
                print("Capture successful. Exiting.")
                break
            elif key == ord("q") or key == 27:
                print("Cancelled.")
                break

    # ------------------------------------------------------------------
    # Data collection mode
    # ------------------------------------------------------------------
    else:
        save_dir = args.save_dir
        if not os.path.isabs(save_dir):
            save_dir = os.path.join(project_root, save_dir)
        os.makedirs(save_dir, exist_ok=True)

        image_counter = len(
            [f for f in os.listdir(save_dir) if f.lower().endswith(".png")]
        )

        print("=" * 50)
        print("  Training data collection mode")
        print("  k       - save current image")
        print("  q / ESC - quit")
        print("=" * 50)
        print(f"Images will be saved to: {save_dir}")
        if image_counter > 0:
            print(f"Resuming from {image_counter} existing images.")
        print()

        while True:
            image = device.get_image()
            cv2.imshow("Data Collection", image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("k"):
                filename = f"{image_counter:06d}.png"
                filepath = os.path.join(save_dir, filename)
                cv2.imwrite(filepath, image)
                image_counter += 1
                print(f"Saved: {filename}")
            elif key == ord("q") or key == 27:
                break

        print(f"\nTotal images saved: {image_counter}")
        if image_counter > 0:
            print(f"Saved in: {save_dir}")

    device.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
