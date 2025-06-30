#!/usr/bin/env python3
"""Utility to gather statistics from the LoveDA dataset."""

import os
import argparse
from collections import defaultdict
from typing import Dict, List
import glob
import numpy as np
from PIL import Image

LABELS = {
    0: "no-data",
    1: "background",
    2: "building",
    3: "road",
    4: "water",
    5: "barren",
    6: "forest",
    7: "agriculture",
}


def analyze_split(split_dir: str) -> Dict:
    """Analyze a single dataset split (Train, Val, or Test)."""
    stats = {
        "total_images": 0,
        "total_masks": 0,
        "scenes": {},
        "pixel_counts": np.zeros(len(LABELS), dtype=np.int64),
    }
    for scene in ["Urban", "Rural"]:
        scene_path = os.path.join(split_dir, scene)
        img_dir = os.path.join(scene_path, "images_png")
        mask_dir = os.path.join(scene_path, "masks_png")
        images = glob.glob(os.path.join(img_dir, "*.png"))
        stats["total_images"] += len(images)
        scene_info = {"images": len(images), "masks": 0}

        if os.path.isdir(mask_dir):
            masks = glob.glob(os.path.join(mask_dir, "*.png"))
            scene_info["masks"] = len(masks)
            stats["total_masks"] += len(masks)
            for m in masks:
                mask = np.array(Image.open(m))
                for val in range(len(LABELS)):
                    stats["pixel_counts"][val] += np.sum(mask == val)
        stats["scenes"][scene] = scene_info
    return stats


def analyze_dataset(dataset_path: str) -> Dict:
    dataset_stats = {}
    for split in ["Train", "Val", "Test"]:
        split_dir = os.path.join(dataset_path, split)
        if os.path.isdir(split_dir):
            dataset_stats[split] = analyze_split(split_dir)
    return dataset_stats


def print_report(stats: Dict) -> None:
    for split, s in stats.items():
        print(f"=== {split} ===")
        print(f"Images: {s['total_images']}")
        if s["total_masks"]:
            print(f"Masks: {s['total_masks']}")
        for scene, info in s["scenes"].items():
            print(f"  {scene}: {info['images']} images", end="")
            if info["masks"]:
                print(f", {info['masks']} masks")
            else:
                print()
        if s["total_masks"]:
            total_pixels = int(s["pixel_counts"].sum())
            print("Class pixel distribution:")
            for val, name in LABELS.items():
                count = int(s["pixel_counts"][val])
                if count > 0:
                    pct = 100 * count / total_pixels
                    print(f"  {name:10s}: {count} pixels ({pct:.2f}%)")
        print()


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze LoveDA dataset")
    parser.add_argument("dataset_path", help="Path to extracted LoveDA dataset")
    args = parser.parse_args(argv)

    stats = analyze_dataset(args.dataset_path)
    print_report(stats)


if __name__ == "__main__":
    main()

