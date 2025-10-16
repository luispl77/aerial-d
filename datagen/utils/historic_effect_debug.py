#!/usr/bin/env python3
"""Generate debug visuals for historic effects on random Aerial-D image."""

import argparse
import random
from pathlib import Path
from typing import List

import cv2

import numpy as np


def add_film_grain(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def adjust_contrast_gamma(image: np.ndarray, contrast: float = 0.8, gamma: float = 1.2) -> np.ndarray:
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    mean_val = np.mean(gamma_corrected)
    contrasted = (gamma_corrected - mean_val) * contrast + mean_val
    return np.clip(contrasted, 0, 255).astype(np.uint8)


def apply_sepia(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    sepia_filter = np.array(
        [[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]]
    )
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255)
    return sepia_image.astype(np.uint8)


def add_noise(image: np.ndarray) -> np.ndarray:
    noise = np.random.randint(0, 50, image.shape, dtype="uint8")
    return cv2.add(image, noise)


def apply_basic_bw_effect(image: np.ndarray) -> tuple[np.ndarray, str]:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    return gray, "Basic_BW"


def apply_bw_grain_effect(image: np.ndarray) -> tuple[np.ndarray, str]:
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    adjusted = adjust_contrast_gamma(gray, contrast=0.85, gamma=1.1)
    grainy = add_film_grain(adjusted, intensity=0.1)
    return grainy, "BW_Grain"


def apply_sepia_with_noise_effect(image: np.ndarray) -> tuple[np.ndarray, str]:
    sepia_image = apply_sepia(image)
    noisy_sepia = add_noise(sepia_image)
    return noisy_sepia, "Sepia_Noise"


def collect_image_paths(dataset_root: Path, split: str | None = None) -> List[Path]:
    """Collect image paths from the specified dataset root."""
    splits = [split] if split else ["train", "val"]
    image_paths: List[Path] = []

    for split_name in splits:
        img_dir = dataset_root / split_name / "images"
        if img_dir.exists():
            image_paths.extend(sorted(img_dir.glob("*.png")))

    return image_paths


def save_image(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Failed to write image to {path}")


def generate_debug_images(dataset_root: Path, output_dir: Path, split: str | None, seed: int) -> Path:
    """Select an image and save historic effect variants; returns output folder."""
    random.seed(seed)

    image_paths = collect_image_paths(dataset_root, split)
    if not image_paths:
        raise FileNotFoundError("No images found in the specified dataset path.")

    chosen_path = random.choice(image_paths)
    original_bgr = cv2.imread(str(chosen_path), cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise RuntimeError(f"Failed to read image: {chosen_path}")

    base_name = chosen_path.stem
    debug_dir = output_dir / base_name

    save_image(debug_dir / f"{base_name}_original.png", original_bgr)

    sepia_image, _ = apply_sepia_with_noise_effect(original_bgr.copy())
    save_image(debug_dir / f"{base_name}_sepia.png", sepia_image)

    bw_image, _ = apply_basic_bw_effect(original_bgr.copy())
    save_image(debug_dir / f"{base_name}_bw.png", bw_image)

    bw_grain_image, _ = apply_bw_grain_effect(original_bgr.copy())
    save_image(debug_dir / f"{base_name}_bw_grain.png", bw_grain_image)

    return debug_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate historic effect previews for Aerial-D images")
    parser.add_argument(
        "--dataset_root",
        type=Path,
        default=Path("/cfs/home/u035679/datasets/aeriald"),
        help="Root directory of the Aerial-D dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/cfs/home/u035679/aerialseg/datagen/utils/debug/historic_effects"),
        help="Directory where debug images will be saved",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val"],
        default=None,
        help="Dataset split to sample from (default: both)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = generate_debug_images(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        split=args.split,
        seed=args.seed if args.seed is not None else random.randrange(1 << 30),
    )

    print("Historic effect previews saved to:")
    print(output_dir.resolve())


if __name__ == "__main__":
    main()
