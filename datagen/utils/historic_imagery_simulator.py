#!/usr/bin/env python3
"""
Historic Aerial Imagery Simulator

This script takes modern aerial images from the dataset and applies various filters
and effects to simulate historic black and white aerial photography from the 1940s.

Usage:
    python utils/historic_imagery_simulator.py L1840
    python utils/historic_imagery_simulator.py P0000
    python utils/historic_imagery_simulator.py --random-seed 42
"""

import os
import sys
import glob
import random
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List
import argparse

def find_image_path(image_id: str) -> Optional[str]:
    """
    Find the full path to an image based on its ID.
    
    Args:
        image_id: Either L1840 format or P0000 format
        
    Returns:
        Full path to the image file, or None if not found
    """
    # Search patterns for different naming conventions
    search_patterns = [
        f"dataset/patches/train/images/{image_id}_patch_0.png",
        f"dataset/patches/val/images/{image_id}_patch_0.png",
        f"dataset/patches/train/images/{image_id}_patch_*.png",
        f"dataset/patches/val/images/{image_id}_patch_*.png",
    ]
    
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        if matches:
            return matches[0]  # Return first match
    
    return None

def get_random_image_path(seed: int) -> Tuple[Optional[str], str]:
    """
    Get a random image from the dataset using the provided seed.
    
    Args:
        seed: Random seed for reproducible results
        
    Returns:
        Tuple of (image_path, image_id) or (None, "") if no images found
    """
    # Set random seed
    random.seed(seed)
    
    # Find all images in the dataset
    search_patterns = [
        "dataset/patches/train/images/*.png",
        "dataset/patches/val/images/*.png",
    ]
    
    all_images = []
    for pattern in search_patterns:
        all_images.extend(glob.glob(pattern))
    
    if not all_images:
        return None, ""
    
    # Pick a random image
    chosen_path = random.choice(all_images)
    
    # Extract image ID from path
    filename = os.path.basename(chosen_path)
    if "_patch_" in filename:
        image_id = filename.split("_patch_")[0]
    else:
        image_id = filename.replace(".png", "")
    
    return chosen_path, image_id

def add_film_grain(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Add film grain noise to simulate old photography."""
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def simulate_old_lens_distortion(image: np.ndarray, strength: float = 0.1) -> np.ndarray:
    """Simulate slight lens distortion and vignetting from old cameras."""
    h, w = image.shape[:2]
    
    # Create vignette effect
    center_x, center_y = w // 2, h // 2
    max_distance = np.sqrt(center_x**2 + center_y**2)
    
    y, x = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    vignette_mask = 1 - (distance_from_center / max_distance) * strength
    vignette_mask = np.clip(vignette_mask, 0.7, 1.0)
    
    if len(image.shape) == 3:
        vignette_mask = np.stack([vignette_mask] * 3, axis=2)
    
    return (image * vignette_mask).astype(np.uint8)

def adjust_contrast_gamma(image: np.ndarray, contrast: float = 0.8, gamma: float = 1.2) -> np.ndarray:
    """Adjust contrast and gamma to simulate old film characteristics."""
    # Apply gamma correction
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    
    # Apply contrast adjustment
    mean_val = np.mean(gamma_corrected)
    contrasted = (gamma_corrected - mean_val) * contrast + mean_val
    
    return np.clip(contrasted, 0, 255).astype(np.uint8)

def apply_sepia(image: np.ndarray) -> np.ndarray:
    """Apply sepia filter using transformation matrix."""
    # Create sepia filter
    sepia_filter = np.array([[0.272, 0.534, 0.131], 
                            [0.349, 0.686, 0.168], 
                            [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255)  # Ensure valid range
    return sepia_image.astype(np.uint8)

def add_noise(image: np.ndarray) -> np.ndarray:
    """Add random noise to simulate old photography grain."""
    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    noisy_image = cv2.add(image, noise)
    return noisy_image

def apply_basic_bw_effect(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Basic black and white conversion."""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    return gray, "Basic B&W"

def apply_bw_grain_effect(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """B&W with grain and contrast adjustment."""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Mild contrast adjustment
    adjusted = adjust_contrast_gamma(gray, contrast=0.85, gamma=1.1)
    
    # Light grain
    grainy = add_film_grain(adjusted, intensity=0.1)
    
    return grainy, "B&W + Grain"

def apply_sepia_with_noise_effect(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Apply sepia tone effect with noise for vintage look."""
    # Apply sepia effect
    sepia_image = apply_sepia(image)
    
    # Add noise
    noisy_sepia = add_noise(sepia_image)
    
    return noisy_sepia, "Sepia + Noise"

def create_comparison_image(original: np.ndarray, effects: List[Tuple[np.ndarray, str]], 
                          image_id: str) -> np.ndarray:
    """Create a side-by-side comparison image."""
    num_images = len(effects) + 1  # +1 for original
    
    # Calculate grid dimensions (2x2 grid for 4 images total)
    cols = 2
    rows = 2
    
    # Get dimensions
    h, w = original.shape[:2]
    
    # Create comparison canvas - make it RGB to handle both color and grayscale
    canvas_h = h * rows + 50 * (rows + 1)  # Extra space for labels
    canvas_w = w * cols + 50 * (cols + 1)
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255
    
    # Place original image (top-left) - keep it in color
    y_offset = 50
    x_offset = 50
    if len(original.shape) == 3:
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = original
    else:
        # If somehow original is grayscale, convert to 3-channel
        orig_3ch = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        canvas[y_offset:y_offset+h, x_offset:x_offset+w] = orig_3ch
    
    # Add label for original
    cv2.putText(canvas, "Original", (x_offset, y_offset - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Place effect images
    positions = [(0, 1), (1, 0), (1, 1)]  # Top-right, bottom-left, bottom-right
    
    for idx, (effect_img, effect_name) in enumerate(effects):
        if idx < len(positions):
            row, col = positions[idx]
            
            y_pos = 50 + row * (h + 50)
            x_pos = 50 + col * (w + 50)
            
            # Handle both grayscale and color effect images
            if len(effect_img.shape) == 2:
                # Convert grayscale to 3-channel for consistent display
                effect_3ch = cv2.cvtColor(effect_img, cv2.COLOR_GRAY2BGR)
                canvas[y_pos:y_pos+h, x_pos:x_pos+w] = effect_3ch
            else:
                canvas[y_pos:y_pos+h, x_pos:x_pos+w] = effect_img
            
            # Add label
            cv2.putText(canvas, effect_name, (x_pos, y_pos - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add title
    title = f"Historic Effects Comparison - {image_id}"
    cv2.putText(canvas, title, (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    return canvas

def main():
    parser = argparse.ArgumentParser(description='Simulate historic aerial imagery effects')
    parser.add_argument('image_id', nargs='?', help='Image ID (e.g., L1840, P0000)')
    parser.add_argument('--random-seed', type=int, help='Random seed to pick an image randomly from dataset')
    parser.add_argument('--output-dir', default='utils/debug', help='Output directory for debug images')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine image path and ID
    if args.random_seed is not None:
        image_path, image_id = get_random_image_path(args.random_seed)
        if not image_path:
            print("Error: No images found in the dataset")
            sys.exit(1)
        print(f"Random seed {args.random_seed} selected: {image_id}")
    elif args.image_id:
        image_path = find_image_path(args.image_id)
        image_id = args.image_id
        if not image_path:
            print(f"Error: Could not find image for ID '{args.image_id}'")
            print("Make sure the image exists in dataset/patches/train/images/ or dataset/patches/val/images/")
            sys.exit(1)
    else:
        print("Error: Must provide either image_id or --random-seed")
        parser.print_help()
        sys.exit(1)
    
    print(f"Found image: {image_path}")
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        sys.exit(1)
    
    print(f"Loaded image with shape: {image.shape}")
    
    # Apply the 3 selected historic effects
    effects = [
        apply_basic_bw_effect(image),
        apply_bw_grain_effect(image),
        apply_sepia_with_noise_effect(image),
    ]
    
    # Create comparison image
    comparison = create_comparison_image(image, effects, image_id)
    
    # Save comparison image
    output_path = os.path.join(args.output_dir, f"{image_id}_historic_effects_comparison.png")
    cv2.imwrite(output_path, comparison)
    print(f"Saved comparison image to: {output_path}")
    
    # Save individual effect images
    for i, (effect_img, effect_name) in enumerate(effects):
        safe_name = effect_name.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "_")
        effect_path = os.path.join(args.output_dir, f"{image_id}_effect_{i+1}_{safe_name}.png")
        cv2.imwrite(effect_path, effect_img)
        print(f"Saved {effect_name} to: {effect_path}")

if __name__ == "__main__":
    main() 