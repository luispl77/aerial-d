#!/usr/bin/env python3

import os
import random
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Historic transformation functions (copied from test.py)
def add_film_grain(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Add film grain noise to simulate old photography."""
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

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
    # Ensure image is 3-channel color
    if len(image.shape) == 2:
        # Convert grayscale to color first
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Convert BGRA to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
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

def apply_basic_bw_effect(image: np.ndarray) -> np.ndarray:
    """Basic black and white conversion."""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    return gray

def apply_bw_grain_effect(image: np.ndarray) -> np.ndarray:
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
    
    return grainy

def apply_sepia_with_noise_effect(image: np.ndarray) -> np.ndarray:
    """Apply sepia tone effect with noise for vintage look."""
    # Apply sepia effect - this should work on the original color image
    sepia_image = apply_sepia(image)
    
    # Add noise
    noisy_sepia = add_noise(sepia_image)
    
    return noisy_sepia

def get_random_images(dataset_root, num_images=5, seed=42):
    """Get random images from the Aerial-D dataset."""
    random.seed(seed)
    
    # Look for images in train/images directory
    images_dir = os.path.join(dataset_root, 'train', 'images')
    if not os.path.exists(images_dir):
        # Try val/images if train doesn't exist
        images_dir = os.path.join(dataset_root, 'val', 'images')
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Could not find images directory in {dataset_root}")
    
    # Get all image files
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if len(image_files) < num_images:
        print(f"Warning: Only found {len(image_files)} images, using all of them")
        num_images = len(image_files)
    
    # Sample random images
    selected_images = random.sample(image_files, num_images)
    
    return [(os.path.join(images_dir, img), img) for img in selected_images]

def create_filter_comparison(image_path, output_path):
    """Create a side-by-side comparison of original and filtered images."""
    # Load image
    image_pil = Image.open(image_path).convert('RGB')
    image_rgb = np.array(image_pil)
    
    # Convert to BGR for OpenCV functions
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    
    # Apply the three filters
    bw_image = apply_basic_bw_effect(image_bgr)
    bw_grain_image = apply_bw_grain_effect(image_bgr)
    sepia_noise_image = apply_sepia_with_noise_effect(image_bgr)
    
    # Convert back to RGB for display
    # For grayscale images, convert to 3-channel for consistent display
    if len(bw_image.shape) == 2:
        bw_image_rgb = cv2.cvtColor(bw_image, cv2.COLOR_GRAY2RGB)
    else:
        bw_image_rgb = cv2.cvtColor(bw_image, cv2.COLOR_BGR2RGB)
        
    if len(bw_grain_image.shape) == 2:
        bw_grain_image_rgb = cv2.cvtColor(bw_grain_image, cv2.COLOR_GRAY2RGB)
    else:
        bw_grain_image_rgb = cv2.cvtColor(bw_grain_image, cv2.COLOR_BGR2RGB)
        
    sepia_noise_image_rgb = cv2.cvtColor(sepia_noise_image, cv2.COLOR_BGR2RGB)
    
    # Create figure with 4 subplots (1x4)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Remove axes and spacing
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    
    # Display images
    axes[0].imshow(image_rgb)
    axes[1].imshow(bw_image_rgb)
    axes[2].imshow(bw_grain_image_rgb)
    axes[3].imshow(sepia_noise_image_rgb)
    
    # Save the figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Generate historic filter examples from Aerial-D dataset')
    parser.add_argument('--dataset_root', type=str, default='../dataset', 
                       help='Root directory of the Aerial-D dataset')
    parser.add_argument('--output_dir', type=str, default='./filter_examples', 
                       help='Output directory for generated images')
    parser.add_argument('--num_images', type=int, default=5, 
                       help='Number of random images to process')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Looking for images in: {args.dataset_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processing {args.num_images} random images...")
    
    # Get random images
    try:
        image_paths = get_random_images(args.dataset_root, args.num_images, args.seed)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    print(f"Selected images: {[img[1] for img in image_paths]}")
    
    # Process each image
    for i, (image_path, image_name) in enumerate(image_paths):
        print(f"Processing {i+1}/{len(image_paths)}: {image_name}")
        
        # Create output filename
        base_name = os.path.splitext(image_name)[0]
        output_path = os.path.join(args.output_dir, f"{base_name}_filters.png")
        
        try:
            create_filter_comparison(image_path, output_path)
            print(f"  Saved: {output_path}")
        except Exception as e:
            print(f"  Error processing {image_name}: {e}")
    
    print(f"\nCompleted! Generated {len(image_paths)} filter comparison images in {args.output_dir}")

if __name__ == '__main__':
    main()