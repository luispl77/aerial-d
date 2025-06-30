#!/usr/bin/env python3
"""
LoveDA Mask Visualizer
Visualizes LoveDA semantic segmentation masks with clear RGB colors for each class.
"""

import os
import numpy as np
import cv2
import argparse
from pathlib import Path

# LoveDA class color mapping (RGB values)
CLASS_COLORS = {
    0: [0, 0, 0],        # no-data: black
    1: [128, 128, 128],  # background: gray
    2: [255, 0, 0],      # building: red
    3: [128, 128, 128],  # road: dark gray  
    4: [0, 0, 255],      # water: blue
    5: [139, 69, 19],    # barren: brown
    6: [0, 255, 0],      # forest: green
    7: [255, 255, 0],    # agriculture: yellow
}

CLASS_NAMES = {
    0: 'no-data',
    1: 'background', 
    2: 'building',
    3: 'road',
    4: 'water',
    5: 'barren',
    6: 'forest',
    7: 'agriculture'
}

def colorize_mask(mask):
    """
    Convert a semantic segmentation mask to RGB colors.
    
    Args:
        mask: 2D numpy array with class IDs (0-7)
        
    Returns:
        3D numpy array (H, W, 3) with RGB colors
    """
    height, width = mask.shape
    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
    
    for class_id, color in CLASS_COLORS.items():
        colored_mask[mask == class_id] = color
    
    return colored_mask

def create_legend():
    """Create a legend image showing class colors and names."""
    legend_height = 30 * len(CLASS_COLORS)
    legend_width = 200
    legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
    
    for i, (class_id, color) in enumerate(CLASS_COLORS.items()):
        y_start = i * 30
        y_end = y_start + 25
        
        # Draw color rectangle
        legend[y_start:y_end, 10:40] = color
        
        # Add text label
        text = f"{class_id}: {CLASS_NAMES[class_id]}"
        cv2.putText(legend, text, (45, y_start + 18), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return legend

def analyze_mask(mask):
    """Analyze mask to show class distribution."""
    unique_classes, counts = np.unique(mask, return_counts=True)
    total_pixels = mask.size
    
    print("\nClass distribution:")
    print("-" * 40)
    for class_id, count in zip(unique_classes, counts):
        percentage = (count / total_pixels) * 100
        class_name = CLASS_NAMES.get(class_id, f"unknown_{class_id}")
        print(f"{class_id}: {class_name:<12} {count:>8} pixels ({percentage:>5.1f}%)")
    print("-" * 40)

def find_loveda_files(image_num, loveda_dir="LoveDA", split="Train", domain=None):
    """
    Find LoveDA image and mask files for a given image number.
    
    Args:
        image_num: LoveDA image number (e.g., "1366", "0", "1000")
        loveda_dir: Path to LoveDA dataset directory
        split: "Train" or "Val"
        domain: "Urban" or "Rural" (if None, searches both)
        
    Returns:
        tuple: (image_path, mask_path, domain_found) or (None, None, None) if not found
    """
    filename = f"{image_num}.png"
    
    domains_to_check = [domain] if domain else ["Urban", "Rural"]
    
    for dom in domains_to_check:
        image_path = os.path.join(loveda_dir, split, dom, "images_png", filename)
        mask_path = os.path.join(loveda_dir, split, dom, "masks_png", filename)
        
        if os.path.exists(image_path) and os.path.exists(mask_path):
            return image_path, mask_path, dom
    
    return None, None, None

def visualize_loveda_image(image_num, loveda_dir="LoveDA", split="Train", domain=None, 
                          output_dir="utils/debug", show_analysis=True):
    """
    Visualize a LoveDA image and its mask with RGB colors.
    
    Args:
        image_num: LoveDA image number (e.g., "1366", "0", "1000")
        loveda_dir: Path to LoveDA dataset directory
        split: "Train" or "Val"
        domain: "Urban" or "Rural" (if None, searches both)
        output_dir: Directory to save debug images
        show_analysis: Whether to print class distribution analysis
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find image and mask files
    image_path, mask_path, domain_found = find_loveda_files(image_num, loveda_dir, split, domain)
    
    if image_path is None:
        print(f"Error: Could not find LoveDA image {image_num} in {split} split!")
        if domain:
            print(f"Searched in {domain} domain only.")
        else:
            print("Searched in both Urban and Rural domains.")
        return
    
    print(f"Found LoveDA image {image_num} in {split}/{domain_found}")
    print(f"Image: {image_path}")
    print(f"Mask: {mask_path}")
    
    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        print(f"Error: Could not load image or mask files!")
        return
    
    print(f"Loaded image: {image.shape}")
    print(f"Loaded mask: {mask.shape}")
    
    # Analyze mask
    if show_analysis:
        analyze_mask(mask)
    
    # Colorize mask
    colored_mask = colorize_mask(mask)
    
    # Create legend
    legend = create_legend()
    
    # Save original image
    image_output = os.path.join(output_dir, f"{image_num}_{domain_found.lower()}_image.png")
    cv2.imwrite(image_output, image)
    print(f"Saved original image to: {image_output}")
    
    # Save colored mask
    colored_output = os.path.join(output_dir, f"{image_num}_{domain_found.lower()}_mask_colored.png")
    cv2.imwrite(colored_output, cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR))
    print(f"Saved colored mask to: {colored_output}")
    
    # Create overlay (colored mask on top of original image)
    colored_mask_bgr = cv2.cvtColor(colored_mask, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(image, 0.7, colored_mask_bgr, 0.3, 0)
    
    overlay_output = os.path.join(output_dir, f"{image_num}_{domain_found.lower()}_overlay.png")
    cv2.imwrite(overlay_output, overlay)
    print(f"Saved overlay to: {overlay_output}")
    
    # Save legend
    legend_output = os.path.join(output_dir, "loveda_legend.png") 
    cv2.imwrite(legend_output, cv2.cvtColor(legend, cv2.COLOR_RGB2BGR))
    print(f"Saved legend to: {legend_output}")
    
    # Create side-by-side comparison: original image | colored mask | overlay | legend
    h, w = image.shape[:2]
    legend_resized = cv2.resize(cv2.cvtColor(legend, cv2.COLOR_RGB2BGR), (w//4, h))
    
    comparison = np.hstack([
        image,
        colored_mask_bgr,
        overlay,
        legend_resized
    ])
    
    comparison_output = os.path.join(output_dir, f"{image_num}_{domain_found.lower()}_comparison.png")
    cv2.imwrite(comparison_output, comparison)
    print(f"Saved comparison to: {comparison_output}")
    
    return image, colored_mask, overlay, legend

def main():
    parser = argparse.ArgumentParser(description='Visualize LoveDA semantic segmentation masks')
    parser.add_argument('image_num', type=str, 
                       help='LoveDA image number (e.g., "1366", "0", "1000")')
    parser.add_argument('--loveda_dir', type=str, default='LoveDA',
                       help='Path to LoveDA dataset directory')
    parser.add_argument('--split', type=str, default='Train', choices=['Train', 'Val'],
                       help='Dataset split to use')
    parser.add_argument('--domain', type=str, choices=['Urban', 'Rural'],
                       help='Domain to search in (if not specified, searches both)')
    parser.add_argument('--output_dir', type=str, default='utils/debug',
                       help='Output directory for debug images')
    parser.add_argument('--no_analysis', action='store_true',
                       help='Skip class distribution analysis')
    
    args = parser.parse_args()
    
    visualize_loveda_image(args.image_num, args.loveda_dir, args.split, args.domain,
                          args.output_dir, not args.no_analysis)

if __name__ == "__main__":
    main() 