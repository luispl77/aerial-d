#!/usr/bin/env python3
"""
Step 6: Historic LoveDA Image Conversion

This script converts a percentage of LoveDA images (L-prefixed) to historic effects
while preserving iSAID images (P-prefixed) untouched.

The historic effects are applied in equal proportions:
- Basic B&W
- B&W + Grain  
- Sepia + Noise
"""

import os
import sys
import glob
import random
import numpy as np
import cv2
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm
import json

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

def apply_basic_bw_effect(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Basic black and white conversion."""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    return gray, "Basic_BW"

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
    
    return grainy, "BW_Grain"

def apply_sepia_with_noise_effect(image: np.ndarray) -> Tuple[np.ndarray, str]:
    """Apply sepia tone effect with noise for vintage look."""
    # Apply sepia effect - this should work on the original color image
    sepia_image = apply_sepia(image)
    
    # Add noise
    noisy_sepia = add_noise(sepia_image)
    
    return noisy_sepia, "Sepia_Noise"

class StyleBank:
    """Build and manage style banks from multiple images"""
    
    def __init__(self):
        self.domain_stats = {}
    
    def build_style_bank(self, image_paths: List[str], domain_name: str, max_samples: int = 500) -> Dict:
        """Build style statistics from multiple images"""
        # Sample random subset for efficiency
        if len(image_paths) > max_samples:
            sampled_paths = random.sample(image_paths, max_samples)
            print(f"Building style bank for {domain_name} from {max_samples} random samples (out of {len(image_paths)} total images)...")
        else:
            sampled_paths = image_paths
            print(f"Building style bank for {domain_name} from all {len(image_paths)} images...")
        
        all_means = []
        all_stds = []
        
        for img_path in tqdm(sampled_paths, desc=f"Processing {domain_name}", leave=False):
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # RGB statistics
            img_float = img.astype(np.float32)
            mean = np.mean(img_float.reshape(-1, 3), axis=0)
            std = np.std(img_float.reshape(-1, 3), axis=0)
            all_means.append(mean)
            all_stds.append(std)
        
        if not all_means:
            print(f"Warning: No valid images found for {domain_name}")
            return {}
        
        # Calculate domain statistics
        domain_stats = {
            'rgb_mean': np.mean(all_means, axis=0).tolist(),
            'rgb_std': np.mean(all_stds, axis=0).tolist(),
            'num_images': len(all_means)
        }
        
        self.domain_stats[domain_name] = domain_stats
        return domain_stats
    
    def apply_style_transfer(self, source: np.ndarray, target_domain: str) -> np.ndarray:
        """Apply style bank statistics to source image using RGB color moment transfer"""
        if target_domain not in self.domain_stats:
            raise ValueError(f"Domain {target_domain} not found in style bank")
        
        target_stats = self.domain_stats[target_domain]
        
        # RGB color space transfer
        source_float = source.astype(np.float32)
        source_mean = np.mean(source_float.reshape(-1, 3), axis=0)
        source_std = np.std(source_float.reshape(-1, 3), axis=0)
        
        target_mean = np.array(target_stats['rgb_mean'])
        target_std = np.array(target_stats['rgb_std'])
        
        result = source_float.copy()
        for channel in range(3):
            if source_std[channel] > 0:
                result[:, :, channel] = (result[:, :, channel] - source_mean[channel]) * \
                                      (target_std[channel] / source_std[channel]) + target_mean[channel]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def save_style_bank(self, filepath: str):
        """Save style bank to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.domain_stats, f, indent=2)
        print(f"Style bank saved to {filepath}")
    
    def load_style_bank(self, filepath: str):
        """Load style bank from JSON file"""
        with open(filepath, 'r') as f:
            self.domain_stats = json.load(f)
        print(f"Style bank loaded from {filepath}")

def get_domain_from_filename(filename: str) -> str:
    """Determine domain based on filename prefix"""
    basename = os.path.basename(filename).upper()
    if basename.startswith('P'):
        return 'iSAID'
    elif basename.startswith('D'):
        return 'DeepGlobe'
    elif basename.startswith('L'):
        return 'LoveDA'
    else:
        return 'Unknown'

def organize_images_by_domain(image_paths: List[str]) -> Dict[str, List[str]]:
    """Organize image paths by domain"""
    domain_images = {'iSAID': [], 'DeepGlobe': [], 'LoveDA': []}
    
    for img_path in image_paths:
        domain = get_domain_from_filename(img_path)
        if domain in domain_images:
            domain_images[domain].append(img_path)
    
    return domain_images

def select_images_for_style_transfer(domain_images: Dict[str, List[str]], seed: int = 42) -> Dict[str, Dict[str, List[str]]]:
    """
    Select 50% of images from each domain for style transfer.
    Within that 50%, 25% goes to each of the other two domains.
    
    Returns:
        Dict with structure: {domain: {'original': [...], 'to_domainA': [...], 'to_domainB': [...]}}
    """
    random.seed(seed)
    domains = ['iSAID', 'DeepGlobe', 'LoveDA']
    result = {}
    
    for source_domain in domains:
        if source_domain not in domain_images or not domain_images[source_domain]:
            result[source_domain] = {'original': [], 'transforms': {}}
            continue
            
        images = domain_images[source_domain].copy()
        random.shuffle(images)
        
        total_images = len(images)
        transform_count = total_images // 2  # 50% for transformation
        original_count = total_images - transform_count
        
        # Split transformation images between the two target domains
        target_domains = [d for d in domains if d != source_domain]
        transform_per_target = transform_count // 2
        
        # Handle remainder
        remainder = transform_count % 2
        target_counts = [transform_per_target + (1 if i < remainder else 0) for i in range(2)]
        
        # Organize images
        original_images = images[:original_count]
        
        transforms = {}
        start_idx = original_count
        for i, target_domain in enumerate(target_domains):
            end_idx = start_idx + target_counts[i]
            transforms[target_domain] = images[start_idx:end_idx]
            start_idx = end_idx
        
        result[source_domain] = {
            'original': original_images,
            'transforms': transforms
        }
    
    return result

def find_all_images_by_split(base_dir: str) -> Tuple[List[str], List[str]]:
    """
    Find all dataset images (LoveDA: L*, DeepGlobe: D*, iSAID: P*) separated by train/val split.
    
    Args:
        base_dir: Base directory containing train/val subdirectories
        
    Returns:
        Tuple of (train_images, val_images) lists
    """
    # Find all image types: LoveDA (L*), DeepGlobe (D*), iSAID (P*)
    train_patterns = [
        os.path.join(base_dir, "train/images/L*.png"),
        os.path.join(base_dir, "train/images/D*.png"),
        os.path.join(base_dir, "train/images/P*.png")
    ]
    val_patterns = [
        os.path.join(base_dir, "val/images/L*.png"),
        os.path.join(base_dir, "val/images/D*.png"),
        os.path.join(base_dir, "val/images/P*.png")
    ]
    
    train_images = []
    val_images = []
    
    for pattern in train_patterns:
        train_images.extend(glob.glob(pattern))
    
    for pattern in val_patterns:
        val_images.extend(glob.glob(pattern))
    
    return train_images, val_images

def select_images_for_conversion(image_paths: List[str], percentage: float, seed: int = 42) -> List[str]:
    """
    Select a percentage of images for historic conversion.
    
    Args:
        image_paths: List of image file paths
        percentage: Percentage of images to convert (0-100)
        seed: Random seed for reproducibility
        
    Returns:
        List of selected image paths
    """
    random.seed(seed)
    
    num_to_select = int(len(image_paths) * (percentage / 100.0))
    selected = random.sample(image_paths, num_to_select)
    
    return selected

def cleanup_existing_patch5_images(base_dir: str):
    """
    Remove any existing files with "_5.png" suffix to ensure clean runs.
    
    Args:
        base_dir: Base directory containing train/val subdirectories
    """
    patterns = [
        os.path.join(base_dir, "train/images/*_5.png"),
        os.path.join(base_dir, "val/images/*_5.png")
    ]
    
    removed_count = 0
    for pattern in patterns:
        existing_files = glob.glob(pattern)
        for file_path in tqdm(existing_files, desc="Removing existing _5 files", leave=False):
            try:
                os.remove(file_path)
                removed_count += 1
            except OSError:
                pass  # Silently skip files that can't be removed
    
    if removed_count > 0:
        print(f"Cleaned up {removed_count} existing _5 images")

def cleanup_existing_style_transfer_images(base_dir: str):
    """
    Remove any existing style transfer files with domain suffixes to ensure clean runs.

    Args:
        base_dir: Base directory containing train/val subdirectories
    """
    patterns = [
        os.path.join(base_dir, "train/images/*_toP.png"),
        os.path.join(base_dir, "train/images/*_toD.png"), 
        os.path.join(base_dir, "train/images/*_toL.png"),
        os.path.join(base_dir, "val/images/*_toP.png"),
        os.path.join(base_dir, "val/images/*_toD.png"),
        os.path.join(base_dir, "val/images/*_toL.png")
    ]
    
    removed_count = 0
    for pattern in patterns:
        existing_files = glob.glob(pattern)
        for file_path in tqdm(existing_files, desc="Removing existing style transfer files", leave=False):
            try:
                os.remove(file_path)
                removed_count += 1
            except OSError:
                pass  # Silently skip files that can't be removed
    
    if removed_count > 0:
        print(f"Cleaned up {removed_count} existing style transfer images")

def apply_historic_effects_equally(image_paths: List[str]) -> List[Tuple[str, str, np.ndarray]]:
    """
    Apply historic effects to images in equal proportions.
    
    Args:
        image_paths: List of image file paths to process
        
    Returns:
        List of tuples (image_path, effect_name, processed_image)
    """
    results = []
    num_images = len(image_paths)
    
    if num_images == 0:
        return results
    
    # Divide images into 3 equal groups for the 3 effects
    group_size = num_images // 3
    remainder = num_images % 3
    
    # Define group sizes (distribute remainder across first groups)
    group_sizes = [group_size + (1 if i < remainder else 0) for i in range(3)]
    
    effect_functions = [
        (apply_basic_bw_effect, "Basic_BW"),
        (apply_bw_grain_effect, "BW_Grain"), 
        (apply_sepia_with_noise_effect, "Sepia_Noise")
    ]
    
    current_idx = 0
    
    for group_idx, (effect_func, effect_name) in enumerate(effect_functions):
        group_start = current_idx
        group_end = current_idx + group_sizes[group_idx]
        
        for i, img_path in enumerate(tqdm(image_paths[group_start:group_end], 
                                        desc=f"Applying {effect_name}", 
                                        leave=False)):
            # Load image
            image = cv2.imread(img_path)
            if image is None:
                continue
            
            # Apply effect
            processed_image, _ = effect_func(image)
            
            # Save immediately instead of batching
            base_path, ext = os.path.splitext(img_path)
            new_path = f"{base_path}_5{ext}"
            cv2.imwrite(new_path, processed_image)
            
            results.append((img_path, effect_name, processed_image))
        
        current_idx = group_end
    
    return results

def apply_style_transfers(selection_result: Dict[str, Dict[str, List[str]]], style_bank: StyleBank) -> List[Tuple[str, str, np.ndarray]]:
    """
    Apply style transfers based on selection result.
    
    Returns:
        List of tuples (original_path, target_domain, processed_image)
    """
    results = []
    
    for source_domain, data in selection_result.items():
        if not data['transforms']:
            continue
            
        for target_domain, image_paths in data['transforms'].items():
            print(f"Transforming {len(image_paths)} {source_domain} images to {target_domain} style...")
            
            for img_path in tqdm(image_paths, desc=f"{source_domain} -> {target_domain}", leave=False):
                # Load image
                image = cv2.imread(img_path)
                if image is None:
                    continue
                
                # Apply style transfer
                try:
                    transferred_image = style_bank.apply_style_transfer(image, target_domain)
                    
                    # Save immediately instead of batching
                    suffix = get_domain_suffix(target_domain)
                    base_path, ext = os.path.splitext(img_path)
                    new_path = f"{base_path}{suffix}{ext}"
                    cv2.imwrite(new_path, transferred_image)
                    
                    results.append((img_path, target_domain, transferred_image))
                except Exception as e:
                    print(f"Warning: Failed to transfer {img_path} to {target_domain}: {e}")
                    continue
    
    return results

def get_domain_suffix(target_domain: str) -> str:
    """Get suffix for transformed images based on target domain"""
    domain_suffixes = {
        'iSAID': '_toP',
        'DeepGlobe': '_toD', 
        'LoveDA': '_toL'
    }
    return domain_suffixes.get(target_domain, '_unk')

def save_style_transferred_images(results: List[Tuple[str, str, np.ndarray]]):
    """
    Save style transferred images with appropriate suffixes.
    
    Examples:
        L1840_patch_0.png -> L1840_patch_0_toP.png (LoveDA to iSAID)
        P2788_patch_002034.png -> P2788_patch_002034_toD.png (iSAID to DeepGlobe)
        D1234_patch_1.png -> D1234_patch_1_toL.png (DeepGlobe to LoveDA)
    """
    for original_path, target_domain, processed_image in tqdm(results, desc="Saving style transferred images"):
        # Get appropriate suffix for target domain
        suffix = get_domain_suffix(target_domain)
        
        # Create new filename with domain suffix
        base_path, ext = os.path.splitext(original_path)
        new_path = f"{base_path}{suffix}{ext}"
        
        # Save processed image
        cv2.imwrite(new_path, processed_image)

def apply_single_effect(image_paths: List[str], effect_name: str) -> List[Tuple[str, str, np.ndarray]]:
    """
    Apply a single specific effect to all images.
    
    Args:
        image_paths: List of image file paths to process
        effect_name: Name of effect to apply ('basic_bw', 'bw_grain', 'sepia_noise')
        
    Returns:
        List of tuples (image_path, effect_name, processed_image)
    """
    results = []
    
    effect_mapping = {
        'basic_bw': (apply_basic_bw_effect, "Basic_BW"),
        'bw_grain': (apply_bw_grain_effect, "BW_Grain"),
        'sepia_noise': (apply_sepia_with_noise_effect, "Sepia_Noise"),
        'sepia_only': (lambda img: (apply_sepia(img), "Sepia_Only"), "Sepia_Only")
    }
    
    if effect_name not in effect_mapping:
        raise ValueError(f"Unknown effect: {effect_name}. Available: {list(effect_mapping.keys())}")
    
    effect_func, display_name = effect_mapping[effect_name]
    
    for img_path in tqdm(image_paths, desc=f"Applying {display_name}"):
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            continue
        
        # Apply effect
        processed_image, _ = effect_func(image)
        
        # Save immediately instead of batching
        base_path, ext = os.path.splitext(img_path)
        new_path = f"{base_path}_5{ext}"
        cv2.imwrite(new_path, processed_image)
        
        results.append((img_path, display_name, processed_image))
    
    return results

def save_processed_images(results: List[Tuple[str, str, np.ndarray]]):
    """
    Save processed images as new files with "_5" suffix, keeping originals intact.
    
    Examples:
        L1840_patch_0.png -> L1840_patch_0_5.png
        P2788_patch_002034.png -> P2788_patch_002034_5.png
        D1234_patch_1.png -> D1234_patch_1_5.png
    
    Args:
        results: List of (original_path, effect_name, processed_image) tuples
    """
    for original_path, effect_name, processed_image in tqdm(results, desc="Saving processed images"):
        # Simply append "_5" before the file extension
        # This works for all naming conventions: L*, D*, P* with any patch numbering
        base_path, ext = os.path.splitext(original_path)
        new_path = f"{base_path}_5{ext}"
        
        # Save processed image as new file with _5 suffix
        cv2.imwrite(new_path, processed_image)

def main():
    parser = argparse.ArgumentParser(description='Apply domain style transfer and historic effects to dataset images (both enabled by default)')
    parser.add_argument('--base-dir', default='dataset/patches', 
                       help='Base directory containing train/val subdirectories (default: dataset/patches)')
    parser.add_argument('--percentage', type=float, default=5.0,
                       help='Percentage of dataset images to convert (default: 5.0) - only for historic effects')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be processed without actually doing it')
    parser.add_argument('--effect', choices=['basic_bw', 'bw_grain', 'sepia_noise', 'sepia_only'], 
                       help='Apply only a specific effect to all selected images')
    parser.add_argument('--no-cleanup', action='store_true',
                       help='Skip cleanup of existing _5 images')
    
    # Style transfer arguments
    parser.add_argument('--no-style-transfer', action='store_true',
                       help='Disable domain style transfer (enabled by default)')
    parser.add_argument('--style-bank-path', type=str, default=None,
                       help='Path to save/load style bank JSON file')
    parser.add_argument('--no-build-style-bank', action='store_true',
                       help='Skip building style banks if they exist')
    parser.add_argument('--style-transfer-only', action='store_true',
                       help='Only do style transfer, skip historic effects')
    parser.add_argument('--historic-only', action='store_true',
                       help='Only do historic effects, skip style transfer')
    
    args = parser.parse_args()
    
    # Set defaults: enable both style transfer and historic effects by default
    args.style_transfer = not args.no_style_transfer and not args.historic_only
    args.do_historic = not args.style_transfer_only
    
    # Validate arguments
    if args.do_historic and not 0 < args.percentage <= 100:
        print("Error: Percentage must be between 0 and 100")
        sys.exit(1)
    
    if args.style_transfer and not args.style_bank_path:
        args.style_bank_path = os.path.join(args.base_dir, 'style_bank.json')
        print(f"Using default style bank path: {args.style_bank_path}")
    
    print(f"Configuration:")
    print(f"  Style transfer: {'Enabled' if args.style_transfer else 'Disabled'}")
    print(f"  Historic effects: {'Enabled' if args.do_historic else 'Disabled'}")
    if args.do_historic:
        print(f"  Historic percentage: {args.percentage}%")
    print(f"  Random seed: {args.seed}")
    print(f"  Base directory: {args.base_dir}")
    print()
    
    # Clean up existing images unless --no-cleanup is specified
    if not args.no_cleanup and not args.dry_run:
        if args.do_historic:
            print("Cleaning up existing _5 images...")
            cleanup_existing_patch5_images(args.base_dir)
        
        if args.style_transfer:
            print("Cleaning up existing style transfer images...")
            cleanup_existing_style_transfer_images(args.base_dir)
        print()
    
    # Find all dataset images by split
    print(f"Searching for all dataset images in {args.base_dir}...")
    print(f"Looking for patterns:")
    print(f"  Train: {args.base_dir}/train/images/[L|D|P]*.png")
    print(f"  Val: {args.base_dir}/val/images/[L|D|P]*.png")
    
    train_images, val_images = find_all_images_by_split(args.base_dir)
    
    if not train_images and not val_images:
        print("Error: No dataset images found!")
        print("Please check if the base directory path is correct and contains train/val/images/ subdirectories")
        sys.exit(1)
    
    print(f"Found {len(train_images)} dataset train images")
    print(f"Found {len(val_images)} dataset val images")
    
    # Style transfer processing
    if args.style_transfer:
        # Combine all images
        all_images = train_images + val_images
        
        # Organize by domain
        domain_images = organize_images_by_domain(all_images)
        print(f"\nImages by domain:")
        for domain, images in domain_images.items():
            print(f"  {domain}: {len(images)} images")
        
        # Initialize style bank
        style_bank = StyleBank()
        
        # Build or load style bank
        if not args.no_build_style_bank or not os.path.exists(args.style_bank_path):
            print(f"\nBuilding style banks for all domains...")
            # Set seed for consistent sampling across runs
            random.seed(args.seed)
            for domain, images in domain_images.items():
                if images:
                    style_bank.build_style_bank(images, domain, max_samples=500)
            style_bank.save_style_bank(args.style_bank_path)
        else:
            print(f"\nLoading existing style bank from {args.style_bank_path}")
            style_bank.load_style_bank(args.style_bank_path)
        
        # Select images for style transfer (50% per domain)
        print(f"\nSelecting images for style transfer (50% per domain)...")
        selection_result = select_images_for_style_transfer(domain_images, args.seed)
        
        # Print selection summary
        for source_domain, data in selection_result.items():
            original_count = len(data['original'])
            total_transforms = sum(len(imgs) for imgs in data['transforms'].values())
            print(f"  {source_domain}: {original_count} original, {total_transforms} transforms")
            for target_domain, imgs in data['transforms'].items():
                print(f"    -> {target_domain}: {len(imgs)} images")
        
        if args.dry_run:
            print("\nDry run - would apply style transfers as shown above")
            if args.do_historic:
                print("Would also apply historic effects to remaining percentage of images")
            return
        
        # Apply style transfers
        print(f"\nApplying style transfers...")
        style_results = apply_style_transfers(selection_result, style_bank)
        
        # Style transferred images already saved during processing
        
        print(f"\nStyle transfer completed! Generated {len(style_results)} transformed images")
        
        # Count results by transformation type
        transform_counts = {}
        for _, target_domain, _ in style_results:
            transform_counts[target_domain] = transform_counts.get(target_domain, 0) + 1
        
        for target_domain, count in transform_counts.items():
            print(f"  -> {target_domain}: {count} images")
    
    # Historic effects processing
    print(f"DEBUG: args.do_historic = {args.do_historic}")
    if args.do_historic:
        print(f"\n=== HISTORIC EFFECTS PROCESSING ===")
        print(f"Train images available: {len(train_images)}")
        print(f"Val images available: {len(val_images)}")
        
        # Select images for conversion from each split separately
        selected_train = select_images_for_conversion(train_images, args.percentage, args.seed)
        selected_val = select_images_for_conversion(val_images, args.percentage, args.seed + 1)  # Different seed for val
        
        print(f"Selected train images: {len(selected_train)}")
        print(f"Selected val images: {len(selected_val)}")
        
        # Combine selected images
        selected_images = selected_train + selected_val
        num_selected = len(selected_images)
        
        print(f"Selected {len(selected_train)} train images ({args.percentage}%) for historic conversion")
        print(f"Selected {len(selected_val)} val images ({args.percentage}%) for historic conversion")
        print(f"Total selected: {num_selected} images")
        
        if args.effect:
            print(f"Applying single effect: {args.effect}")
        else:
            # Calculate distribution across effects
            group_size = num_selected // 3
            remainder = num_selected % 3
            group_sizes = [group_size + (1 if i < remainder else 0) for i in range(3)]
            
            print(f"Effect distribution:")
            print(f"  - Basic B&W: {group_sizes[0]} images")
            print(f"  - B&W + Grain: {group_sizes[1]} images") 
            print(f"  - Sepia + Noise: {group_sizes[2]} images")
        
        if args.dry_run:
            print("\nDry run - would process these files:")
            if args.effect:
                for img_path in selected_images:
                    print(f"  {os.path.basename(img_path)} -> {args.effect}")
            else:
                for i, img_path in enumerate(selected_images):
                    if num_selected >= 3:
                        group_size = num_selected // 3
                        remainder = num_selected % 3
                        group_sizes = [group_size + (1 if j < remainder else 0) for j in range(3)]
                        
                        # Determine which group this image belongs to
                        cumulative = 0
                        effect_idx = 0
                        for j, gs in enumerate(group_sizes):
                            if i < cumulative + gs:
                                effect_idx = j
                                break
                            cumulative += gs
                    else:
                        effect_idx = i
                    
                    effect_names = ["Basic B&W", "B&W + Grain", "Sepia + Noise"]
                    print(f"  {os.path.basename(img_path)} -> {effect_names[effect_idx]}")
            return
        
        # Apply historic effects
        print(f"\nApplying historic effects...")
        if args.effect:
            # Apply single effect to all selected images
            all_selected = selected_train + selected_val
            results = apply_single_effect(all_selected, args.effect)
        else:
            # Apply effects separately to train and val to ensure proper distribution
            results = []
            
            if selected_train:
                print(f"Processing {len(selected_train)} train images...")
                train_results = apply_historic_effects_equally(selected_train)
                results.extend(train_results)
            
            if selected_val:
                print(f"Processing {len(selected_val)} val images...")
                val_results = apply_historic_effects_equally(selected_val)
                results.extend(val_results)
        
        # Historic images already saved during processing
        
        print(f"\nCompleted! Processed {len(results)} images:")
        effect_counts = {}
        for _, effect_name, _ in results:
            effect_counts[effect_name] = effect_counts.get(effect_name, 0) + 1
        
        for effect_name, count in effect_counts.items():
            print(f"  - {effect_name}: {count} images")

if __name__ == "__main__":
    main() 