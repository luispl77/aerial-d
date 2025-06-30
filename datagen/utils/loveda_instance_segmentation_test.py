#!/usr/bin/env python3
"""
LoveDA Instance Segmentation Test Script

This script tests strategies for converting LoveDA semantic segmentation masks
into instance segmentation masks for buildings, roads, and water classes.

Strategy 1: Connected Components Analysis
- Separate buildings/water into individual polygons using connected components
- Apply morphological operations to clean up the masks
- Filter by size and shape criteria

All results are saved to utils/debug/ directory.

Usage:
    # Process a single specific image
    python utils/loveda_instance_segmentation_test.py --image_name 1658
    python utils/loveda_instance_segmentation_test.py --image_name 1658 --domain Rural --split Val
    
    # Process multiple random images with buildings/roads/water
    python utils/loveda_instance_segmentation_test.py --random_images 5
    python utils/loveda_instance_segmentation_test.py --random_images 10 --seed 123
    
    # Adjust area thresholds for filtering
    python utils/loveda_instance_segmentation_test.py --random_images 3 --min_building_area 100 --max_building_area 20000 --min_road_area 300
"""

import os
import sys
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
from skimage import measure, morphology
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects
from skimage.segmentation import watershed
from scipy import ndimage
import xml.etree.ElementTree as ET
import pycocotools.mask as mask_util
import random
import glob

# LoveDA class mapping for our target classes
TARGET_CLASSES = {
    2: 'building',
    3: 'road',
    4: 'water'
}

def find_loveda_image(image_name, domain='Urban', split='Train', loveda_dir='LoveDA'):
    """
    Find a LoveDA image across different splits and domains
    
    Args:
        image_name: Name of the image (e.g., '1658')
        domain: 'Urban' or 'Rural' (default: 'Urban')
        split: 'Train' or 'Val' (default: 'Train')
        loveda_dir: Path to LoveDA dataset
    
    Returns:
        tuple: (image_path, mask_path) if found, (None, None) otherwise
    """
    # Try specified domain and split first
    base_path = os.path.join(loveda_dir, split, domain)
    image_path = os.path.join(base_path, 'images_png', f'{image_name}.png')
    mask_path = os.path.join(base_path, 'masks_png', f'{image_name}.png')
    
    if os.path.exists(image_path) and os.path.exists(mask_path):
        return image_path, mask_path
    
    # Search in all combinations if not found
    for s in ['Train', 'Val']:
        for d in ['Urban', 'Rural']:
            if s == split and d == domain:
                continue  # Already checked
            
            base_path = os.path.join(loveda_dir, s, d)
            image_path = os.path.join(base_path, 'images_png', f'{image_name}.png')
            mask_path = os.path.join(base_path, 'masks_png', f'{image_name}.png')
            
            if os.path.exists(image_path) and os.path.exists(mask_path):
                print(f"Found {image_name} in {s}/{d}")
                return image_path, mask_path
    
    return None, None

def get_all_loveda_images(loveda_dir='LoveDA', splits=['Train', 'Val'], domains=['Urban', 'Rural']):
    """
    Get all LoveDA image names from specified splits and domains
    
    Args:
        loveda_dir: Path to LoveDA dataset
        splits: List of splits to include ['Train', 'Val']
        domains: List of domains to include ['Urban', 'Rural']
    
    Returns:
        list: List of dictionaries with image info
    """
    all_images = []
    
    for split in splits:
        for domain in domains:
            images_dir = os.path.join(loveda_dir, split, domain, 'images_png')
            masks_dir = os.path.join(loveda_dir, split, domain, 'masks_png')
            
            if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
                continue
            
            # Get all image files
            image_files = glob.glob(os.path.join(images_dir, '*.png'))
            
            for image_file in image_files:
                image_name = os.path.splitext(os.path.basename(image_file))[0]
                mask_file = os.path.join(masks_dir, f'{image_name}.png')
                
                if os.path.exists(mask_file):
                    all_images.append({
                        'name': image_name,
                        'split': split,
                        'domain': domain,
                        'image_path': image_file,
                        'mask_path': mask_file
                    })
    
    return all_images

def select_random_images_with_targets(all_images, num_images, target_classes=[2, 3, 4], max_attempts=None):
    """
    Select random images that contain the target classes (buildings and/or water)
    by checking images randomly until we find enough valid ones.
    
    Args:
        all_images: List of image dictionaries from get_all_loveda_images
        num_images: Number of images to select
        target_classes: List of class IDs to look for [2=building, 3=road, 4=water]
        max_attempts: Maximum number of images to check (default: 3 * num_images)
    
    Returns:
        list: Selected image dictionaries
    """
    if max_attempts is None:
        max_attempts = max(num_images * 3, 50)  # Check at most 3x the requested number or 50 images
    
    valid_images = []
    checked_indices = set()
    attempts = 0
    
    print(f"Randomly selecting {num_images} images with buildings/roads/water...")
    
    while len(valid_images) < num_images and attempts < max_attempts:
        # Pick a random image we haven't checked yet
        if len(checked_indices) >= len(all_images):
            print(f"Warning: Checked all {len(all_images)} images, found only {len(valid_images)} valid ones")
            break
            
        random_idx = random.randint(0, len(all_images) - 1)
        
        # Skip if we already checked this image
        if random_idx in checked_indices:
            continue
            
        checked_indices.add(random_idx)
        attempts += 1
        
        img_info = all_images[random_idx]
        
        try:
            # Quick check if image contains target classes
            mask = cv2.imread(img_info['mask_path'], cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue
                
            building_pixels = np.sum(mask == 2)
            road_pixels = np.sum(mask == 3)
            water_pixels = np.sum(mask == 4)
            
            # Only include if there's a reasonable amount of target pixels
            if building_pixels > 100 or road_pixels > 100 or water_pixels > 100:
                img_info['building_pixels'] = building_pixels
                img_info['road_pixels'] = road_pixels
                img_info['water_pixels'] = water_pixels
                valid_images.append(img_info)
                print(f"  Found valid image {len(valid_images)}/{num_images}: {img_info['name']} "
                      f"({img_info['split']}/{img_info['domain']}) - "
                      f"{building_pixels} building, {road_pixels} road, {water_pixels} water pixels")
                    
        except Exception as e:
            print(f"  Error checking {img_info['name']}: {e}")
            continue
    
    print(f"Selected {len(valid_images)} valid images after checking {attempts} images")
    
    # Sort by split/domain/name for consistent processing order
    valid_images.sort(key=lambda x: (x['split'], x['domain'], x['name']))
    
    return valid_images

def load_image_and_mask(image_path, mask_path):
    """Load image and semantic mask"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        raise ValueError(f"Could not load image or mask from {image_path}, {mask_path}")
    
    return image, mask

def extract_building_instances(semantic_mask, min_area=50, max_area=50000):
    """
    Extract individual building instances from semantic mask using connected components
    
    Args:
        semantic_mask: 2D array with class pixel values
        min_area: Minimum area threshold for buildings
        max_area: Maximum area threshold for buildings
    
    Returns:
        list: List of instance masks and bounding boxes
    """
    # Create binary mask for buildings (class 2)
    building_mask = (semantic_mask == 2).astype(np.uint8)
    
    if np.sum(building_mask) == 0:
        return []
    
    # Apply morphological operations to clean up the mask
    # Remove small noise
    kernel = disk(1)
    building_mask = binary_opening(building_mask, kernel)
    
    # Fill small holes
    kernel = disk(2) 
    building_mask = binary_closing(building_mask, kernel)
    
    # Remove very small objects
    building_mask = remove_small_objects(building_mask.astype(bool), min_size=min_area)
    building_mask = building_mask.astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(building_mask)
    
    instances = []
    for label_id in range(1, num_labels):  # Skip background (0)
        # Create instance mask
        instance_mask = (labels == label_id).astype(np.uint8)
        area = np.sum(instance_mask)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Get bounding box
        rows, cols = np.where(instance_mask == 1)
        if len(rows) == 0:
            continue
            
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        bbox = [min_col, min_row, max_col + 1, max_row + 1]  # [xmin, ymin, xmax, ymax]
        
        # Calculate some shape features for filtering
        # Aspect ratio
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Solidity (area / convex hull area)
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            convex_hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(convex_hull)
            solidity = area / hull_area if hull_area > 0 else 0
        else:
            solidity = 0
        
        # Filter by shape (buildings should be reasonably compact)
        if solidity < 0.3 or aspect_ratio > 10 or aspect_ratio < 0.1:
            continue
        
        # Convert to RLE for storage efficiency
        rle = mask_util.encode(np.asfortranarray(instance_mask))
        
        instances.append({
            'mask': instance_mask,
            'bbox': bbox,
            'area': int(area),
            'rle': rle,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'label_id': label_id
        })
    
    return instances

def extract_water_instances(semantic_mask, min_area=100, max_area=100000):
    """
    Extract individual water instances from semantic mask
    
    Water bodies can be more irregular than buildings, so we use different parameters
    """
    # Create binary mask for water (class 4)
    water_mask = (semantic_mask == 4).astype(np.uint8)
    
    if np.sum(water_mask) == 0:
        return []
    
    # Apply lighter morphological operations for water (preserve shape better)
    kernel = disk(1)
    water_mask = binary_opening(water_mask, kernel)
    
    # Fill holes in water bodies
    kernel = disk(3)
    water_mask = binary_closing(water_mask, kernel)
    
    # Remove very small objects
    water_mask = remove_small_objects(water_mask.astype(bool), min_size=min_area)
    water_mask = water_mask.astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(water_mask)
    
    instances = []
    for label_id in range(1, num_labels):  # Skip background (0)
        # Create instance mask
        instance_mask = (labels == label_id).astype(np.uint8)
        area = np.sum(instance_mask)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Get bounding box
        rows, cols = np.where(instance_mask == 1)
        if len(rows) == 0:
            continue
            
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        bbox = [min_col, min_row, max_col + 1, max_row + 1]  # [xmin, ymin, xmax, ymax]
        
        # Convert to RLE
        rle = mask_util.encode(np.asfortranarray(instance_mask))
        
        instances.append({
            'mask': instance_mask,
            'bbox': bbox,
            'area': int(area),
            'rle': rle,
            'label_id': label_id
        })
    
    return instances

def extract_road_instances(semantic_mask, min_area=200, max_area=200000):
    """
    Extract individual road instances from semantic mask
    
    Roads can be long and thin, so we use different parameters and allow higher aspect ratios
    """
    # Create binary mask for roads (class 3)
    road_mask = (semantic_mask == 3).astype(np.uint8)
    
    if np.sum(road_mask) == 0:
        return []
    
    # Apply lighter morphological operations for roads (preserve connectivity)
    kernel = disk(1)
    road_mask = binary_opening(road_mask, kernel)
    
    # Fill small gaps in roads
    kernel = disk(2)
    road_mask = binary_closing(road_mask, kernel)
    
    # Remove very small objects
    road_mask = remove_small_objects(road_mask.astype(bool), min_size=min_area)
    road_mask = road_mask.astype(np.uint8)
    
    # Find connected components
    num_labels, labels = cv2.connectedComponents(road_mask)
    
    instances = []
    for label_id in range(1, num_labels):  # Skip background (0)
        # Create instance mask
        instance_mask = (labels == label_id).astype(np.uint8)
        area = np.sum(instance_mask)
        
        # Filter by area
        if area < min_area or area > max_area:
            continue
        
        # Get bounding box
        rows, cols = np.where(instance_mask == 1)
        if len(rows) == 0:
            continue
            
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        bbox = [min_col, min_row, max_col + 1, max_row + 1]  # [xmin, ymin, xmax, ymax]
        
        # Calculate aspect ratio (roads can be very elongated)
        width = max_col - min_col + 1
        height = max_row - min_row + 1
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Solidity check (roads should be reasonably solid)
        contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            convex_hull = cv2.convexHull(contours[0])
            hull_area = cv2.contourArea(convex_hull)
            solidity = area / hull_area if hull_area > 0 else 0
        else:
            solidity = 0
        
        # More lenient filtering for roads (they can be very long and thin)
        if solidity < 0.1 or aspect_ratio > 50 or aspect_ratio < 0.02:
            continue
        
        # Convert to RLE
        rle = mask_util.encode(np.asfortranarray(instance_mask))
        
        instances.append({
            'mask': instance_mask,
            'bbox': bbox,
            'area': int(area),
            'rle': rle,
            'aspect_ratio': aspect_ratio,
            'solidity': solidity,
            'label_id': label_id
        })
    
    return instances

def visualize_results(image, semantic_mask, building_instances, road_instances, water_instances, save_path=None):
    """
    Visualize the original image, semantic mask, and extracted instances
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 18))
    
    # Original image
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')
    
    # Semantic mask (show buildings, roads, and water)
    semantic_viz = np.zeros_like(semantic_mask)
    semantic_viz[semantic_mask == 2] = 1  # Buildings
    semantic_viz[semantic_mask == 3] = 2  # Roads
    semantic_viz[semantic_mask == 4] = 3  # Water
    
    im1 = axes[0, 1].imshow(semantic_viz, cmap='viridis', vmin=0, vmax=3)
    axes[0, 1].set_title('Semantic Mask\n(Buildings=1, Roads=2, Water=3)')
    axes[0, 1].axis('off')
    
    # Building instances
    building_viz = np.zeros_like(semantic_mask)
    for i, instance in enumerate(building_instances):
        building_viz[instance['mask'] == 1] = i + 1
    
    im2 = axes[0, 2].imshow(building_viz, cmap='tab20')
    axes[0, 2].set_title(f'Building Instances ({len(building_instances)} found)')
    axes[0, 2].axis('off')
    
    # Road instances
    road_viz = np.zeros_like(semantic_mask)
    for i, instance in enumerate(road_instances):
        road_viz[instance['mask'] == 1] = i + 1
        
    im3 = axes[1, 0].imshow(road_viz, cmap='tab20')
    axes[1, 0].set_title(f'Road Instances ({len(road_instances)} found)')
    axes[1, 0].axis('off')
    
    # Water instances  
    water_viz = np.zeros_like(semantic_mask)
    for i, instance in enumerate(water_instances):
        water_viz[instance['mask'] == 1] = i + 1
        
    im4 = axes[1, 1].imshow(water_viz, cmap='tab20')
    axes[1, 1].set_title(f'Water Instances ({len(water_instances)} found)')
    axes[1, 1].axis('off')
    
    # Combined instances with bounding boxes
    axes[1, 2].imshow(image)
    
    # Draw building bounding boxes
    for instance in building_instances:
        bbox = instance['bbox']
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                        linewidth=2, edgecolor='red', facecolor='none')
        axes[1, 2].add_patch(rect)
    
    # Draw road bounding boxes
    for instance in road_instances:
        bbox = instance['bbox']
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                        linewidth=2, edgecolor='green', facecolor='none')
        axes[1, 2].add_patch(rect)
    
    # Draw water bounding boxes
    for instance in water_instances:
        bbox = instance['bbox']
        rect = Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], 
                        linewidth=2, edgecolor='blue', facecolor='none')
        axes[1, 2].add_patch(rect)
    
    axes[1, 2].set_title('Instances with Bounding Boxes\n(Red=Buildings, Green=Roads, Blue=Water)')
    axes[1, 2].axis('off')
    
    # Statistics
    stats_text = f"""Instance Statistics:

Buildings: {len(building_instances)}
- Total area: {sum(inst['area'] for inst in building_instances)} pixels
- Avg area: {np.mean([inst['area'] for inst in building_instances]):.0f} pixels
- Size range: {min([inst['area'] for inst in building_instances]) if building_instances else 0} - {max([inst['area'] for inst in building_instances]) if building_instances else 0}

Roads: {len(road_instances)}
- Total area: {sum(inst['area'] for inst in road_instances)} pixels
- Avg area: {np.mean([inst['area'] for inst in road_instances]):.0f} pixels
- Size range: {min([inst['area'] for inst in road_instances]) if road_instances else 0} - {max([inst['area'] for inst in road_instances]) if road_instances else 0}

Water: {len(water_instances)}
- Total area: {sum(inst['area'] for inst in water_instances)} pixels  
- Avg area: {np.mean([inst['area'] for inst in water_instances]):.0f} pixels
- Size range: {min([inst['area'] for inst in water_instances]) if water_instances else 0} - {max([inst['area'] for inst in water_instances]) if water_instances else 0}

Semantic totals:
- Building pixels: {np.sum(semantic_mask == 2)}
- Road pixels: {np.sum(semantic_mask == 3)}
- Water pixels: {np.sum(semantic_mask == 4)}
"""
    
    axes[2, 0].text(0.1, 0.9, stats_text, transform=axes[2, 0].transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
    axes[2, 0].set_title('Instance Statistics')
    axes[2, 0].axis('off')
    
    # Combined visualization - all instances colored by type
    combined_viz = np.zeros_like(semantic_mask)
    
    # Add building instances
    for instance in building_instances:
        combined_viz[instance['mask'] == 1] = 1
    
    # Add road instances  
    for instance in road_instances:
        combined_viz[instance['mask'] == 1] = 2
        
    # Add water instances
    for instance in water_instances:
        combined_viz[instance['mask'] == 1] = 3
    
    im5 = axes[2, 1].imshow(combined_viz, cmap='viridis', vmin=0, vmax=3)
    axes[2, 1].set_title('All Extracted Instances\n(Buildings=1, Roads=2, Water=3)')
    axes[2, 1].axis('off')
    
    # Hide the empty subplot
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to: {save_path}")
    
    plt.show()

def save_instances_xml(image_name, width, height, building_instances, road_instances, water_instances, output_path):
    """
    Save instances in XML format compatible with the pipeline
    """
    root = ET.Element('annotation')
    
    # Add filename
    ET.SubElement(root, 'filename').text = f"{image_name}.png"
    
    # Add size
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    
    # Add instances
    instance_id = 1
    
    # Add building instances
    for instance in building_instances:
        object_elem = ET.SubElement(root, 'object')
        ET.SubElement(object_elem, 'name').text = 'building'
        ET.SubElement(object_elem, 'id').text = str(instance_id)
        
        # Bounding box
        bbox = instance['bbox']
        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(bbox[0])
        ET.SubElement(bndbox, 'ymin').text = str(bbox[1])
        ET.SubElement(bndbox, 'xmax').text = str(bbox[2])
        ET.SubElement(bndbox, 'ymax').text = str(bbox[3])
        
        # Add segmentation in RLE format
        rle_str = f"{instance['rle']['counts'].decode('utf-8')},{instance['rle']['size'][0]},{instance['rle']['size'][1]}"
        ET.SubElement(object_elem, 'segmentation').text = rle_str
        
        # Add area
        ET.SubElement(object_elem, 'area').text = str(instance['area'])
        
        # Add additional metrics
        ET.SubElement(object_elem, 'aspect_ratio').text = str(round(instance['aspect_ratio'], 3))
        ET.SubElement(object_elem, 'solidity').text = str(round(instance['solidity'], 3))
        
        instance_id += 1
    
    # Add road instances
    for instance in road_instances:
        object_elem = ET.SubElement(root, 'object')
        ET.SubElement(object_elem, 'name').text = 'road'
        ET.SubElement(object_elem, 'id').text = str(instance_id)
        
        # Bounding box
        bbox = instance['bbox']
        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(bbox[0])
        ET.SubElement(bndbox, 'ymin').text = str(bbox[1])
        ET.SubElement(bndbox, 'xmax').text = str(bbox[2])
        ET.SubElement(bndbox, 'ymax').text = str(bbox[3])
        
        # Add segmentation in RLE format
        rle_str = f"{instance['rle']['counts'].decode('utf-8')},{instance['rle']['size'][0]},{instance['rle']['size'][1]}"
        ET.SubElement(object_elem, 'segmentation').text = rle_str
        
        # Add area
        ET.SubElement(object_elem, 'area').text = str(instance['area'])
        
        # Add additional metrics
        ET.SubElement(object_elem, 'aspect_ratio').text = str(round(instance['aspect_ratio'], 3))
        ET.SubElement(object_elem, 'solidity').text = str(round(instance['solidity'], 3))
        
        instance_id += 1
    
    # Add water instances
    for instance in water_instances:
        object_elem = ET.SubElement(root, 'object')
        ET.SubElement(object_elem, 'name').text = 'water'
        ET.SubElement(object_elem, 'id').text = str(instance_id)
        
        # Bounding box
        bbox = instance['bbox']
        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(bbox[0])
        ET.SubElement(bndbox, 'ymin').text = str(bbox[1])
        ET.SubElement(bndbox, 'xmax').text = str(bbox[2])
        ET.SubElement(bndbox, 'ymax').text = str(bbox[3])
        
        # Add segmentation in RLE format
        rle_str = f"{instance['rle']['counts'].decode('utf-8')},{instance['rle']['size'][0]},{instance['rle']['size'][1]}"
        ET.SubElement(object_elem, 'segmentation').text = rle_str
        
        # Add area
        ET.SubElement(object_elem, 'area').text = str(instance['area'])
        
        instance_id += 1
    
    # Pretty print XML
    def indent(elem, level=0):
        i = "\n" + level * "  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent(elem, level + 1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    indent(root)
    
    # Write XML
    tree = ET.ElementTree(root)
    tree.write(output_path, encoding='utf-8', xml_declaration=True)
    print(f"XML annotation saved to: {output_path}")

def process_single_image(img_info, output_dir, min_building_area, max_building_area, 
                        min_road_area, max_road_area, min_water_area, max_water_area, show_plot=True):
    """
    Process a single image and extract instances
    
            Args:
        img_info: Dictionary with image information
        output_dir: Output directory for results
        min_building_area, max_building_area: Area thresholds for buildings
        min_road_area, max_road_area: Area thresholds for roads
        min_water_area, max_water_area: Area thresholds for water
        show_plot: Whether to display the plot (set to False for batch processing)
    
    Returns:
        dict: Results summary
    """
    image_name = img_info['name']
    print(f"\nProcessing {image_name} ({img_info['split']}/{img_info['domain']})")
    
    try:
        # Load image and mask
        image, semantic_mask = load_image_and_mask(img_info['image_path'], img_info['mask_path'])
        
        # Check what classes are present
        unique_classes = np.unique(semantic_mask)
        building_pixels = np.sum(semantic_mask == 2)
        road_pixels = np.sum(semantic_mask == 3)
        water_pixels = np.sum(semantic_mask == 4)
        
        print(f"  Building pixels: {building_pixels}, Road pixels: {road_pixels}, Water pixels: {water_pixels}")
        
        if building_pixels == 0 and road_pixels == 0 and water_pixels == 0:
            print("  Warning: No buildings, roads, or water found")
            return None
        
        # Extract instances
        building_instances = extract_building_instances(
            semantic_mask, 
            min_area=min_building_area, 
            max_area=max_building_area
        )
        
        road_instances = extract_road_instances(
            semantic_mask,
            min_area=min_road_area,
            max_area=max_road_area
        )
        
        water_instances = extract_water_instances(
            semantic_mask, 
            min_area=min_water_area, 
            max_area=max_water_area
        )
        
        print(f"  Found {len(building_instances)} building, {len(road_instances)} road, {len(water_instances)} water instances")
        
        # Create output filenames
        viz_path = os.path.join(output_dir, f"{image_name}_instance_segmentation.png")
        xml_path = os.path.join(output_dir, f"{image_name}_instances.xml")
        
        # Visualize results
        plt.ioff() if not show_plot else plt.ion()  # Turn off interactive mode for batch processing
        visualize_results(image, semantic_mask, building_instances, road_instances, water_instances, viz_path)
        if not show_plot:
            plt.close()  # Close the figure to save memory
        
        # Save XML annotation
        save_instances_xml(image_name, image.shape[1], image.shape[0], 
                          building_instances, road_instances, water_instances, xml_path)
        
        # Return summary
        return {
            'name': image_name,
            'split': img_info['split'],
            'domain': img_info['domain'],
            'building_pixels': building_pixels,
            'road_pixels': road_pixels,
            'water_pixels': water_pixels,
            'building_instances': len(building_instances),
            'road_instances': len(road_instances),
            'water_instances': len(water_instances),
            'total_instances': len(building_instances) + len(road_instances) + len(water_instances)
        }
        
    except Exception as e:
        print(f"  Error processing {image_name}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description='Test LoveDA instance segmentation strategy')
    parser.add_argument('--image_name', type=str, 
                       help='Name of the image to process (e.g., 1658). Mutually exclusive with --random_images')
    parser.add_argument('--random_images', type=int,
                       help='Number of random images to process. Mutually exclusive with --image_name')
    parser.add_argument('--domain', type=str, default='Urban', choices=['Urban', 'Rural'],
                       help='Domain to search in (default: Urban)')
    parser.add_argument('--split', type=str, default='Train', choices=['Train', 'Val'],
                       help='Split to search in (default: Train)')
    parser.add_argument('--loveda_dir', type=str, default='LoveDA',
                       help='Path to LoveDA dataset directory')
    parser.add_argument('--min_building_area', type=int, default=50,
                       help='Minimum area for building instances')
    parser.add_argument('--max_building_area', type=int, default=50000,
                       help='Maximum area for building instances')
    parser.add_argument('--min_road_area', type=int, default=200,
                       help='Minimum area for road instances')
    parser.add_argument('--max_road_area', type=int, default=200000,
                       help='Maximum area for road instances')
    parser.add_argument('--min_water_area', type=int, default=100,
                       help='Minimum area for water instances')
    parser.add_argument('--max_water_area', type=int, default=100000,
                       help='Maximum area for water instances')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducible results')
    
    args = parser.parse_args()
    
    # Validation
    if args.image_name and args.random_images:
        print("Error: Cannot specify both --image_name and --random_images")
        return
    
    if not args.image_name and not args.random_images:
        print("Error: Must specify either --image_name or --random_images")
        return
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory in utils/debug
    output_dir = os.path.join('utils', 'debug')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Single image mode
    if args.image_name:
        print(f"Single image mode: Processing {args.image_name}")
        
        # Find the image
        image_path, mask_path = find_loveda_image(args.image_name, args.domain, args.split, args.loveda_dir)
        
        if image_path is None:
            print(f"Error: Could not find image {args.image_name} in LoveDA dataset")
            return
        
        img_info = {
            'name': args.image_name,
            'split': args.split,
            'domain': args.domain,
            'image_path': image_path,
            'mask_path': mask_path
        }
        
        result = process_single_image(
            img_info, output_dir, 
            args.min_building_area, args.max_building_area,
            args.min_road_area, args.max_road_area,
            args.min_water_area, args.max_water_area,
            show_plot=True  # Show plot for single image
        )
        
        if result:
            print(f"\nResults for {args.image_name}:")
            print(f"  Buildings: {result['building_instances']} instances ({result['building_pixels']} pixels)")
            print(f"  Roads: {result['road_instances']} instances ({result['road_pixels']} pixels)")
            print(f"  Water: {result['water_instances']} instances ({result['water_pixels']} pixels)")
        
    # Random images mode
    else:
        print(f"Random images mode: Processing {args.random_images} images")
        
        # Get all available images
        print("Scanning LoveDA dataset...")
        all_images = get_all_loveda_images(args.loveda_dir)
        print(f"Found {len(all_images)} total images in dataset")
        
        # Select random images with target classes
        selected_images = select_random_images_with_targets(all_images, args.random_images)
        
        if not selected_images:
            print("No valid images found!")
            return
        
        print(f"\nProcessing {len(selected_images)} selected images...")
        
        # Process all selected images
        results = []
        for i, img_info in enumerate(selected_images):
            print(f"\n[{i+1}/{len(selected_images)}]", end=" ")
            result = process_single_image(
                img_info, output_dir,
                args.min_building_area, args.max_building_area,
                args.min_road_area, args.max_road_area,
                args.min_water_area, args.max_water_area,
                show_plot=False  # Don't show plots for batch processing
            )
            
            if result:
                results.append(result)
        
        # Print summary
        if results:
            print(f"\n{'='*60}")
            print(f"BATCH PROCESSING SUMMARY")
            print(f"{'='*60}")
            print(f"Successfully processed: {len(results)} images")
            
            total_building_instances = sum(r['building_instances'] for r in results)
            total_road_instances = sum(r['road_instances'] for r in results)
            total_water_instances = sum(r['water_instances'] for r in results)
            total_building_pixels = sum(r['building_pixels'] for r in results)
            total_road_pixels = sum(r['road_pixels'] for r in results)
            total_water_pixels = sum(r['water_pixels'] for r in results)
            
            print(f"Total building instances: {total_building_instances}")
            print(f"Total road instances: {total_road_instances}")
            print(f"Total water instances: {total_water_instances}")
            print(f"Total building pixels: {total_building_pixels}")
            print(f"Total road pixels: {total_road_pixels}")
            print(f"Total water pixels: {total_water_pixels}")
            
            if total_building_instances > 0:
                avg_building_size = total_building_pixels / total_building_instances
                print(f"Average building size: {avg_building_size:.1f} pixels")
            
            if total_road_instances > 0:
                avg_road_size = total_road_pixels / total_road_instances
                print(f"Average road size: {avg_road_size:.1f} pixels")
            
            if total_water_instances > 0:
                avg_water_size = total_water_pixels / total_water_instances
                print(f"Average water size: {avg_water_size:.1f} pixels")
            
            # Show top results
            print(f"\nTop images by total instances:")
            top_results = sorted(results, key=lambda x: x['total_instances'], reverse=True)[:5]
            for i, result in enumerate(top_results):
                print(f"  {i+1}. {result['name']} ({result['split']}/{result['domain']}): "
                      f"{result['total_instances']} instances "
                      f"({result['building_instances']}B + {result['road_instances']}R + {result['water_instances']}W)")
    
    print(f"\nAll results saved to: {output_dir}")
    print(f"Processing complete!")

if __name__ == "__main__":
    main()