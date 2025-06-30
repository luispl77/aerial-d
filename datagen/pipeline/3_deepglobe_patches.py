#!/usr/bin/env python3
"""
Pipeline Step 3: Process DeepGlobe Road Extraction Dataset
Converts DeepGlobe road extraction data to instance segmentation format
compatible with the iSAID pipeline.

This script:
1. Loads DeepGlobe images (1024x1024) and corresponding road masks
2. Crops each image into 4 tiles (512x512 each) or resizes to 480x480
3. Extracts connected components for road pixels
4. Generates XML annotations in the same format as iSAID pipeline
"""

import os
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pycocotools.mask as mask_util
from PIL import Image
import argparse
from skimage import measure
from skimage.morphology import binary_opening, disk
import shutil
import multiprocessing
from multiprocessing import Pool, Value
import time
from skimage.morphology import binary_opening, disk

# DeepGlobe class mapping (for road extraction there's only one class)
DEEPGLOBE_CLASSES = {
    255: 'road'  # White pixels represent roads
}

def create_xml_annotation(filename, width, height, objects):
    """Create XML annotation in the same format as iSAID pipeline"""
    root = ET.Element('annotation')
    
    # Add filename
    ET.SubElement(root, 'filename').text = filename
    
    # Add size
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    
    # Add objects
    for obj in objects:
        object_elem = ET.SubElement(root, 'object')
        ET.SubElement(object_elem, 'name').text = obj['category_name']
        
        # Bounding box
        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(obj['bbox'][0]))
        ET.SubElement(bndbox, 'ymin').text = str(int(obj['bbox'][1]))
        ET.SubElement(bndbox, 'xmax').text = str(int(obj['bbox'][2]))
        ET.SubElement(bndbox, 'ymax').text = str(int(obj['bbox'][3]))
        
        # Add ID
        ET.SubElement(object_elem, 'id').text = str(obj['id'])
        
        # Add segmentation in RLE format
        ET.SubElement(object_elem, 'segmentation').text = str(obj['segmentation_rle'])
        
        # Add area
        ET.SubElement(object_elem, 'area').text = str(obj['area'])
        
        # Add iscrowd (always 0 for individual instances)
        ET.SubElement(object_elem, 'iscrowd').text = "0"
    
    # Add empty nodes section (for future referring expressions)
    ET.SubElement(root, 'nodes')
    
    # Add groups section for class-level groups
    groups_elem = ET.SubElement(root, 'groups')
    
    # Group objects by category (only roads in this case)
    category_groups = {}
    for obj in objects:
        category = obj['category_name']
        if category not in category_groups:
            category_groups[category] = []
        category_groups[category].append(obj)
    
    # Create groups for each category (only roads)
    group_id = 1000000  # Start from high number like in iSAID
    for category, category_objects in category_groups.items():
        if len(category_objects) == 0:
            continue
            
        # Calculate centroid (average of all bounding box centers)
        total_x = 0
        total_y = 0
        for obj in category_objects:
            bbox = obj['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            total_x += center_x
            total_y += center_y
        
        centroid_x = total_x / len(category_objects)
        centroid_y = total_y / len(category_objects)
        
        # Create group element
        group_elem = ET.SubElement(groups_elem, 'group')
        
        # Group ID
        ET.SubElement(group_elem, 'id').text = str(group_id)
        
        # Instance IDs (comma-separated)
        instance_ids = [str(obj['id']) for obj in category_objects]
        ET.SubElement(group_elem, 'instance_ids').text = ','.join(instance_ids)
        
        # Size
        ET.SubElement(group_elem, 'size').text = str(len(category_objects))
        
        # Centroid
        centroid_elem = ET.SubElement(group_elem, 'centroid')
        ET.SubElement(centroid_elem, 'x').text = str(round(centroid_x, 1))
        ET.SubElement(centroid_elem, 'y').text = str(round(centroid_y, 1))
        
        # Category
        ET.SubElement(group_elem, 'category').text = category
        
        # Create combined group segmentation mask
        if category_objects:
            # Start with empty mask
            combined_mask = np.zeros((height, width), dtype=np.uint8)
            
            # Combine all individual masks
            for obj in category_objects:
                if 'segmentation_rle' in obj:
                    try:
                        # Decode individual mask
                        rle = {'size': obj['segmentation_rle']['size'], 
                               'counts': obj['segmentation_rle']['counts'].encode('utf-8')}
                        individual_mask = mask_util.decode(rle)
                        # Add to combined mask
                        combined_mask = np.logical_or(combined_mask, individual_mask).astype(np.uint8)
                    except Exception as e:
                        print(f"Error combining mask for object {obj['id']}: {e}")
            
            # Encode combined mask as RLE
            if np.any(combined_mask):
                combined_rle = mask_util.encode(np.asfortranarray(combined_mask))
                combined_rle['counts'] = combined_rle['counts'].decode('utf-8')
                ET.SubElement(group_elem, 'segmentation').text = str(combined_rle)
        
        # Expressions
        expressions_elem = ET.SubElement(group_elem, 'expressions')
        expression_elem = ET.SubElement(expressions_elem, 'expression')
        expression_elem.set('id', '0')
        
        # Create class-level expressions for roads
        expression_text = "all roads in the image"
        expression_elem.text = expression_text
        
        group_id += 1
    
    # Add proper indentation
    def indent(elem, level=0):
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    indent(root)
    return ET.ElementTree(root)

def create_road_group_annotation(filename, width, height, road_mask):
    """Create XML annotation with only a road group (no individual instances)"""
    root = ET.Element('annotation')
    
    # Add filename
    ET.SubElement(root, 'filename').text = filename
    
    # Add size
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    
    # Empty objects section (no individual instances)
    ET.SubElement(root, 'objects')
    
    # Add groups section with one road group
    groups_elem = ET.SubElement(root, 'groups')
    
    # Create road group
    group_elem = ET.SubElement(groups_elem, 'group')
    
    # Group ID
    ET.SubElement(group_elem, 'id').text = "1000000"
    
    # Instance IDs (empty since no individual instances)
    ET.SubElement(group_elem, 'instance_ids').text = ""
    
    # Size (1 for the group)
    ET.SubElement(group_elem, 'size').text = "1"
    
    # Calculate centroid from road mask
    road_binary = (road_mask > 127).astype(np.uint8)
    
    # Apply light morphological opening to clean up noise
    kernel = disk(1)
    road_binary = binary_opening(road_binary, kernel).astype(np.uint8)
    
    # Find bounding box and centroid
    rows, cols = np.where(road_binary == 1)
    if len(rows) > 0:
        min_row, max_row = rows.min(), rows.max()
        min_col, max_col = cols.min(), cols.max()
        centroid_x = (min_col + max_col) / 2
        centroid_y = (min_row + max_row) / 2
    else:
        # Fallback to center if no roads found
        centroid_x = width / 2
        centroid_y = height / 2
    
    # Centroid
    centroid_elem = ET.SubElement(group_elem, 'centroid')
    ET.SubElement(centroid_elem, 'x').text = str(round(centroid_x, 1))
    ET.SubElement(centroid_elem, 'y').text = str(round(centroid_y, 1))
    
    # Grid position (required by step 5)
    grid_position = determine_grid_position(centroid_x, centroid_y, width, height)
    ET.SubElement(group_elem, 'grid_position').text = grid_position
    
    # Category
    ET.SubElement(group_elem, 'category').text = "road"
    
    # Segmentation (entire road mask)
    rle = mask_to_rle(road_binary)
    ET.SubElement(group_elem, 'segmentation').text = str(rle)
    
    # Expressions
    expressions_elem = ET.SubElement(group_elem, 'expressions')
    expression_elem = ET.SubElement(expressions_elem, 'expression')
    expression_elem.set('id', '0')
    expression_elem.text = "all roads in the image"
    
    # Add proper indentation
    def indent(elem, level=0):
        i = "\n" + level*"  "
        if len(elem):
            if not elem.text or not elem.text.strip():
                elem.text = i + "  "
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
            for elem in elem:
                indent(elem, level+1)
            if not elem.tail or not elem.tail.strip():
                elem.tail = i
        else:
            if level and (not elem.tail or not elem.tail.strip()):
                elem.tail = i
    
    indent(root)
    return ET.ElementTree(root)

def determine_grid_position(x, y, image_width, image_height):
    """Determine grid position label based on the 3x3 grid"""
    # Calculate grid boundaries
    third_w = image_width / 3
    third_h = image_height / 3
    
    # Determine vertical position
    if y < third_h:
        vertical = "top"
    elif y < 2*third_h:
        vertical = "center"
    else:
        vertical = "bottom"
    
    # Determine horizontal position
    if x < third_w:
        horizontal = "left"
    elif x < 2*third_w:
        horizontal = "center"
    else:
        horizontal = "right"
    
    # Combine positions
    if vertical == "center" and horizontal == "center":
        position = "in the center"
    else:
        position = f"in the {vertical} {horizontal}"
    
    return position

def mask_to_rle(mask):
    """Convert binary mask to RLE format"""
    rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def extract_instances_from_road_mask(road_mask, min_area=100, max_area=None):
    """
    Extract individual road instances from binary road mask using connected components.
    
    Args:
        road_mask: Binary mask where roads are 255 and background is 0
        min_area: Minimum area for a road component to be kept
        max_area: Maximum area for a road component (None = no limit)
    
    Returns:
        List of road instance dictionaries
    """
    # Convert to binary
    binary_mask = (road_mask > 127).astype(np.uint8)
    
    # Apply morphological opening to clean up noise
    kernel = disk(2)  # Small kernel for roads
    binary_mask = binary_opening(binary_mask, kernel)
    
    # Find connected components
    labeled_mask = measure.label(binary_mask, connectivity=2)
    regions = measure.regionprops(labeled_mask)
    
    instances = []
    for region in regions:
        # Filter by area
        if region.area < min_area:
            continue
        if max_area is not None and region.area > max_area:
            continue
        
        # Extract instance mask
        instance_mask = (labeled_mask == region.label).astype(np.uint8)
        
        # Get bounding box
        bbox = region.bbox  # (min_row, min_col, max_row, max_col)
        bbox = [bbox[1], bbox[0], bbox[3], bbox[2]]  # Convert to [xmin, ymin, xmax, ymax]
        
        # Convert mask to RLE
        rle = mask_to_rle(instance_mask)
        
        instances.append({
            'category_name': 'road',
            'bbox': bbox,
            'area': region.area,
            'segmentation_rle': rle
        })
    
    return instances

# Global counter for multiprocessing
image_counter = None

def init_globals(counter):
    global image_counter
    image_counter = counter

def process_single_deepglobe_image(args):
    global image_counter
    image_path, mask_path, output_prefix, image_output_dir, annotation_output_dir, crop = args
    
    with image_counter.get_lock():
        image_counter.value += 1
    
    return process_deepglobe_image(image_path, mask_path, output_prefix, 
                                 image_output_dir, annotation_output_dir, crop)

def process_deepglobe_image(image_path, mask_path, output_prefix, 
                          image_output_dir, annotation_output_dir, crop=False):
    """
    Process a single DeepGlobe image and generate patches/annotations.
    
    Args:
        image_path: Path to source image
        mask_path: Path to corresponding road mask
        output_prefix: Prefix for output files
        image_output_dir: Directory to save processed images
        annotation_output_dir: Directory to save annotations
        crop: If True, crop into 4 tiles; if False, resize to 480x480
    
    Returns:
        Number of tiles generated
    """
    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        print(f"Warning: Could not load {image_path} or {mask_path}")
        return 0
    
    tiles_generated = 0
    tiles_data = []
    
    if crop:
        # Crop image into 4 tiles (2x2 grid)
        h, w = image.shape[:2]
        tile_h, tile_w = h // 2, w // 2
        
        for row in range(2):
            for col in range(2):
                y1, y2 = row * tile_h, (row + 1) * tile_h
                x1, x2 = col * tile_w, (col + 1) * tile_w
                
                image_tile = image[y1:y2, x1:x2]
                mask_tile = mask[y1:y2, x1:x2]
                
                # Resize to 480x480
                image_tile_resized = cv2.resize(image_tile, (480, 480), interpolation=cv2.INTER_LINEAR)
                mask_tile_resized = cv2.resize(mask_tile, (480, 480), interpolation=cv2.INTER_NEAREST)
                
                tile_id = row * 2 + col
                tiles_data.append((image_tile_resized, mask_tile_resized, tile_id))
    else:
        # Just resize the full image to 480x480
        image_resized = cv2.resize(image, (480, 480), interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, (480, 480), interpolation=cv2.INTER_NEAREST)
        tiles_data.append((image_resized, mask_resized, 0))  # Use tile_id 0 for full image
    
    # Process each tile
    for image_tile, mask_tile, tile_id in tiles_data:
        # Generate filename
        image_filename = f"{output_prefix}_patch_{tile_id}.png"
        xml_filename = f"{output_prefix}_patch_{tile_id}.xml"
        
        # Check if there are roads in this tile
        road_pixels = np.sum(mask_tile > 127)
        
        # Skip tiles with no roads
        if road_pixels < 100:  # Minimum threshold for roads
            continue
        
        # Save image
        image_output_path = os.path.join(image_output_dir, image_filename)
        cv2.imwrite(image_output_path, image_tile)
        
        # Create and save XML annotation with road group
        xml_tree = create_road_group_annotation(image_filename, 480, 480, mask_tile)
        xml_output_path = os.path.join(annotation_output_dir, xml_filename)
        xml_tree.write(xml_output_path, encoding='utf-8', xml_declaration=True)
        
        tiles_generated += 1
    
    return tiles_generated

def get_train_val_split(all_files, train_ratio=0.8, random_seed=None):
    """
    Split files into train and validation sets.
    
    Args:
        all_files: List of all files
        train_ratio: Fraction of files to use for training (default: 0.8)
        random_seed: Random seed for reproducible splits
    
    Returns:
        train_files, val_files: Two lists of files
    """
    import random
    
    # Set random seed for reproducible splits
    if random_seed is not None:
        random.seed(random_seed)
    
    # Shuffle files for random split
    files_copy = all_files.copy()
    random.shuffle(files_copy)
    
    # Calculate split point
    split_point = int(len(files_copy) * train_ratio)
    
    train_files = files_copy[:split_point]
    val_files = files_copy[split_point:]
    
    return train_files, val_files

def process_deepglobe_split(deepglobe_dir, split_name, output_base_dir, num_workers=None, crop=False,
                          num_images=None, random_seed=None):
    """
    Process DeepGlobe dataset with internal train/val split.
    
    Args:
        deepglobe_dir: Path to DeepGlobe dataset directory
        split_name: 'train' or 'val' (both will use train folder but split internally)
        output_base_dir: Base output directory (dataset/)
        num_workers: Number of worker processes (defaults to CPU count)
        crop: If True, crop into 4 tiles instead of resizing full image to 480x480
        num_images: If specified, only process this many images from the split
        random_seed: If specified, use this seed for random image selection and train/val split
    """
    print(f"\nProcessing DeepGlobe {split_name} split...")
    
    # Setup output directories
    split_lower = split_name.lower()
    
    # Images go to dataset/patches/{train|val}/images/
    image_output_dir = os.path.join(output_base_dir, 'patches', split_lower, 'images')
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Annotations go to dataset/patches_rules_expressions_unique/{train|val}/annotations/
    annotation_output_dir = os.path.join(output_base_dir, 'patches_rules_expressions_unique', 
                                      split_lower, 'annotations')
    os.makedirs(annotation_output_dir, exist_ok=True)
    
    # DeepGlobe only has train directory (valid was removed due to no masks)
    # File naming: {id}_sat.jpg for images, {id}_mask.png for masks
    train_dir = os.path.join(deepglobe_dir, 'train')
    
    if not os.path.exists(train_dir):
        print(f"Warning: Missing train directory")
        print(f"Expected: {train_dir}")
        return
        
    # Get all satellite image files (ending with _sat.jpg)
    all_files = os.listdir(train_dir)
    all_image_files = [f for f in all_files if f.endswith('_sat.jpg')]
    
    # Filter to only include files that have corresponding masks
    valid_image_files = []
    for image_file in all_image_files:
        image_id = image_file.replace('_sat.jpg', '')
        mask_path = os.path.join(train_dir, f"{image_id}_mask.png")
        if os.path.exists(mask_path):
            valid_image_files.append(image_file)
    
    valid_image_files.sort()
    print(f"Found {len(valid_image_files)} valid image-mask pairs")
    
    # Split into train/val (80/20)
    train_files, val_files = get_train_val_split(valid_image_files, train_ratio=0.8, random_seed=random_seed)
    
    print(f"Split: {len(train_files)} train, {len(val_files)} val")
    
    # Select the appropriate files based on requested split
    if split_name.lower() == 'train':
        selected_files = train_files
    elif split_name.lower() == 'val':
        selected_files = val_files
    else:
        print(f"Warning: Unknown split '{split_name}', using train")
        selected_files = train_files
    
    print(f"Processing {len(selected_files)} files for {split_name} split")
    
    # Collect all image processing tasks
    all_tasks = []
    
    for image_file in selected_files:
        # Extract the base ID from filename (remove _sat.jpg)
        image_id = image_file.replace('_sat.jpg', '')
        
        # Create output prefix: D{image_id} for DeepGlobe
        output_prefix = f"D{image_id}"
        
        # Full paths
        image_path = os.path.join(train_dir, image_file)
        mask_path = os.path.join(train_dir, f"{image_id}_mask.png")
        
        # Add task to processing queue
        all_tasks.append((image_path, mask_path, output_prefix, 
                        image_output_dir, annotation_output_dir, crop))
    
    # Apply random sampling if requested
    if num_images is not None and len(all_tasks) > num_images:
        import random
        if random_seed is not None:
            random.seed(random_seed)
        original_count = len(all_tasks)
        all_tasks = random.sample(all_tasks, num_images)
        print(f"Randomly selected {num_images} images from {original_count} total")
    
    if not all_tasks:
        print(f"No images found in {split_name} split")
        return
    
    # Set up multiprocessing
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    
    mode_str = "4-tile cropping" if crop else "full image resize"
    print(f"Processing {len(all_tasks)} images using {num_workers} workers ({mode_str})...")
    
    # Initialize global counter for progress tracking
    global image_counter
    image_counter = Value('i', 0)
    
    # Start timing
    start_time = time.time()
    
    # Process images in parallel
    total_tiles = 0
    with Pool(processes=num_workers, initializer=init_globals, initargs=(image_counter,)) as pool:
        # Use tqdm to show progress
        with tqdm(total=len(all_tasks), desc=f"Processing {split_name} images", unit="images") as pbar:
            for tiles_generated in pool.imap_unordered(process_single_deepglobe_image, all_tasks):
                total_tiles += tiles_generated
                pbar.update(1)
    
    # End timing
    end_time = time.time()
    
    print(f"\n{split_name} split complete:")
    print(f"  Processed {len(all_tasks)} images")
    print(f"  Generated {total_tiles} tiles")
    print(f"  Processing time: {end_time - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Process DeepGlobe dataset for RRSIS pipeline')
    parser.add_argument('--deepglobe_dir', type=str, default='deepglobe',
                       help='Path to DeepGlobe dataset directory')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Output directory for processed data')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                       help='Which splits to process (train, val)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker processes (defaults to CPU count)')
    parser.add_argument('--crop', action='store_true',
                       help='Crop into 4 tiles instead of resizing full 1024x1024 image to 480x480')
    parser.add_argument('--num_images', type=int, default=None,
                       help='Number of images to process from each split (default: all)')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for image selection (default: none)')
    
    args = parser.parse_args()
    
    # Verify DeepGlobe directory exists
    if not os.path.exists(args.deepglobe_dir):
        print(f"Error: DeepGlobe directory '{args.deepglobe_dir}' not found!")
        print("Please ensure you have downloaded the DeepGlobe dataset using download_deepglobe.sh.")
        return
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print number of workers being used
    if args.num_workers is None:
        num_workers = multiprocessing.cpu_count()
        print(f"Using default number of workers: {num_workers} (all available CPU cores)")
    else:
        num_workers = args.num_workers
        print(f"Using {num_workers} worker processes")
    
    # Start overall timing
    overall_start_time = time.time()
    
    # Process each split
    for split in args.splits:
        if split not in ['train', 'val']:
            print(f"Warning: Unknown split '{split}', skipping...")
            continue
            
        process_deepglobe_split(args.deepglobe_dir, split, args.output_dir, num_workers, args.crop,
                              args.num_images, args.random_seed)
    
    # End overall timing
    overall_end_time = time.time()
    
    mode_str = "4-tile cropping" if args.crop else "full image resize"
    print(f"\nDeepGlobe processing complete ({mode_str})!")
    print(f"Total processing time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Images saved to: {args.output_dir}/patches/{{train,val}}/images/")
    print(f"Annotations saved to: {args.output_dir}/patches_rules_expressions_unique/{{train,val}}/annotations/")

if __name__ == "__main__":
    main() 