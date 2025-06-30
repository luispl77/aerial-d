#!/usr/bin/env python3
"""
Pipeline Step 5: Process LoveDA Dataset
Converts LoveDA semantic segmentation data to instance segmentation format
compatible with the iSAID pipeline.

This script:
1. Loads LoveDA images (1024x1024) and corresponding semantic masks
2. Crops each image into 4 tiles (512x512 each)
3. Resizes tiles to 480x480 to match iSAID patch format
4. Extracts connected components for each semantic class
5. Generates XML annotations in the same format as iSAID pipeline
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
from skimage import measure
from skimage.morphology import binary_opening, binary_closing, disk, remove_small_objects

# LoveDA class mapping (pixel values to class names)
LOVEDA_INSTANCE_CLASSES = {
    2: 'building',
    4: 'water'
}

LOVEDA_GROUP_CLASSES = {
    5: 'barren',
    6: 'forest',
    7: 'agriculture'
}

def create_xml_annotation(filename, width, height, objects, semantic_mask=None):
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
    
        # Create groups for forest, agriculture, barren by checking semantic mask directly
    # These classes are NOT individual instances, they ONLY exist as groups
    group_id = 1000000  # Start from high number like in iSAID
    category_groups = {}
    
    # Create groups for each group class present in the semantic mask
    # Include both INSTANCE classes (building, water) and GROUP classes (barren, forest, agriculture)
    if semantic_mask is not None:
        for class_id, class_name in [(2, 'building'), (4, 'water'), (5, 'barren'), (6, 'forest'), (7, 'agriculture')]:
            # Check if this class exists in the semantic mask
            class_pixels = np.sum(semantic_mask == class_id)
            if class_pixels < 100:  # Skip if too few pixels
                continue
                
            # Create a virtual group instance for this class
            class_mask = (semantic_mask == class_id).astype(np.uint8)
            
            # Apply light morphological opening to clean up noise
            kernel = disk(1)
            class_mask = binary_opening(class_mask, kernel).astype(np.uint8)
            
            total_area = np.sum(class_mask)
            if total_area < 100:
                continue
            
            # Find bounding box of ALL pixels of this class
            rows, cols = np.where(class_mask == 1)
            if len(rows) == 0:
                continue
                
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            # Calculate centroid as center of bounding box
            centroid_x = (min_col + max_col) / 2
            centroid_y = (min_row + max_row) / 2
            
            # Create virtual object for group creation
            virtual_obj = {
                'id': 999999,  # Virtual ID, won't conflict with real instances
                'category_name': class_name,
                'bbox': [min_col, min_row, max_col + 1, max_row + 1],
                'area': int(total_area),
                'segmentation_rle': mask_to_rle(class_mask),
            }
            
            category_groups[class_name] = [virtual_obj]
    
    # Create groups for each group category that was found
    for category, category_objects in category_groups.items():
        if len(category_objects) == 0:
            continue
            
        # Calculate centroid from the virtual object
        obj = category_objects[0]  # Only one object per group class
        bbox = obj['bbox']
        centroid_x = (bbox[0] + bbox[2]) / 2
        centroid_y = (bbox[1] + bbox[3]) / 2
        
        # Create group element
        group_elem = ET.SubElement(groups_elem, 'group')
        
        # Group ID
        ET.SubElement(group_elem, 'id').text = str(group_id)
        
        # Instance IDs (comma-separated) - virtual IDs for group classes
        instance_ids = [str(obj['id']) for obj in category_objects]
        ET.SubElement(group_elem, 'instance_ids').text = ','.join(instance_ids)
        
        # Size (always 1 for group classes since they're single masks)
        ET.SubElement(group_elem, 'size').text = str(len(category_objects))
        
        # Centroid
        centroid_elem = ET.SubElement(group_elem, 'centroid')
        ET.SubElement(centroid_elem, 'x').text = str(round(centroid_x, 1))
        ET.SubElement(centroid_elem, 'y').text = str(round(centroid_y, 1))
        
        # Grid position (required by step 5)
        grid_position = determine_grid_position(centroid_x, centroid_y, width, height)
        ET.SubElement(group_elem, 'grid_position').text = grid_position
        
        # Category
        ET.SubElement(group_elem, 'category').text = category
        
        # Use the segmentation from the virtual object (single mask for group class)
        if category_objects and 'segmentation_rle' in category_objects[0]:
            ET.SubElement(group_elem, 'segmentation').text = str(category_objects[0]['segmentation_rle'])
        
        # Expressions
        expressions_elem = ET.SubElement(group_elem, 'expressions')
        expression_elem = ET.SubElement(expressions_elem, 'expression')
        expression_elem.set('id', '0')
        
        # Create class-level expressions (since each "object" is actually all pixels of that class)
        class_expressions = {
            'building': 'all buildings in the image',
            'water': 'all water in the image',
            'barren': 'all barren land in the image',
            'forest': 'all forest areas in the image',
            'agriculture': 'all agricultural land in the image'
        }
        
        # Since we now have one mask per class containing ALL pixels of that class,
        # we always use the "all" form regardless of the number of objects
        expression_text = class_expressions.get(category, f"all {category} in the image")
        
        expression_elem.text = expression_text
        
        group_id += 1
    
    # Combination groups removed - not useful for the dataset
    
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

def mask_to_rle(mask):
    """Convert binary mask to RLE format"""
    rle = mask_util.encode(np.asfortranarray(mask))
    rle_string = {'size': rle['size'], 'counts': rle['counts'].decode('utf-8')}
    return rle_string

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

def extract_building_instances(semantic_mask, min_area=50, max_area=50000):
    """
    Extract individual building instances from semantic mask using connected components
    
    Args:
        semantic_mask: 2D array with class pixel values
        min_area: Minimum area threshold for buildings
        max_area: Maximum area threshold for buildings
    
    Returns:
        list: List of instance dictionaries
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
        rle = mask_to_rle(instance_mask)
        
        instances.append({
            'id': 1,  # Will be updated globally later
            'category_name': 'building',
            'bbox': bbox,
            'area': int(area),
            'segmentation_rle': rle,
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
        rle = mask_to_rle(instance_mask)
        
        instances.append({
            'id': 1,  # Will be updated globally later
            'category_name': 'water',
            'bbox': bbox,
            'area': int(area),
            'segmentation_rle': rle,
        })
    
    return instances

def extract_group_class_mask(semantic_mask, class_id, class_name, min_area=100):
    """
    Extract single mask for group classes (agriculture, forest, barren)
    
    Args:
        semantic_mask: 2D array with class pixel values
        class_id: The class ID to extract
        class_name: The name of the class
        min_area: Minimum area for the class to be included
        
    Returns:
        Dictionary with single instance containing all pixels of this class, or None
    """
    # Create binary mask for this class
    class_mask = (semantic_mask == class_id).astype(np.uint8)
    
    if np.sum(class_mask) == 0:
        return None
    
    # Apply light morphological opening to clean up noise
    kernel = disk(1)
    class_mask = binary_opening(class_mask, kernel).astype(np.uint8)
    
    # Check if there are still pixels after cleaning
    total_area = np.sum(class_mask)
    if total_area < min_area:
        return None
    
    # Find bounding box of ALL pixels of this class
    rows, cols = np.where(class_mask == 1)
    if len(rows) == 0:
        return None
        
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()
    
    # Convert to RLE (the mask contains ALL pixels of this class)
    rle = mask_to_rle(class_mask)
    
    # Create single instance dictionary for this class
    instance = {
        'id': 1,  # Will be updated globally later
        'category_name': class_name,
        'bbox': [min_col, min_row, max_col + 1, max_row + 1],  # [xmin, ymin, xmax, ymax]
        'area': int(total_area),
        'segmentation_rle': rle,
    }
    
    return instance



# Global counter for progress tracking
image_counter = None

def init_globals(counter):
    global image_counter
    image_counter = counter

def process_single_loveda_image(args):
    """Process a single LoveDA image - worker function for multiprocessing"""
    image_path, mask_path, output_prefix, image_output_dir, annotation_output_dir, crop = args
    
    try:
        tiles_processed = process_loveda_image(image_path, mask_path, output_prefix, 
                                             image_output_dir, annotation_output_dir, crop)
        
        # Update progress counter
        with image_counter.get_lock():
            image_counter.value += 1
            
        return tiles_processed
    except Exception as e:
        print(f"Error processing {os.path.basename(image_path)}: {str(e)}")
        return 0

def process_loveda_image(image_path, mask_path, output_prefix, 
                        image_output_dir, annotation_output_dir, crop=False):
    """
    Process a single LoveDA image:
    1. Load image and mask
    2. Either resize full image OR create 4 tiles (2x2 grid) (based on crop flag)
    3. Resize each tile/image to 480x480
    4. Extract instances from each tile/image
    5. Save images and XML annotations
    """
    # Load image and mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None or mask is None:
        print(f"Warning: Could not load {image_path} or {mask_path}")
        return
    
    # Verify dimensions
    if image.shape[:2] != (1024, 1024) or mask.shape != (1024, 1024):
        print(f"Warning: Unexpected dimensions for {image_path}")
        return
    
    # Process image based on crop flag
    tiles_data = []
    tiles_generated = 0
    
    if crop:
        # Create 4 tiles (2x2 grid)
        tile_size = 512
        
        for row in range(2):
            for col in range(2):
                y_start = row * tile_size
                y_end = y_start + tile_size
                x_start = col * tile_size  
                x_end = x_start + tile_size
                
                # Extract tile
                image_tile = image[y_start:y_end, x_start:x_end]
                mask_tile = mask[y_start:y_end, x_start:x_end]
                
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
        
        # Extract ONLY individual instances for buildings and water
        # These will go through the rule-based generation pipeline (steps 4-6)
        all_instances = []
        global_instance_id = 1
        
        # Extract building instances using connected components
        building_instances = extract_building_instances(mask_tile)
        for instance in building_instances:
            instance['id'] = global_instance_id
            global_instance_id += 1
        all_instances.extend(building_instances)
        
        # Extract water instances using connected components  
        water_instances = extract_water_instances(mask_tile)
        for instance in water_instances:
            instance['id'] = global_instance_id
            global_instance_id += 1
        all_instances.extend(water_instances)
        
        # NOTE: forest, agriculture, barren are NOT added as individual instances
        # They will ONLY exist as groups with pre-written expressions
        
        # Skip tiles with no instances
        if len(all_instances) == 0:
            continue
            
        # Save image
        image_output_path = os.path.join(image_output_dir, image_filename)
        cv2.imwrite(image_output_path, image_tile)
        
        # Create and save XML annotation
        xml_tree = create_xml_annotation(image_filename, 480, 480, all_instances, mask_tile)
        xml_output_path = os.path.join(annotation_output_dir, xml_filename)
        xml_tree.write(xml_output_path, encoding='utf-8', xml_declaration=True)
        
        tiles_generated += 1
    
    return tiles_generated

def process_loveda_split(loveda_dir, split_name, output_base_dir, num_workers=None, crop=False, 
                        num_images=None, random_seed=None):
    """
    Process a complete LoveDA split (Train or Val) using multiprocessing.
    
    Args:
        loveda_dir: Path to LoveDA dataset directory
        split_name: 'Train' or 'Val' 
        output_base_dir: Base output directory (dataset/)
        num_workers: Number of worker processes (defaults to CPU count)
        crop: If True, crop into 4 tiles instead of resizing full image to 480x480
        num_images: If specified, only process this many images from the split
        random_seed: If specified, use this seed for random image selection
    """
    print(f"\nProcessing LoveDA {split_name} split...")
    
    # Setup output directories
    split_lower = split_name.lower()
    
    # Images go to dataset/patches/{train|val}/images/
    image_output_dir = os.path.join(output_base_dir, 'patches', split_lower, 'images')
    os.makedirs(image_output_dir, exist_ok=True)
    
    # Annotations go to dataset/patches/{train|val}/annotations/ (to be processed by rule generation pipeline)
    annotation_output_dir = os.path.join(output_base_dir, 'patches', split_lower, 'annotations')
    os.makedirs(annotation_output_dir, exist_ok=True)
    
    # Collect all image processing tasks
    all_tasks = []
    
    # Process both Urban and Rural
    for domain in ['Urban', 'Rural']:
        images_dir = os.path.join(loveda_dir, split_name, domain, 'images_png')
        masks_dir = os.path.join(loveda_dir, split_name, domain, 'masks_png')
        
        if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
            print(f"Warning: Missing {domain} directory for {split_name}")
            continue
            
        # Get all image files
        image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
        image_files.sort()
        
        print(f"Found {len(image_files)} {domain.lower()} images")
        
        for image_file in image_files:
            image_name = os.path.splitext(image_file)[0]            
            # Create output prefix: L{image_number}
            output_prefix = f"L{image_name}"
            
            # Full paths
            image_path = os.path.join(images_dir, image_file)
            mask_path = os.path.join(masks_dir, image_file)
            
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
            for tiles_generated in pool.imap_unordered(process_single_loveda_image, all_tasks):
                total_tiles += tiles_generated
                pbar.update(1)
    
    # End timing
    end_time = time.time()
    
    print(f"\n{split_name} split complete:")
    print(f"  Processed {len(all_tasks)} images")
    print(f"  Generated {total_tiles} tiles")
    print(f"  Processing time: {end_time - start_time:.2f} seconds")

def main():
    parser = argparse.ArgumentParser(description='Process LoveDA dataset for RRSIS pipeline')
    parser.add_argument('--loveda_dir', type=str, default='LoveDA',
                       help='Path to LoveDA dataset directory')
    parser.add_argument('--output_dir', type=str, default='dataset',
                       help='Output directory for processed data')
    parser.add_argument('--splits', nargs='+', default=['Train', 'Val'],
                       help='Which splits to process (Train, Val)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of worker processes (defaults to CPU count)')
    parser.add_argument('--crop', action='store_true',
                       help='Crop into 4 tiles instead of resizing full 1024x1024 image to 480x480')
    parser.add_argument('--num_images', type=int, default=None,
                       help='Number of images to process from each split (default: all)')
    parser.add_argument('--random_seed', type=int, default=None,
                       help='Random seed for image selection (default: none)')
    
    args = parser.parse_args()
    
    # Verify LoveDA directory exists
    if not os.path.exists(args.loveda_dir):
        print(f"Error: LoveDA directory '{args.loveda_dir}' not found!")
        print("Please ensure you have downloaded the LoveDA dataset.")
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
        if split not in ['Train', 'Val']:
            print(f"Warning: Unknown split '{split}', skipping...")
            continue
            
        split_dir = os.path.join(args.loveda_dir, split)
        if not os.path.exists(split_dir):
            print(f"Warning: Split directory '{split_dir}' not found, skipping...")
            continue
            
        process_loveda_split(args.loveda_dir, split, args.output_dir, num_workers, args.crop,
                            args.num_images, args.random_seed)
    
    # End overall timing
    overall_end_time = time.time()
    
    mode_str = "4-tile cropping" if args.crop else "full image resize"
    print(f"\nLoveDA processing complete ({mode_str})!")
    print(f"Total processing time: {overall_end_time - overall_start_time:.2f} seconds")
    print(f"Images saved to: {args.output_dir}/patches/{{train,val}}/images/")
    print(f"Annotations saved to: {args.output_dir}/patches/{{train,val}}/annotations/")

if __name__ == "__main__":
    main() 