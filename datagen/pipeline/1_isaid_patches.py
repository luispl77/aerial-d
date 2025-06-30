import os
import json
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from tqdm import tqdm
import pycocotools.mask as mask_util
from PIL import Image, ImageDraw
import argparse
import multiprocessing
from multiprocessing import Pool, Value, Manager
import time
import random  # Add random import at the top

def load_instances(isaid_dir, split):
    """Load instance annotations for a specific split"""
    instances_file = os.path.join(isaid_dir, split, 'Annotations', f'instances_{split}.json')
    with open(instances_file, 'r') as f:
        instances = json.load(f)
    return instances

def create_xml_annotation(filename, width, height, objects):
    """Create XML annotation similar to the example format with proper indentation"""
    root = ET.Element('annotation')
    
    # Add filename
    ET.SubElement(root, 'filename').text = filename
    
    # Add size (keep only width and height)
    size = ET.SubElement(root, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    
    # Add objects section for individual instances
    for obj in objects:
        object_elem = ET.SubElement(root, 'object')
        ET.SubElement(object_elem, 'name').text = obj['category_name']
        
        # Bounding box
        bndbox = ET.SubElement(object_elem, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(int(obj['bbox'][0]))
        ET.SubElement(bndbox, 'ymin').text = str(int(obj['bbox'][1]))
        ET.SubElement(bndbox, 'xmax').text = str(int(obj['bbox'][0] + obj['bbox'][2]))
        ET.SubElement(bndbox, 'ymax').text = str(int(obj['bbox'][1] + obj['bbox'][3]))
        
        # Add ID
        ET.SubElement(object_elem, 'id').text = str(obj['id'])
        
        # Add is_cutoff flag
        ET.SubElement(object_elem, 'is_cutoff').text = str(obj.get('is_cutoff', False))
        
        # Add segmentation in RLE format
        if 'segmentation_rle' in obj:
            ET.SubElement(object_elem, 'segmentation').text = str(obj['segmentation_rle'])
        
        # Add area if available
        if 'area' in obj:
            ET.SubElement(object_elem, 'area').text = str(obj['area'])
        
        # Add iscrowd if available
        if 'iscrowd' in obj:
            ET.SubElement(object_elem, 'iscrowd').text = str(obj['iscrowd'])
    
    # Add nodes section for referring expressions (can contain one or more objects)
    nodes_elem = ET.SubElement(root, 'nodes')
    # This section will be populated later with entries like:
    # <node>
    #   <object_ids>1,2,3</object_ids>
    #   <expression>a group of cars</expression>
    # </node>
    
    # Create the ElementTree and add proper indentation
    tree = ET.ElementTree(root)
    
    # Function to add indentation
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
    
    # Apply indentation
    indent(root)
    
    return tree

def polygon_to_mask(polygon, height, width):
    """Convert polygon to binary mask"""
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Reshape polygon to a numpy array with points formatted for cv2
    polygon_np = np.array(polygon).reshape(-1, 2)
    
    # Convert polygon to mask using fillPoly
    cv2.fillPoly(mask, [polygon_np.astype(np.int32)], 1)
    
    return mask

def mask_to_rle(mask):
    """Convert binary mask to RLE format"""
    rle = mask_util.encode(np.asfortranarray(mask))
    # Convert binary data to string representation
    rle_string = {'size': rle['size'], 'counts': rle['counts'].decode('utf-8')}
    return rle_string

def crop_polygon(polygon, patch_x, patch_y, patch_size):
    """Crop a polygon to a specific patch window"""
    # Process two points at a time (x,y pairs)
    patched_polygon = []
    
    for i in range(0, len(polygon), 2):
        if i+1 < len(polygon):  # Ensure we have both x and y
            x, y = polygon[i], polygon[i+1]
            
            # Adjust points relative to patch
            new_x = x - patch_x
            new_y = y - patch_y
            
            # Only include points inside the patch area
            if 0 <= new_x < patch_size and 0 <= new_y < patch_size:
                patched_polygon.extend([new_x, new_y])
    
    return patched_polygon

def is_too_black(image, threshold=0.5):
    """
    Check if an image has too many black pixels
    
    Args:
        image: The image to check (BGR format from cv2)
        threshold: Maximum allowed ratio of black pixels (0-1)
        
    Returns:
        bool: True if image has too many black pixels
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Count black pixels (value < 10)
    black_pixels = np.sum(gray < 10)
    total_pixels = gray.size
    
    # Return True if ratio of black pixels exceeds threshold
    return (black_pixels / total_pixels) > threshold

def sliding_window_patches(image, instances, image_id, patch_size, overlap, min_instance_area=1000, min_pixels_inside=500):
    """
    Generate patches using a sliding window with overlap
    
    Args:
        image: The full image
        instances: Instance annotations
        image_id: ID of the image
        patch_size: Size of each patch
        overlap: Overlap between windows as a ratio (0-0.9)
        min_instance_area: Minimum area threshold for cut-off instances
        min_pixels_inside: Minimum number of pixels that must be inside the patch for a cut-off instance
        
    Returns:
        List of (patched_image, objects, patch_coords) tuples
    """
    height, width = image.shape[:2]
    
    # Ensure the image is large enough to patch
    if width < patch_size or height < patch_size:
        return []
    
    # Calculate step size based on overlap
    step_size = int(patch_size * (1 - overlap))
    step_size = max(1, step_size)  # Ensure step size is at least 1
    
    patches = []
    
    # Generate all possible patches with the sliding window
    for y in range(0, height - patch_size + 1, step_size):
        for x in range(0, width - patch_size + 1, step_size):
            # Get the patch and its objects
            patched_image = image[y:y+patch_size, x:x+patch_size]
            
            # Skip patches that are too black
            if is_too_black(patched_image):
                continue
                
            patched_objects = []
            
            for ann in instances['annotations']:
                if ann['image_id'] != image_id:
                    continue
                
                # Get the original bbox for quick intersection check
                bbox = ann['bbox']
                bbox_x, bbox_y, bbox_w, bbox_h = bbox
                
                # Quick check if the object might intersect with the patch
                if (bbox_x + bbox_w > x and bbox_x < x + patch_size and 
                    bbox_y + bbox_h > y and bbox_y < y + patch_size):
                    
                    # Handle segmentation and calculate actual areas
                    if 'segmentation' in ann and isinstance(ann['segmentation'], list) and len(ann['segmentation']) > 0:
                        # First create the full mask for the entire image
                        full_mask = np.zeros((height, width), dtype=np.uint8)
                        
                        # Process each polygon
                        for polygon in ann['segmentation']:
                            # Convert polygon to mask in full image coordinates
                            polygon_np = np.array(polygon).reshape(-1, 2)
                            cv2.fillPoly(full_mask, [polygon_np.astype(np.int32)], 1)
                        
                        # Now crop the mask to the patch region
                        patch_mask = full_mask[y:y+patch_size, x:x+patch_size]
                        
                        # Calculate actual intersection area using the mask
                        intersection_area = np.sum(patch_mask)
                        original_area = ann['area']  # Use the original area from annotation
                        
                        # Calculate the ratio of the object that is inside the patch
                        intersection_ratio = intersection_area / original_area
                        
                        # Mark as cutoff if:
                        # 1. Less than 50% is inside AND
                        # 2. Less than min_pixels_inside pixels are visible
                        is_cutoff = intersection_ratio < 0.5 and intersection_area < min_pixels_inside
                        
                        # Compute new bounding box from the cropped mask
                        # Find all non-zero points in the mask
                        points = np.argwhere(patch_mask > 0)
                        if len(points) > 0:
                            # Get min and max coordinates
                            min_y, min_x = points.min(axis=0)
                            max_y, max_x = points.max(axis=0)
                            
                            # Create new bbox coordinates
                            new_bbox_x = min_x
                            new_bbox_y = min_y
                            new_bbox_w = max_x - min_x + 1
                            new_bbox_h = max_y - min_y + 1
                            
                            # Create a new annotation for the patched object
                            patched_ann = {
                                'id': ann['id'],
                                'category_id': ann['category_id'],
                                'category_name': ann['category_name'],
                                'bbox': [new_bbox_x, new_bbox_y, new_bbox_w, new_bbox_h],
                                'area': intersection_area,  # Use actual intersection area
                                'iscrowd': ann.get('iscrowd', 0),
                                'is_cutoff': is_cutoff  # Mark as cutoff based on both ratio and pixel count
                            }
                            
                            # Add RLE segmentation
                            segmentation_rle = mask_to_rle(patch_mask.astype(np.uint8))
                            patched_ann['segmentation_rle'] = segmentation_rle
                            
                            patched_objects.append(patched_ann)
            
            # Only include patches with objects
            if patched_objects:
                patches.append((patched_image, patched_objects, (x, y)))
    
    return patches

# Global counter for progress tracking
patch_counter = None

def init_globals(counter):
    global patch_counter
    patch_counter = counter

def process_single_image(args):
    """Process a single image and generate its patches"""
    img_info, all_instances, isaid_dir, output_dir, patch_size, window_overlap, min_instance_area, min_pixels_inside = args
    
    split = img_info['split']
    image_id = img_info['id']
    original_image_name = os.path.splitext(os.path.basename(img_info['file_name']))[0]
    
    # Load the image
    image_path = os.path.join(isaid_dir, split, 'images', 'images', img_info['file_name'])
    image = cv2.imread(image_path)
    
    if image is None:
        return 0
    
    height, width = image.shape[:2]
    
    # Skip images that are too small
    if width < patch_size or height < patch_size:
        return 0
    
    # Get all sliding window patches
    all_patches = sliding_window_patches(
        image, all_instances[split], image_id, patch_size, 
        window_overlap, min_instance_area=min_instance_area,
        min_pixels_inside=min_pixels_inside)
    
    patches_generated = 0
    images_dir = os.path.join(output_dir, 'images')
    annotations_dir = os.path.join(output_dir, 'annotations')
    
    # Process each patch
    for patched_image, patched_objects, patch_coords in all_patches:
        # Get current patch count and increment
        with patch_counter.get_lock():
            current_count = patch_counter.value
            patch_counter.value += 1
        
        # Save the patched image
        patch_filename = f"{original_image_name}_patch_{current_count:06d}.png"
        patch_image_path = os.path.join(images_dir, patch_filename)
        cv2.imwrite(patch_image_path, patched_image)
        
        # Create and save XML annotation
        xml_tree = create_xml_annotation(
            patch_filename, patch_size, patch_size, patched_objects)
        
        xml_path = os.path.join(annotations_dir, f"{original_image_name}_patch_{current_count:06d}.xml")
        xml_tree.write(xml_path)
        
        patches_generated += 1
    
    return patches_generated

def main(isaid_dir, output_dir="patches", num_images=None, start_image_id=None, end_image_id=None, patch_size=480, window_overlap=0.2, min_instance_area=1000, min_pixels_inside=500, num_workers=None, random_seed=None):
    """
    Generate patches from iSAID dataset using parallel processing
    
    Args:
        isaid_dir: Path to iSAID dataset
        output_dir: Output directory for patches
        num_images: Number of iSAID images to process per split (None for full dataset, if specified will randomly select this many images from each split)
        start_image_id: Starting iSAID image ID (inclusive)
        end_image_id: Ending iSAID image ID (inclusive)
        patch_size: Size of each patch
        window_overlap: Overlap between windows as a ratio (0-0.9)
        min_instance_area: Minimum area threshold for cut-off instances
        min_pixels_inside: Minimum number of pixels that must be inside the patch for a cut-off instance
        num_workers: Number of worker processes (defaults to CPU count)
        random_seed: Seed for random selection of images (None for no seed)
    """
    # Set random seed if provided
    if random_seed is not None:
        random.seed(random_seed)
        print(f"\nUsing random seed: {random_seed}")
    
    # Process both splits
    splits = ['train', 'val']
    
    # Print number of workers being used
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
        print(f"\nUsing default number of workers: {num_workers} (all available CPU cores)")
    else:
        print(f"\nUsing {num_workers} worker processes")
    
    # Process each split separately
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Create split-specific output directories
        split_output_dir = os.path.join(output_dir, split)
        images_dir = os.path.join(split_output_dir, 'images')
        annotations_dir = os.path.join(split_output_dir, 'annotations')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Load instances for this split
        instances = load_instances(isaid_dir, split)
        
        # Add split information to images
        all_images = []
        for img in instances['images']:
            img['split'] = split
            all_images.append(img)
        
        # Filter images by ID range if specified
        if start_image_id is not None or end_image_id is not None:
            filtered_images = []
            for img in all_images:
                img_id = img['id']
                if start_image_id is not None and img_id < start_image_id:
                    continue
                if end_image_id is not None and img_id > end_image_id:
                    continue
                filtered_images.append(img)
            all_images = filtered_images
            print(f"Filtered to {len(all_images)} images in ID range {start_image_id or 'min'} to {end_image_id or 'max'}")
        
        # Randomly select specified number of images if num_images is specified
        if num_images is not None:
            if num_images > len(all_images):
                print(f"Warning: Requested {num_images} images but only {len(all_images)} available. Using all available images.")
                num_images = len(all_images)
            all_images = random.sample(all_images, num_images)
            print(f"Randomly selected {num_images} images from {split} split")
        
        # Initialize global counter for progress tracking
        global patch_counter
        patch_counter = Value('i', 0)
        
        # Prepare arguments for parallel processing
        process_args = [
            (img_info, {split: instances}, isaid_dir, split_output_dir, patch_size, window_overlap, min_instance_area, min_pixels_inside)
            for img_info in all_images
        ]
        
        # Set up progress bar for images instead of patches
        pbar = tqdm(total=len(all_images), 
                    desc=f"Processing {split} images", unit="images")
        
        # Start time
        start_time = time.time()
        
        # Process images in parallel
        with Pool(processes=num_workers, initializer=init_globals, initargs=(patch_counter,)) as pool:
            # Process images and update progress
            for patches_generated in pool.imap_unordered(process_single_image, process_args):
                pbar.update(1)  # Update by 1 image at a time
        
        # End time
        end_time = time.time()
        
        pbar.close()
        print(f"\nGenerated {patch_counter.value} patches for {split} split")
        print(f"Patch size: {patch_size}x{patch_size}")
        print(f"Window overlap: {window_overlap * 100:.1f}%")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    print("\nAll splits processed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate patches from iSAID dataset')
    parser.add_argument('--isaid_dir', type=str, default='./isaid',
                      help='Path to iSAID dataset')
    parser.add_argument('--output_dir', type=str, default='dataset/patches',
                      help='Output directory for patches')
    parser.add_argument('--num_images', type=int, default=None,
                      help='Number of iSAID images to process per split (None for full dataset, if specified will randomly select this many images from each split)')
    parser.add_argument('--start_image_id', type=int, default=None,
                      help='Starting iSAID image ID (inclusive)')
    parser.add_argument('--end_image_id', type=int, default=None,
                      help='Ending iSAID image ID (inclusive)')
    parser.add_argument('--patch_size', type=int, default=480,
                      help='Size of each patch')
    parser.add_argument('--window_overlap', type=float, default=0.2,
                      help='Overlap between windows as a ratio (0-0.9)')
    parser.add_argument('--min_instance_area', type=int, default=1000,
                      help='Minimum area threshold for cut-off instances (default: 500 pixels)')
    parser.add_argument('--min_pixels_inside', type=int, default=500,
                      help='Minimum number of pixels that must be inside the patch for a cut-off instance')
    parser.add_argument('--num_workers', type=int, default=None,
                      help='Number of worker processes (defaults to CPU count)')
    parser.add_argument('--random_seed', type=int, default=None,
                      help='Seed for random selection of images (None for no seed)')
    
    args = parser.parse_args()
    main(args.isaid_dir, args.output_dir, args.num_images,
         args.start_image_id, args.end_image_id,
         args.patch_size, args.window_overlap, args.min_instance_area,
         args.min_pixels_inside, args.num_workers, args.random_seed) 