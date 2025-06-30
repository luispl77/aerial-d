import os
import json
import numpy as np
import cv2
from PIL import Image
import shutil

def load_instances(isaid_dir, split):
    """Load instance annotations for a specific split"""
    instances_file = os.path.join(isaid_dir, split, 'Annotations', f'instances_{split}.json')
    with open(instances_file, 'r') as f:
        instances = json.load(f)
    return instances

def visualize_patches(image_path, image_id, instances, patch_size, overlap, output_dir):
    """Visualize all patches from an image, marking valid and invalid ones"""
    # Create output directories
    valid_dir = os.path.join(output_dir, 'valid_patches')
    invalid_dir = os.path.join(output_dir, 'invalid_patches')
    os.makedirs(valid_dir, exist_ok=True)
    os.makedirs(invalid_dir, exist_ok=True)
    
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    height, width = image.shape[:2]
    
    # Calculate step size based on overlap
    step_size = int(patch_size * (1 - overlap))
    step_size = max(1, step_size)  # Ensure step size is at least 1
    
    # Create a copy of the image for visualization
    vis_image = image.copy()
    
    # Track patch counts
    valid_count = 0
    invalid_count = 0
    
    # Generate all possible patches with the sliding window
    for y in range(0, height - patch_size + 1, step_size):
        for x in range(0, width - patch_size + 1, step_size):
            has_valid_object = False
            
            # Check each object
            for ann in instances['annotations']:
                if ann['image_id'] != image_id:
                    continue
                
                # Get the original bbox
                bbox = ann['bbox']
                bbox_x, bbox_y, bbox_w, bbox_h = bbox
                
                # Check if the object intersects with the patch
                if (bbox_x + bbox_w > x and bbox_x < x + patch_size and 
                    bbox_y + bbox_h > y and bbox_y < y + patch_size):
                    
                    # Calculate the intersection area
                    intersection_x1 = max(bbox_x, x)
                    intersection_y1 = max(bbox_y, y)
                    intersection_x2 = min(bbox_x + bbox_w, x + patch_size)
                    intersection_y2 = min(bbox_y + bbox_h, y + patch_size)
                    
                    intersection_width = max(0, intersection_x2 - intersection_x1)
                    intersection_height = max(0, intersection_y2 - intersection_y1)
                    intersection_area = intersection_width * intersection_height
                    
                    # Calculate the original bbox area
                    original_area = bbox_w * bbox_h
                    
                    # Check if at least 50% of the bbox is in the patch
                    if intersection_area / original_area >= 0.5:
                        has_valid_object = True
                        break
            
            # Extract the patch
            patch = image[y:y+patch_size, x:x+patch_size]
            
            # Draw rectangle on visualization image
            color = (0, 255, 0) if has_valid_object else (0, 0, 255)  # Green for valid, Red for invalid
            cv2.rectangle(vis_image, (x, y), (x + patch_size, y + patch_size), color, 2)
            
            # Save the patch
            if has_valid_object:
                cv2.imwrite(os.path.join(valid_dir, f'patch_{valid_count:04d}.png'), patch)
                valid_count += 1
            else:
                cv2.imwrite(os.path.join(invalid_dir, f'patch_{invalid_count:04d}.png'), patch)
                invalid_count += 1
    
    # Save the visualization image
    cv2.imwrite(os.path.join(output_dir, 'patch_visualization.png'), vis_image)
    
    # Save a summary
    with open(os.path.join(output_dir, 'summary.txt'), 'w') as f:
        f.write(f"Image: {os.path.basename(image_path)}\n")
        f.write(f"Total patches: {valid_count + invalid_count}\n")
        f.write(f"Valid patches (with objects): {valid_count}\n")
        f.write(f"Invalid patches (no objects): {invalid_count}\n")
        f.write(f"Patch size: {patch_size}x{patch_size}\n")
        f.write(f"Window overlap: {overlap * 100:.1f}%\n")
        f.write(f"Step size: {step_size} pixels\n")

def main(isaid_dir, output_dir="patch_visualization"):
    # Use the same parameters as in create_patches.py
    patch_size = 480
    window_overlap = 0.2
    
    # Clean up output directory if it exists
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    # Get the first image from the train split
    split = 'train'
    instances = load_instances(isaid_dir, split)
    
    if not instances['images']:
        print("No images found in the dataset")
        return
    
    # Get the first image
    first_image = instances['images'][0]
    image_path = os.path.join(isaid_dir, split, 'images', 'images', first_image['file_name'])
    
    print(f"Processing first image: {first_image['file_name']}")
    visualize_patches(image_path, first_image['id'], instances, patch_size, window_overlap, output_dir)
    print(f"\nVisualization complete. Results saved in: {output_dir}")
    print("Check patch_visualization.png to see all patches marked on the image")
    print("Valid patches are in green, invalid patches are in red")

if __name__ == '__main__':
    isaid_dir = "./isaid"  # Replace with your iSAID dataset path
    main(isaid_dir) 