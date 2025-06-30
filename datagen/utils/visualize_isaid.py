import os
import json
import cv2
import numpy as np
import argparse
from pathlib import Path

def load_instances(isaid_dir, split):
    """Load instance annotations for a specific split"""
    instances_file = os.path.join(isaid_dir, split, 'Annotations', f'instances_{split}.json')
    with open(instances_file, 'r') as f:
        instances = json.load(f)
    return instances

def find_image_info(instances, image_id):
    """Find image info in instances data"""
    for img in instances['images']:
        if img['file_name'].startswith(image_id):
            return img
    return None

def visualize_annotations(isaid_dir, image_id, output_path):
    """
    Visualize iSAID annotations for a specific image
    
    Args:
        isaid_dir: Path to iSAID dataset
        image_id: Image ID to visualize (e.g., 'P0707')
        output_path: Path to save the visualization
    """
    # Try to find the image in both train and val sets
    for split in ['train', 'val']:
        instances = load_instances(isaid_dir, split)
        img_info = find_image_info(instances, image_id)
        
        if img_info is not None:
            print(f"Found image in {split} set")
            print(f"Image file: {img_info['file_name']}")
            print(f"Image ID: {img_info['id']}")
            
            # Load the image
            image_path = os.path.join(isaid_dir, split, 'images', 'images', img_info['file_name'])
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Failed to load image: {image_path}")
                return
            
            # Print image size from the loaded image
            height, width = image.shape[:2]
            print(f"Image size: {width}x{height}")
            
            # Create a copy for visualization
            vis_image = image.copy()
            
            # Get all annotations for this image
            image_annotations = [ann for ann in instances['annotations'] if ann['image_id'] == img_info['id']]
            print(f"\nFound {len(image_annotations)} annotations")
            
            # Create a color map for different categories
            categories = {cat['id']: cat['name'] for cat in instances['categories']}
            num_categories = len(categories)
            colors = np.random.randint(0, 255, size=(num_categories, 3), dtype=np.uint8)
            
            # Print category distribution
            category_counts = {}
            for ann in image_annotations:
                cat_name = categories[ann['category_id']]
                category_counts[cat_name] = category_counts.get(cat_name, 0) + 1
            
            print("\nCategory distribution:")
            for cat_name, count in category_counts.items():
                print(f"{cat_name}: {count}")
            
            # Prepare annotation data for JSON output
            annotation_data = {
                'image_info': {
                    'id': img_info['id'],
                    'file_name': img_info['file_name'],
                    'width': width,
                    'height': height,
                    'split': split
                },
                'categories': {str(cat_id): cat_name for cat_id, cat_name in categories.items()},
                'annotations': []
            }
            
            # Draw each instance
            for ann in image_annotations:
                category_id = ann['category_id']
                category_name = categories[category_id]
                color = colors[category_id % num_categories].tolist()
                
                # Create mask from segmentation
                if 'segmentation' in ann and isinstance(ann['segmentation'], list):
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    for polygon in ann['segmentation']:
                        polygon_np = np.array(polygon).reshape(-1, 2)
                        cv2.fillPoly(mask, [polygon_np.astype(np.int32)], 1)
                    
                    # Overlay mask with transparency
                    overlay = vis_image.copy()
                    overlay[mask > 0] = color
                    cv2.addWeighted(overlay, 0.5, vis_image, 0.5, 0, vis_image)
                    
                    # Draw category name and instance ID
                    if 'bbox' in ann:
                        x, y, w, h = [int(v) for v in ann['bbox']]
                        label = f"{category_name} (ID: {ann['id']})"
                        cv2.putText(vis_image, label, (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # Add to annotation data
                    annotation_data['annotations'].append({
                        'id': ann['id'],
                        'category_id': category_id,
                        'category_name': category_name,
                        'bbox': ann['bbox'],
                        'area': ann['area'],
                        'segmentation': ann['segmentation'],
                        'iscrowd': ann.get('iscrowd', 0)
                    })
                else:
                    print(f"Warning: Instance {ann['id']} ({category_name}) has no valid segmentation")
            
            # Save visualization
            cv2.imwrite(output_path, vis_image)
            print(f"\nSaved visualization to {output_path}")
            
            # Save annotation data as JSON
            json_path = str(output_path).replace('.png', '_annotations.json')
            with open(json_path, 'w') as f:
                json.dump(annotation_data, f, indent=2)
            print(f"Saved annotation data to {json_path}")
            
            return
    
    print(f"Image {image_id} not found in train or val sets")

def main():
    parser = argparse.ArgumentParser(description='Visualize iSAID annotations')
    parser.add_argument('image_id', help='Image ID to visualize (e.g., P0707)')
    parser.add_argument('--isaid_dir', default='isaid', help='Path to iSAID dataset')
    parser.add_argument('--output', default=None, help='Output path for visualization')
    
    args = parser.parse_args()
    
    # Set default output path in utils folder
    if args.output is None:
        output_path = Path(__file__).parent / f"{args.image_id}_visualization.png"
    else:
        output_path = Path(args.output)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    visualize_annotations(args.isaid_dir, args.image_id, str(output_path))

if __name__ == '__main__':
    main() 