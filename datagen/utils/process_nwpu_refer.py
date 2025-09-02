#!/usr/bin/env python3
"""
Simple script to process NWPU-Refer dataset.
Loads .jpg images, creates masks from polygon annotations, and referring expressions from pickle files.
"""

import os
import cv2
import json
import pickle
import numpy as np
from PIL import Image, ImageDraw
import argparse
import matplotlib.pyplot as plt


class NWPUReferProcessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images_dir = os.path.join(dataset_path, 'image')
        
        # Load annotations from JSON
        instances_path = os.path.join(dataset_path, 'new_instances.json')
        with open(instances_path, 'r') as f:
            self.instances_data = json.load(f)
        
        # Create lookup dictionaries
        self.images_by_id = {img['id']: img for img in self.instances_data['images']}
        self.annotations_by_id = {ann['id']: ann for ann in self.instances_data['annotations']}
        self.annotations_by_image_id = {}
        for ann in self.instances_data['annotations']:
            image_id = ann['image_id']
            if image_id not in self.annotations_by_image_id:
                self.annotations_by_image_id[image_id] = []
            self.annotations_by_image_id[image_id].append(ann)
        
        # Load referring expressions from pickle
        refs_path = os.path.join(dataset_path, 'new_refs(unc).p')
        with open(refs_path, 'rb') as f:
            self.refs_data = pickle.load(f)
        
        # Group references by split and merge refs with same image_id + expression
        self.refs_by_split = {'train': [], 'val': [], 'test': []}
        self._group_and_merge_refs()
    
    def _group_and_merge_refs(self):
        """Group references by image_id + expression and merge annotations."""
        # First, group by (image_id, expression, split)
        expr_groups = {}
        for ref in self.refs_data:
            # Get English expression if available, otherwise first expression
            expression = None
            for sentence in ref['sentences']:
                if sentence['sent'].replace(' ', '').isascii():  # Simple check for English
                    expression = sentence['sent']
                    break
            if expression is None and ref['sentences']:
                expression = ref['sentences'][0]['sent']  # Fall back to first sentence
            
            key = (ref['image_id'], expression, ref['split'])
            if key not in expr_groups:
                expr_groups[key] = []
            expr_groups[key].append(ref)
        
        # Create merged references
        for (image_id, expression, split), refs in expr_groups.items():
            if split not in self.refs_by_split:
                continue
                
            # Merge all annotation IDs for this expression
            merged_ref = refs[0].copy()  # Start with first ref
            merged_ref['ann_ids'] = [ref['ann_id'] for ref in refs]  # Collect all ann_ids
            merged_ref['ref_ids'] = [ref['ref_id'] for ref in refs]  # Keep track of original ref_ids
            merged_ref['merged_expression'] = expression
            
            self.refs_by_split[split].append(merged_ref)
    
    def load_image(self, file_name):
        """Load image by filename."""
        image_path = os.path.join(self.images_dir, file_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path)
        return np.array(image)
    
    def create_mask_from_annotations(self, ann_ids, image_width, image_height):
        """Create combined binary mask from multiple annotation IDs."""
        mask = Image.new('L', (image_width, image_height), 0)
        
        for ann_id in ann_ids:
            if ann_id not in self.annotations_by_id:
                print(f"Warning: Annotation ID {ann_id} not found")
                continue
                
            annotation = self.annotations_by_id[ann_id]
            segmentation = annotation['segmentation']
            
            if segmentation:
                # Handle polygon format (list of [x1,y1,x2,y2,...])
                for polygon in segmentation:
                    # Convert flat list to list of tuples
                    polygon_points = [(polygon[i], polygon[i+1]) for i in range(0, len(polygon), 2)]
                    ImageDraw.Draw(mask).polygon(polygon_points, outline=255, fill=255)
        
        return np.array(mask)
    
    def get_sample(self, split='val', index=0):
        """Get a sample (image, mask, expression) from specified split."""
        if split not in self.refs_by_split:
            raise ValueError(f"Invalid split: {split}")
        
        refs = self.refs_by_split[split]
        if index >= len(refs):
            raise IndexError(f"Index {index} out of range for {split} split")
        
        ref = refs[index]
        image_id = ref['image_id']
        ann_ids = ref['ann_ids']  # Now a list
        
        # Get image info
        image_info = self.images_by_id[image_id]
        file_name = image_info['file_name']
        image_width = image_info['width']
        image_height = image_info['height']
        
        # Use the merged expression
        expression = ref['merged_expression']
        
        try:
            # Load image
            image = self.load_image(file_name)
            
            # Create combined mask from all annotations
            mask = self.create_mask_from_annotations(ann_ids, image_width, image_height)
            
            # Get bounding boxes of all annotations
            bboxes = []
            category_ids = []
            for ann_id in ann_ids:
                if ann_id in self.annotations_by_id:
                    ann = self.annotations_by_id[ann_id]
                    bboxes.append(ann['bbox'])
                    category_ids.append(ann['category_id'])
            
            return {
                'ref_ids': ref['ref_ids'],  # Now a list
                'image_id': image_id,
                'ann_ids': ann_ids,  # Now a list
                'file_name': file_name,
                'image': image,
                'mask': mask,
                'expression': expression,
                'category_ids': category_ids,  # List of category IDs
                'bboxes': bboxes,  # List of bounding boxes
                'num_objects': len(ann_ids)
            }
        except FileNotFoundError as e:
            print(f"Error loading sample {index}: {e}")
            return None
    
    def get_dataset_info(self):
        """Get basic dataset statistics."""
        return {
            'train_samples': len(self.refs_by_split['train']),
            'val_samples': len(self.refs_by_split['val']),
            'test_samples': len(self.refs_by_split['test']),
            'total_samples': len(self.refs_data),
            'total_images': len(self.instances_data['images']),
            'total_annotations': len(self.instances_data['annotations']),
            'categories': len(self.instances_data['categories'])
        }
    
    def visualize_sample(self, split='val', index=0, save_dir='.'):
        """Visualize a sample with image, mask, and expression."""
        sample = self.get_sample(split, index)
        if sample is None:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(sample['image'])
        axes[0].set_title(f"Image {sample['image_id']} ({sample['file_name']})")
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(sample['mask'], cmap='gray')
        axes[1].set_title("Mask")
        axes[1].axis('off')
        
        # Overlay
        overlay = sample['image'].copy()
        if len(overlay.shape) == 3:
            # Create red overlay for mask
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 0] = sample['mask']  # Red channel for mask
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        # Add expression as title
        plt.suptitle(f"Expression: '{sample['expression']}'", fontsize=14, wrap=True)
        plt.tight_layout()
        
        # Save instead of show since we're in terminal
        output_path = os.path.join(save_dir, f"nwpu_sample_{split}_{index}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")
        
        return output_path


def main():
    parser = argparse.ArgumentParser(description='Process NWPU-Refer dataset')
    parser.add_argument('--dataset_path', type=str, 
                       default='/cfs/home/u035679/aerialseg/datagen/NWPU-Refer',
                       help='Path to NWPU-Refer dataset')
    parser.add_argument('--split', type=str, default='val', 
                       choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--index', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the sample')
    parser.add_argument('--info', action='store_true',
                       help='Show dataset info')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize (when --visualize is used)')
    parser.add_argument('--save_dir', type=str, default='.',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = NWPUReferProcessor(args.dataset_path)
    
    if args.info:
        info = processor.get_dataset_info()
        print("NWPU-Refer Dataset Info:")
        print(f"  Train samples: {info['train_samples']}")
        print(f"  Val samples: {info['val_samples']}")
        print(f"  Test samples: {info['test_samples']}")
        print(f"  Total samples: {info['total_samples']}")
        print(f"  Total images: {info['total_images']}")
        print(f"  Total annotations: {info['total_annotations']}")
        print(f"  Categories: {info['categories']}")
        return
    
    if args.visualize:
        # Create multiple visualizations
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Creating {args.num_samples} visualizations from {args.split} split...")
        
        for i in range(args.num_samples):
            try:
                output_path = processor.visualize_sample(args.split, i, args.save_dir)
                print(f"Sample {i}: {output_path}")
            except (IndexError, FileNotFoundError) as e:
                print(f"Could not create sample {i}: {e}")
                break
        return
    
    # Load and display single sample
    sample = processor.get_sample(args.split, args.index)
    if sample is None:
        print(f"Could not load sample {args.index} from {args.split} split")
        return
    
    print(f"Sample {args.index} from {args.split} split:")
    print(f"  Ref IDs: {sample['ref_ids']}")
    print(f"  Image ID: {sample['image_id']}")
    print(f"  Annotation IDs: {sample['ann_ids']}")
    print(f"  File name: {sample['file_name']}")
    print(f"  Expression: '{sample['expression']}'")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Mask unique values: {np.unique(sample['mask'])}")
    print(f"  Category IDs: {sample['category_ids']}")
    print(f"  Number of objects: {sample['num_objects']}")
    print(f"  Bboxes: {sample['bboxes']}")


if __name__ == "__main__":
    main()