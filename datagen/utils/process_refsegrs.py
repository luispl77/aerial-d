#!/usr/bin/env python3
"""
Simple script to process RefSegRS dataset.
Loads .tif images, masks, and referring expressions from .txt files.
"""

import os
import cv2
import numpy as np
from PIL import Image
import argparse


class RefSegRSProcessor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.images_dir = os.path.join(dataset_path, 'images')
        self.masks_dir = os.path.join(dataset_path, 'masks')
        
        # Load referring expressions from txt files
        self.train_expressions = self._load_expressions('output_phrase_train.txt')
        self.val_expressions = self._load_expressions('output_phrase_val.txt')
        self.test_expressions = self._load_expressions('output_phrase_test.txt')
        
    def _load_expressions(self, filename):
        """Load referring expressions from txt file."""
        expressions = {}
        txt_path = os.path.join(self.dataset_path, filename)
        
        if not os.path.exists(txt_path):
            print(f"Warning: {txt_path} not found")
            return expressions
            
        with open(txt_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and ' ' in line:
                    # Parse format: "1361 paved road"
                    parts = line.split(' ', 1)
                    try:
                        image_id = int(parts[0])
                        expression = parts[1]
                        expressions[line_num] = {
                            'image_id': image_id,
                            'expression': expression
                        }
                    except ValueError:
                        continue
        return expressions
    
    def load_image(self, image_id):
        """Load .tif image by ID."""
        image_path = os.path.join(self.images_dir, f"{image_id}.tif")
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load with PIL to handle .tif format properly
        image = Image.open(image_path)
        return np.array(image)
    
    def load_mask(self, image_id):
        """Load .tif mask by ID."""
        mask_path = os.path.join(self.masks_dir, f"{image_id}.tif")
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Load with PIL to handle .tif format properly
        mask = Image.open(mask_path)
        return np.array(mask)
    
    def get_sample(self, split='train', index=0):
        """Get a sample (image, mask, expression) from specified split."""
        if split == 'train':
            expressions = self.train_expressions
        elif split == 'val':
            expressions = self.val_expressions
        elif split == 'test':
            expressions = self.test_expressions
        else:
            raise ValueError(f"Invalid split: {split}")
        
        expr_ids = list(expressions.keys())
        if index >= len(expr_ids):
            raise IndexError(f"Index {index} out of range for {split} split")
        
        expr_id = expr_ids[index]
        expr_data = expressions[expr_id]
        image_id = expr_data['image_id']
        expression = expr_data['expression']
        
        try:
            image = self.load_image(image_id)
            mask = self.load_mask(image_id)
            
            return {
                'expr_id': expr_id,
                'image_id': image_id,
                'image': image,
                'mask': mask,
                'expression': expression
            }
        except FileNotFoundError as e:
            print(f"Error loading sample {expr_id}: {e}")
            return None
    
    def get_dataset_info(self):
        """Get basic dataset statistics."""
        return {
            'train_samples': len(self.train_expressions),
            'val_samples': len(self.val_expressions),
            'test_samples': len(self.test_expressions),
            'total_samples': len(self.train_expressions) + len(self.val_expressions) + len(self.test_expressions)
        }
    
    def visualize_sample(self, split='train', index=0):
        """Visualize a sample with image, mask, and expression."""
        import matplotlib.pyplot as plt
        
        sample = self.get_sample(split, index)
        if sample is None:
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(sample['image'])
        axes[0].set_title(f"Image {sample['image_id']}")
        axes[0].axis('off')
        
        # Convert mask to grayscale if needed
        mask = sample['mask']
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        
        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title("Mask")
        axes[1].axis('off')
        
        # Overlay
        overlay = sample['image'].copy()
        if len(overlay.shape) == 3:
            mask_colored = np.zeros_like(overlay)
            mask_colored[:, :, 0] = mask  # Red channel for mask
            overlay = cv2.addWeighted(overlay, 0.7, mask_colored, 0.3, 0)
        
        axes[2].imshow(overlay)
        axes[2].set_title("Overlay")
        axes[2].axis('off')
        
        plt.suptitle(f"Expression: '{sample['expression']}'", fontsize=14)
        plt.tight_layout()
        
        # Save instead of show since we're in terminal
        output_path = f"sample_{split}_{index}.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Process RefSegRS dataset')
    parser.add_argument('--dataset_path', type=str, 
                       default='/cfs/home/u035679/aerialseg/datagen/refsegrs/RefSegRS',
                       help='Path to RefSegRS dataset')
    parser.add_argument('--split', type=str, default='train', 
                       choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--index', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize the sample')
    parser.add_argument('--info', action='store_true',
                       help='Show dataset info')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = RefSegRSProcessor(args.dataset_path)
    
    if args.info:
        info = processor.get_dataset_info()
        print("RefSegRS Dataset Info:")
        print(f"  Train samples: {info['train_samples']}")
        print(f"  Val samples: {info['val_samples']}")
        print(f"  Test samples: {info['test_samples']}")
        print(f"  Total samples: {info['total_samples']}")
        return
    
    # Load and display sample
    sample = processor.get_sample(args.split, args.index)
    if sample is None:
        print(f"Could not load sample {args.index} from {args.split} split")
        return
    
    print(f"Sample {args.index} from {args.split} split:")
    print(f"  Expression ID: {sample['expr_id']}")
    print(f"  Image ID: {sample['image_id']}")
    print(f"  Expression: '{sample['expression']}'")
    print(f"  Image shape: {sample['image'].shape}")
    print(f"  Mask shape: {sample['mask'].shape}")
    print(f"  Mask unique values: {np.unique(sample['mask'])}")
    
    if args.visualize:
        processor.visualize_sample(args.split, args.index)


if __name__ == "__main__":
    main()