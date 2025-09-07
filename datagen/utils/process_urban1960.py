#!/usr/bin/env python3
"""
Script to process Urban1960SatBench dataset and visualize samples.
Analyzes mask encoding format and creates sample visualizations.
"""

import os
import cv2
import numpy as np
from PIL import Image
import argparse
import matplotlib.pyplot as plt


class Urban1960Processor:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        
        # Initialize both ISP and SS subdatasets
        self.isp_path = os.path.join(dataset_path, 'Urban1960SatISP')
        self.ss_path = os.path.join(dataset_path, 'Urban1960SatSS')
        
        # Class mappings
        self.class_names = {
            'ISP': {
                0: 'Non-impervious (natural/vegetation)',
                1: 'Impervious surface (buildings/roads)'
            },
            'SS': {
                0: 'Background/non-urban',
                1: 'Buildings',
                2: 'Roads & transportation',
                3: 'Water bodies',
                4: 'Green spaces',
                5: 'Farmland',
                6: 'Other urban-use'
            }
        }
        
        print(f"DEBUG: Initializing Urban1960Processor")
        print(f"  Dataset path: {dataset_path}")
        print(f"  ISP path: {self.isp_path}")
        print(f"  SS path: {self.ss_path}")
        
        # Check if paths exist and list contents
        for name, path in [('ISP', self.isp_path), ('SS', self.ss_path)]:
            if os.path.exists(path):
                contents = os.listdir(path)
                print(f"  {name} directory contents: {contents}")
            else:
                print(f"  {name} directory not found!")
        
        # Load split files for both datasets
        self.splits = {'ISP': {}, 'SS': {}}
        self._load_splits()
    
    def _load_splits(self):
        """Load train/val/test splits from text files."""
        print("DEBUG: Loading dataset splits...")
        
        for dataset_name, dataset_path in [('ISP', self.isp_path), ('SS', self.ss_path)]:
            print(f"DEBUG: Checking {dataset_name} at {dataset_path}")
            
            if not os.path.exists(dataset_path):
                print(f"Warning: {dataset_path} not found")
                continue
                
            for split in ['train', 'val', 'test']:
                split_file = os.path.join(dataset_path, f'labelled_{split}.txt')
                print(f"  DEBUG: Looking for {split_file}")
                
                if os.path.exists(split_file):
                    with open(split_file, 'r') as f:
                        # Remove .png extension and whitespace
                        samples = [line.strip().replace('.png', '') for line in f.readlines() if line.strip()]
                        self.splits[dataset_name][split] = samples
                        print(f"    Loaded {len(samples)} samples for {dataset_name} {split}")
                        if samples:
                            print(f"    First few samples: {samples[:3]}")
                else:
                    self.splits[dataset_name][split] = []
                    print(f"    {split_file} not found, empty split")
    
    def load_image(self, image_name, dataset='ISP'):
        """Load image by name from specified dataset (ISP or SS)."""
        dataset_path = self.isp_path if dataset == 'ISP' else self.ss_path
        image_path = os.path.join(dataset_path, 'image', f'{image_name}.png')
        
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        return np.array(image)
    
    def load_mask(self, image_name, dataset='ISP'):
        """Load mask by name from specified dataset (ISP or SS)."""
        dataset_path = self.isp_path if dataset == 'ISP' else self.ss_path
        mask_dir = 'mask_gt_ISP' if dataset == 'ISP' else 'mask_gt'
        mask_path = os.path.join(dataset_path, mask_dir, f'{image_name}.png')
        
        print(f"  DEBUG: Looking for mask at: {mask_path}")
        
        if not os.path.exists(mask_path):
            print(f"  DEBUG: Mask not found, checking available masks...")
            available_masks = os.listdir(os.path.join(dataset_path, mask_dir))
            print(f"  DEBUG: Available masks (first 5): {available_masks[:5]}")
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Load mask and analyze its properties
        mask_pil = Image.open(mask_path)
        print(f"  DEBUG: PIL Image mode: {mask_pil.mode}, size: {mask_pil.size}")
        
        # Try different loading approaches
        mask_array = np.array(mask_pil)
        print(f"  DEBUG: Loaded mask shape: {mask_array.shape}, dtype: {mask_array.dtype}")
        print(f"  DEBUG: Mask value range: {mask_array.min()} to {mask_array.max()}")
        print(f"  DEBUG: Unique values (first 10): {np.unique(mask_array)[:10]}")
        
        # If it's RGB/RGBA, try to extract class info from different channels
        if len(mask_array.shape) == 3:
            print(f"  DEBUG: Multi-channel mask detected")
            for ch in range(mask_array.shape[2]):
                ch_unique = np.unique(mask_array[:,:,ch])
                print(f"    Channel {ch}: unique values {ch_unique[:10]}...")
            
            # For masks, usually just use first channel (assuming class IDs are in first channel)
            mask_array = mask_array[:,:,0]
            print(f"  DEBUG: Using first channel, unique values: {np.unique(mask_array)[:10]}")
        
        return mask_array
    
    def analyze_mask_encoding(self, dataset='ISP', num_samples=10):
        """Analyze how classes are encoded in the mask pixels."""
        print(f"\nAnalyzing mask encoding for {dataset} dataset...")
        
        # Get some samples from training set
        train_samples = self.splits[dataset].get('train', [])[:num_samples]
        
        all_unique_values = set()
        mask_stats = []
        
        for i, image_name in enumerate(train_samples):
            try:
                mask = self.load_mask(image_name, dataset)
                unique_values = np.unique(mask)
                all_unique_values.update(unique_values)
                
                mask_stats.append({
                    'image_name': image_name,
                    'shape': mask.shape,
                    'unique_values': unique_values,
                    'min_val': unique_values.min(),
                    'max_val': unique_values.max(),
                    'dtype': mask.dtype
                })
                
                if i < 3:  # Show details for first few
                    print(f"  {image_name}: shape={mask.shape}, unique={unique_values}, dtype={mask.dtype}")
                    
            except FileNotFoundError:
                print(f"  Warning: Could not load mask for {image_name}")
        
        print(f"\nOverall statistics for {dataset}:")
        print(f"  Total unique pixel values across all samples: {sorted(all_unique_values)}")
        print(f"  Number of classes detected: {len(all_unique_values)}")
        
        # Try to infer class mapping
        if len(all_unique_values) <= 20:  # Reasonable number of classes
            print(f"  Likely class encoding: pixel value = class ID")
            print(f"  Classes: {sorted(all_unique_values)}")
        
        return mask_stats, sorted(all_unique_values)
    
    def get_sample(self, split='train', index=0, dataset='ISP'):
        """Get a sample (image, mask) from specified split and dataset."""
        print(f"\nDEBUG: Getting sample {index} from {dataset} {split}")
        
        if split not in self.splits[dataset]:
            raise ValueError(f"Invalid split: {split}")
        
        samples = self.splits[dataset][split]
        print(f"DEBUG: Available samples in {dataset} {split}: {len(samples)}")
        
        if index >= len(samples):
            raise IndexError(f"Index {index} out of range for {split} split ({len(samples)} samples)")
        
        image_name = samples[index]
        print(f"DEBUG: Loading sample: {image_name}")
        
        try:
            # Load image and mask
            print(f"DEBUG: Loading image...")
            image = self.load_image(image_name, dataset)
            print(f"DEBUG: Image loaded successfully, shape: {image.shape}")
            
            print(f"DEBUG: Loading mask...")
            mask = self.load_mask(image_name, dataset)
            print(f"DEBUG: Mask loaded successfully")
            
            unique_classes = np.unique(mask)
            print(f"DEBUG: Found {len(unique_classes)} unique classes: {unique_classes}")
            
            return {
                'image_name': image_name,
                'dataset': dataset,
                'image': image,
                'mask': mask,
                'unique_classes': unique_classes,
                'image_shape': image.shape,
                'mask_shape': mask.shape
            }
        except FileNotFoundError as e:
            print(f"Error loading sample {index}: {e}")
            return None
    
    def visualize_sample(self, split='train', index=0, dataset='ISP', save_dir='.'):
        """Visualize a sample with image, mask, and class analysis."""
        sample = self.get_sample(split, index, dataset)
        if sample is None:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(sample['image'])
        axes[0].set_title(f"Image: {sample['image_name']}")
        axes[0].axis('off')
        
        # Mask with labeled colors
        mask_display = axes[1].imshow(sample['mask'], cmap='tab20')
        axes[1].set_title(f"Segmentation Mask")
        axes[1].axis('off')
        
        # Create color legend that matches actual displayed colors
        unique_classes = sample['unique_classes']
        cmap = plt.cm.tab20
        
        # Get the actual color scaling used by imshow
        vmin = sample['mask'].min()
        vmax = sample['mask'].max()
        
        legend_elements = []
        for cls in unique_classes:
            class_name = self.class_names[dataset].get(cls, f'Unknown {cls}')
            # Use the same normalization as imshow
            if vmax > vmin:
                normalized_value = (cls - vmin) / (vmax - vmin)
            else:
                normalized_value = 0
            color = cmap(normalized_value)
            count = np.sum(sample['mask'] == cls)
            percentage = (count / sample['mask'].size) * 100
            legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=color, 
                                               label=f'{cls}: {class_name} ({percentage:.1f}%)'))
        
        axes[1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Class statistics
        axes[2].axis('off')
        class_counts = [(cls, np.sum(sample['mask'] == cls)) for cls in unique_classes]
        
        stats_text = f"Dataset: {dataset}\n"
        stats_text += f"Image shape: {sample['image_shape']}\n"
        stats_text += f"Mask shape: {sample['mask_shape']}\n"
        stats_text += f"Unique classes: {len(unique_classes)}\n\n"
        stats_text += "Class Mapping:\n"
        
        for cls in unique_classes:
            count = np.sum(sample['mask'] == cls)
            percentage = (count / sample['mask'].size) * 100
            class_name = self.class_names[dataset].get(cls, f'Unknown class {cls}')
            stats_text += f"  {cls}: {class_name}\n"
            stats_text += f"     {count:,} pixels ({percentage:.1f}%)\n"
        
        axes[2].text(0.05, 0.95, stats_text, transform=axes[2].transAxes,
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f"Urban1960SatBench Sample - {dataset} {split} #{index}", fontsize=14)
        plt.tight_layout()
        
        # Save visualization
        output_path = os.path.join(save_dir, f"urban1960_{dataset.lower()}_{split}_{index}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Visualization saved to: {output_path}")
        
        return output_path
    
    def get_dataset_info(self):
        """Get basic dataset statistics."""
        info = {}
        
        for dataset in ['ISP', 'SS']:
            dataset_info = {
                'train_samples': len(self.splits[dataset].get('train', [])),
                'val_samples': len(self.splits[dataset].get('val', [])),
                'test_samples': len(self.splits[dataset].get('test', [])),
            }
            dataset_info['total_samples'] = sum(dataset_info.values())
            info[dataset] = dataset_info
        
        return info


def main():
    parser = argparse.ArgumentParser(description='Process Urban1960SatBench dataset')
    parser.add_argument('--dataset_path', type=str,
                       default='/cfs/home/u035679/aerialseg/datagen/Urban1960SatBench',
                       help='Path to Urban1960SatBench dataset')
    parser.add_argument('--dataset', type=str, default='ISP',
                       choices=['ISP', 'SS'],
                       help='Dataset to process (ISP or SS)')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to process')
    parser.add_argument('--index', type=int, default=0,
                       help='Sample index to visualize')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize samples')
    parser.add_argument('--analyze', action='store_true',
                       help='Analyze mask encoding format')
    parser.add_argument('--info', action='store_true',
                       help='Show dataset info')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to process')
    parser.add_argument('--save_dir', type=str, default='.',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = Urban1960Processor(args.dataset_path)
    
    if args.info:
        info = processor.get_dataset_info()
        print("Urban1960SatBench Dataset Info:")
        for dataset, dataset_info in info.items():
            print(f"\n  {dataset} Dataset:")
            print(f"    Train samples: {dataset_info['train_samples']}")
            print(f"    Val samples: {dataset_info['val_samples']}")
            print(f"    Test samples: {dataset_info['test_samples']}")
            print(f"    Total samples: {dataset_info['total_samples']}")
        return
    
    if args.analyze:
        processor.analyze_mask_encoding(args.dataset, args.num_samples)
        return
    
    if args.visualize:
        # Create multiple visualizations
        os.makedirs(args.save_dir, exist_ok=True)
        print(f"Creating {args.num_samples} visualizations from {args.dataset} {args.split} split...")
        
        for i in range(args.num_samples):
            try:
                output_path = processor.visualize_sample(args.split, i, args.dataset, args.save_dir)
                print(f"Sample {i}: {output_path}")
            except (IndexError, FileNotFoundError) as e:
                print(f"Could not create sample {i}: {e}")
                break
        return
    
    # Load and display single sample info
    sample = processor.get_sample(args.split, args.index, args.dataset)
    if sample is None:
        print(f"Could not load sample {args.index} from {args.dataset} {args.split} split")
        return
    
    print(f"Sample {args.index} from {args.dataset} {args.split} split:")
    print(f"  Image name: {sample['image_name']}")
    print(f"  Dataset: {sample['dataset']}")
    print(f"  Image shape: {sample['image_shape']}")
    print(f"  Mask shape: {sample['mask_shape']}")
    print(f"  Unique classes in mask: {sample['unique_classes']}")
    print(f"  Number of classes: {len(sample['unique_classes'])}")


if __name__ == "__main__":
    main()