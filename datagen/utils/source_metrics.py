#!/usr/bin/env python3
"""
Source Dataset Metrics Analysis

This script analyzes the original source datasets (iSAID and LoveDA) to provide:
- iSAID: Number of images, resolution range, total instances
- LoveDA: Number of images only (no instances since it's semantic segmentation)

Based on the parsing logic from pipeline steps 1 and 2.
"""

import os
import json
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import argparse

class SourceDatasetMetrics:
    def __init__(self):
        self.metrics = {
            'isaid': {
                'train': {'images': 0, 'instances': 0, 'resolutions': []},
                'val': {'images': 0, 'instances': 0, 'resolutions': []},
                'total': {'images': 0, 'instances': 0, 'min_resolution': None, 'max_resolution': None, 'avg_resolution': None}
            },
            'loveda': {
                'train': {'images': 0},
                'val': {'images': 0},
                'total': {'images': 0}
            }
        }
    
    def analyze_isaid(self, isaid_dir: str):
        """
        Analyze iSAID dataset structure and content.
        Based on pipeline/1_isaid_patches.py parsing logic.
        
        Expected structure:
        isaid_dir/
        ├── train/
        │   ├── images/images/  # actual images
        │   └── Annotations/instances_train.json
        └── val/
            ├── images/images/  # actual images  
            └── Annotations/instances_val.json
        """
        print("Analyzing iSAID dataset...")
        
        if not os.path.exists(isaid_dir):
            print(f"iSAID directory not found: {isaid_dir}")
            return
        
        splits = ['train', 'val']
        
        for split in splits:
            print(f"\nProcessing iSAID {split} split...")
            
            # Load instance annotations
            instances_file = os.path.join(isaid_dir, split, 'Annotations', f'instances_{split}.json')
            if not os.path.exists(instances_file):
                print(f"  Annotations file not found: {instances_file}")
                continue
                
            with open(instances_file, 'r') as f:
                instances = json.load(f)
            
            # Count instances (annotations)
            num_instances = len(instances.get('annotations', []))
            self.metrics['isaid'][split]['instances'] = num_instances
            
            # Analyze images
            images_dir = os.path.join(isaid_dir, split, 'images', 'images')
            if not os.path.exists(images_dir):
                print(f"  Images directory not found: {images_dir}")
                continue
            
            images_info = instances.get('images', [])
            num_images = len(images_info)
            self.metrics['isaid'][split]['images'] = num_images
            
            # Analyze image resolutions from a sample of images
            print(f"  Analyzing resolutions from {num_images} images...")
            resolutions = []
            
            # Sample up to 100 images for resolution analysis to avoid processing all
            sample_size = min(100, len(images_info))
            step = max(1, len(images_info) // sample_size)
            
            for i in tqdm(range(0, len(images_info), step), desc=f"  Sampling {split} images"):
                img_info = images_info[i]
                img_path = os.path.join(images_dir, img_info['file_name'])
                
                if os.path.exists(img_path):
                    # Read image to get actual dimensions
                    img = cv2.imread(img_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        resolutions.append((width, height))
                else:
                    # Fall back to metadata if image file not found
                    if 'width' in img_info and 'height' in img_info:
                        resolutions.append((img_info['width'], img_info['height']))
            
            self.metrics['isaid'][split]['resolutions'] = resolutions
            
            print(f"  Found {num_images} images, {num_instances} instances")
            if resolutions:
                min_res = min(resolutions, key=lambda x: x[0]*x[1])
                max_res = max(resolutions, key=lambda x: x[0]*x[1])
                print(f"  Resolution range: {min_res} to {max_res}")
        
        # Calculate totals
        self._calculate_isaid_totals()
    
    def _calculate_isaid_totals(self):
        """Calculate total statistics for iSAID."""
        total_images = self.metrics['isaid']['train']['images'] + self.metrics['isaid']['val']['images']
        total_instances = self.metrics['isaid']['train']['instances'] + self.metrics['isaid']['val']['instances']
        
        self.metrics['isaid']['total']['images'] = total_images
        self.metrics['isaid']['total']['instances'] = total_instances
        
        # Combine all resolutions
        all_resolutions = (self.metrics['isaid']['train']['resolutions'] + 
                          self.metrics['isaid']['val']['resolutions'])
        
        if all_resolutions:
            # Find min/max by total pixel count
            min_res = min(all_resolutions, key=lambda x: x[0]*x[1])
            max_res = max(all_resolutions, key=lambda x: x[0]*x[1])
            
            # Calculate average resolution
            avg_width = sum(r[0] for r in all_resolutions) / len(all_resolutions)
            avg_height = sum(r[1] for r in all_resolutions) / len(all_resolutions)
            
            self.metrics['isaid']['total']['min_resolution'] = f"{min_res[0]}x{min_res[1]}"
            self.metrics['isaid']['total']['max_resolution'] = f"{max_res[0]}x{max_res[1]}"
            self.metrics['isaid']['total']['avg_resolution'] = f"{avg_width:.0f}x{avg_height:.0f}"
    
    def analyze_loveda(self, loveda_dir: str):
        """
        Analyze LoveDA dataset structure and content.
        Based on pipeline/2_loveda_patches.py parsing logic.
        
        Expected structure:
        loveda_dir/
        ├── Train/
        │   ├── Urban/
        │   │   ├── images_png/
        │   │   └── masks_png/
        │   └── Rural/
        │       ├── images_png/
        │       └── masks_png/
        └── Val/
            ├── Urban/
            │   ├── images_png/
            │   └── masks_png/
            └── Rural/
                ├── images_png/
                └── masks_png/
        """
        print("\nAnalyzing LoveDA dataset...")
        
        if not os.path.exists(loveda_dir):
            print(f"LoveDA directory not found: {loveda_dir}")
            return
        
        splits = ['Train', 'Val']
        domains = ['Urban', 'Rural']
        
        for split in splits:
            split_lower = split.lower()
            print(f"\nProcessing LoveDA {split} split...")
            
            total_images = 0
            
            for domain in domains:
                images_dir = os.path.join(loveda_dir, split, domain, 'images_png')
                
                if not os.path.exists(images_dir):
                    print(f"  {domain} images directory not found: {images_dir}")
                    continue
                
                # Count PNG images
                image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
                num_images = len(image_files)
                total_images += num_images
                
                print(f"  {domain}: {num_images} images")
            
            self.metrics['loveda'][split_lower]['images'] = total_images
            print(f"  Total {split}: {total_images} images")
        
        # Calculate totals
        total_images = self.metrics['loveda']['train']['images'] + self.metrics['loveda']['val']['images']
        self.metrics['loveda']['total']['images'] = total_images
    
    def generate_report(self, output_path: str = None):
        """Generate a comprehensive report of source dataset metrics."""
        print("\nGenerating source metrics report...")
        
        report = []
        report.append("=== Source Dataset Metrics Report ===")
        report.append("")
        
        # iSAID Section
        report.append("=== iSAID Dataset ===")
        isaid = self.metrics['isaid']
        
        for split in ['train', 'val']:
            report.append(f"\n{split.upper()} Split:")
            report.append(f"  Images: {isaid[split]['images']:,}")
            report.append(f"  Instances: {isaid[split]['instances']:,}")
            
            if isaid[split]['resolutions']:
                resolutions = isaid[split]['resolutions']
                min_res = min(resolutions, key=lambda x: x[0]*x[1])
                max_res = max(resolutions, key=lambda x: x[0]*x[1])
                avg_width = sum(r[0] for r in resolutions) / len(resolutions)
                avg_height = sum(r[1] for r in resolutions) / len(resolutions)
                
                report.append(f"  Resolution range: {min_res[0]}x{min_res[1]} to {max_res[0]}x{max_res[1]}")
                report.append(f"  Average resolution: {avg_width:.0f}x{avg_height:.0f}")
                report.append(f"  Resolutions analyzed: {len(resolutions)} images")
        
        report.append(f"\nTOTAL iSAID:")
        report.append(f"  Images: {isaid['total']['images']:,}")
        report.append(f"  Instances: {isaid['total']['instances']:,}")
        if isaid['total']['min_resolution']:
            report.append(f"  Resolution range: {isaid['total']['min_resolution']} to {isaid['total']['max_resolution']}")
            report.append(f"  Average resolution: {isaid['total']['avg_resolution']}")
        
        # LoveDA Section
        report.append("\n=== LoveDA Dataset ===")
        loveda = self.metrics['loveda']
        
        for split in ['train', 'val']:
            report.append(f"\n{split.upper()} Split:")
            report.append(f"  Images: {loveda[split]['images']:,}")
            report.append("  Note: LoveDA uses semantic segmentation (no individual instances)")
        
        report.append(f"\nTOTAL LoveDA:")
        report.append(f"  Images: {loveda['total']['images']:,}")
        report.append("  Note: All images are 1024x1024 pixels")
        
        # Summary
        report.append("\n=== Summary ===")
        total_isaid_images = isaid['total']['images']
        total_loveda_images = loveda['total']['images']
        total_images = total_isaid_images + total_loveda_images
        
        report.append(f"Total source images: {total_images:,}")
        report.append(f"  iSAID: {total_isaid_images:,} images ({total_isaid_images/total_images*100:.1f}%)")
        report.append(f"  LoveDA: {total_loveda_images:,} images ({total_loveda_images/total_images*100:.1f}%)")
        report.append(f"Total instances (iSAID only): {isaid['total']['instances']:,}")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            print(f"Source metrics report saved to {output_path}")
        
        return report_text

def main():
    parser = argparse.ArgumentParser(description='Analyze source datasets (iSAID and LoveDA)')
    parser.add_argument('--isaid_dir', type=str, default='./isaid',
                       help='Path to iSAID dataset directory')
    parser.add_argument('--loveda_dir', type=str, default='./LoveDA',
                       help='Path to LoveDA dataset directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for report (default: same as script)')
    
    args = parser.parse_args()
    
    # Initialize metrics analyzer
    analyzer = SourceDatasetMetrics()
    
    # Analyze datasets
    analyzer.analyze_isaid(args.isaid_dir)
    analyzer.analyze_loveda(args.loveda_dir)
    
    # Generate report
    if args.output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        report_path = os.path.join(script_dir, "source_metrics_report.txt")
    else:
        report_path = os.path.join(args.output_dir, "source_metrics_report.txt")
    
    report = analyzer.generate_report(report_path)
    
    # Print preview
    print("\nReport Preview:")
    print("=" * 50)
    print(report[:800] + "..." if len(report) > 800 else report)

if __name__ == "__main__":
    main()