#!/usr/bin/env python3
"""
Domain Style Transfer Test Script

This script implements and tests various style transfer methods to transform
aerial images from one domain (e.g., iSAID) to look like another domain (e.g., DeepGlobe).
It supports both single-image and multi-image style learning approaches.

Usage:
    python test_style_transfer.py --source_dir /path/to/source --target_dir /path/to/target
    python test_style_transfer.py --method histogram --source img1.png --target img2.png
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.transforms.functional import to_pil_image, to_tensor
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import glob
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from sklearn.metrics import euclidean_distances
import seaborn as sns

class StyleTransferEvaluator:
    """Evaluate style transfer quality using various metrics"""
    
    @staticmethod
    def histogram_distance(img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate Earth Mover's Distance between image histograms"""
        hist1 = cv2.calcHist([img1], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        hist2 = cv2.calcHist([img2], [0, 1, 2], None, [50, 50, 50], [0, 256, 0, 256, 0, 256])
        
        hist1 = hist1.flatten() / hist1.sum()
        hist2 = hist2.flatten() / hist2.sum()
        
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    
    @staticmethod
    def color_moment_distance(img1: np.ndarray, img2: np.ndarray) -> Dict[str, float]:
        """Compare color moments (mean, std) between images"""
        img1_mean = np.mean(img1.reshape(-1, 3), axis=0)
        img2_mean = np.mean(img2.reshape(-1, 3), axis=0)
        
        img1_std = np.std(img1.reshape(-1, 3), axis=0)
        img2_std = np.std(img2.reshape(-1, 3), axis=0)
        
        mean_dist = np.linalg.norm(img1_mean - img2_mean)
        std_dist = np.linalg.norm(img1_std - img2_std)
        
        return {
            'mean_distance': float(mean_dist),
            'std_distance': float(std_dist),
            'total_distance': float(mean_dist + std_dist)
        }

class SimpleStyleTransfer:
    """Simple statistical style transfer methods"""
    
    @staticmethod
    def histogram_matching(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Match histogram of source image to target image"""
        result = np.zeros_like(source)
        
        for channel in range(3):
            source_channel = source[:, :, channel]
            target_channel = target[:, :, channel]
            
            # Calculate cumulative distribution functions
            source_hist, source_bins = np.histogram(source_channel.flatten(), 256, [0, 256])
            target_hist, target_bins = np.histogram(target_channel.flatten(), 256, [0, 256])
            
            source_cdf = source_hist.cumsum()
            target_cdf = target_hist.cumsum()
            
            # Normalize CDFs
            source_cdf_normalized = source_cdf / source_cdf[-1]
            target_cdf_normalized = target_cdf / target_cdf[-1]
            
            # Create mapping
            mapping = np.interp(source_cdf_normalized, target_cdf_normalized, range(256))
            
            # Apply mapping
            result[:, :, channel] = mapping[source_channel]
        
        return result.astype(np.uint8)
    
    @staticmethod
    def color_moment_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Transfer color moments (mean and standard deviation) from target to source"""
        source_float = source.astype(np.float32)
        target_float = target.astype(np.float32)
        
        # Calculate statistics
        source_mean = np.mean(source_float.reshape(-1, 3), axis=0)
        source_std = np.std(source_float.reshape(-1, 3), axis=0)
        
        target_mean = np.mean(target_float.reshape(-1, 3), axis=0)
        target_std = np.std(target_float.reshape(-1, 3), axis=0)
        
        # Apply transformation
        result = source_float.copy()
        for channel in range(3):
            if source_std[channel] > 0:
                result[:, :, channel] = (result[:, :, channel] - source_mean[channel]) * \
                                      (target_std[channel] / source_std[channel]) + target_mean[channel]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def lab_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Transfer color statistics in LAB color space"""
        # Convert to LAB
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        # Calculate statistics for each channel
        source_mean = np.mean(source_lab.reshape(-1, 3), axis=0)
        source_std = np.std(source_lab.reshape(-1, 3), axis=0)
        
        target_mean = np.mean(target_lab.reshape(-1, 3), axis=0)
        target_std = np.std(target_lab.reshape(-1, 3), axis=0)
        
        # Apply transformation
        result_lab = source_lab.copy()
        for channel in range(3):
            if source_std[channel] > 0:
                result_lab[:, :, channel] = (result_lab[:, :, channel] - source_mean[channel]) * \
                                          (target_std[channel] / source_std[channel]) + target_mean[channel]
        
        # Convert back to BGR
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        return result

class AdaINStyleTransfer(nn.Module):
    """Adaptive Instance Normalization style transfer"""
    
    def __init__(self):
        super().__init__()
        # Simple encoder-decoder architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 3, 1, 1), nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1), nn.Tanh(),
        )
    
    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        """Apply AdaIN style transfer"""
        content_features = self.encoder(content)
        style_features = self.encoder(style)
        
        # Apply AdaIN
        stylized_features = self.adaptive_instance_norm(content_features, style_features)
        
        # Decode
        output = self.decoder(stylized_features)
        return output
    
    def adaptive_instance_norm(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        """Adaptive Instance Normalization"""
        assert content_feat.size()[:2] == style_feat.size()[:2]
        size = content_feat.size()
        
        # Calculate statistics
        style_mean = style_feat.view(size[0], size[1], -1).mean(2).view(size[0], size[1], 1, 1)
        style_std = style_feat.view(size[0], size[1], -1).std(2).view(size[0], size[1], 1, 1)
        
        content_mean = content_feat.view(size[0], size[1], -1).mean(2).view(size[0], size[1], 1, 1)
        content_std = content_feat.view(size[0], size[1], -1).std(2).view(size[0], size[1], 1, 1)
        
        # Normalize content and apply style stats
        normalized_content = (content_feat - content_mean) / (content_std + 1e-8)
        return normalized_content * style_std + style_mean

class StyleBank:
    """Build and manage style banks from multiple images"""
    
    def __init__(self):
        self.domain_stats = {}
    
    def build_style_bank(self, image_paths: List[str], domain_name: str) -> Dict:
        """Build style statistics from multiple images"""
        print(f"Building style bank for {domain_name} from {len(image_paths)} images...")
        
        all_means = []
        all_stds = []
        all_lab_means = []
        all_lab_stds = []
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            # RGB statistics
            img_float = img.astype(np.float32)
            mean = np.mean(img_float.reshape(-1, 3), axis=0)
            std = np.std(img_float.reshape(-1, 3), axis=0)
            all_means.append(mean)
            all_stds.append(std)
            
            # LAB statistics
            img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
            lab_mean = np.mean(img_lab.reshape(-1, 3), axis=0)
            lab_std = np.std(img_lab.reshape(-1, 3), axis=0)
            all_lab_means.append(lab_mean)
            all_lab_stds.append(lab_std)
        
        # Calculate domain statistics
        domain_stats = {
            'rgb_mean': np.mean(all_means, axis=0).tolist(),
            'rgb_std': np.mean(all_stds, axis=0).tolist(),
            'lab_mean': np.mean(all_lab_means, axis=0).tolist(),
            'lab_std': np.mean(all_lab_stds, axis=0).tolist(),
            'num_images': len(all_means)
        }
        
        self.domain_stats[domain_name] = domain_stats
        return domain_stats
    
    def apply_style_bank(self, source: np.ndarray, target_domain: str, color_space: str = 'rgb') -> np.ndarray:
        """Apply style bank statistics to source image"""
        if target_domain not in self.domain_stats:
            raise ValueError(f"Domain {target_domain} not found in style bank")
        
        target_stats = self.domain_stats[target_domain]
        
        if color_space == 'lab':
            # LAB color space transfer
            source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
            source_mean = np.mean(source_lab.reshape(-1, 3), axis=0)
            source_std = np.std(source_lab.reshape(-1, 3), axis=0)
            
            target_mean = np.array(target_stats['lab_mean'])
            target_std = np.array(target_stats['lab_std'])
            
            result_lab = source_lab.copy()
            for channel in range(3):
                if source_std[channel] > 0:
                    result_lab[:, :, channel] = (result_lab[:, :, channel] - source_mean[channel]) * \
                                              (target_std[channel] / source_std[channel]) + target_mean[channel]
            
            result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
            return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
        
        else:  # RGB color space
            source_float = source.astype(np.float32)
            source_mean = np.mean(source_float.reshape(-1, 3), axis=0)
            source_std = np.std(source_float.reshape(-1, 3), axis=0)
            
            target_mean = np.array(target_stats['rgb_mean'])
            target_std = np.array(target_stats['rgb_std'])
            
            result = source_float.copy()
            for channel in range(3):
                if source_std[channel] > 0:
                    result[:, :, channel] = (result[:, :, channel] - source_mean[channel]) * \
                                          (target_std[channel] / source_std[channel]) + target_mean[channel]
            
            return np.clip(result, 0, 255).astype(np.uint8)
    
    def save_style_bank(self, filepath: str):
        """Save style bank to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(self.domain_stats, f, indent=2)
        print(f"Style bank saved to {filepath}")
    
    def load_style_bank(self, filepath: str):
        """Load style bank from JSON file"""
        with open(filepath, 'r') as f:
            self.domain_stats = json.load(f)
        print(f"Style bank loaded from {filepath}")

class StyleTransferVisualizer:
    """Create visualizations and comparisons"""
    
    @staticmethod
    def create_comparison_grid(original: np.ndarray, methods_results: Dict[str, np.ndarray], 
                             target_reference: np.ndarray, save_path: str = None) -> plt.Figure:
        """Create a comparison grid showing original, target reference, and all transfer results"""
        num_methods = len(methods_results)
        fig, axes = plt.subplots(2, max(3, (num_methods + 2) // 2), figsize=(4 * max(3, (num_methods + 2) // 2), 8))
        axes = axes.flatten() if num_methods > 1 else [axes] if num_methods == 1 else axes.flatten()
        
        # Original image
        axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original (Source)')
        axes[0].axis('off')
        
        # Target reference
        axes[1].imshow(cv2.cvtColor(target_reference, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Target Reference')
        axes[1].axis('off')
        
        # Method results
        for idx, (method_name, result) in enumerate(methods_results.items()):
            if idx + 2 < len(axes):
                axes[idx + 2].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                axes[idx + 2].set_title(f'{method_name}')
                axes[idx + 2].axis('off')
        
        # Hide unused subplots
        for idx in range(len(methods_results) + 2, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison grid saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_evaluation_metrics(metrics: Dict[str, Dict], save_path: str = None) -> plt.Figure:
        """Plot evaluation metrics for different methods"""
        methods = list(metrics.keys())
        metric_names = list(next(iter(metrics.values())).keys())
        
        fig, axes = plt.subplots(1, len(metric_names), figsize=(5 * len(metric_names), 5))
        if len(metric_names) == 1:
            axes = [axes]
        
        for idx, metric_name in enumerate(metric_names):
            values = [metrics[method][metric_name] for method in methods]
            bars = axes[idx].bar(methods, values)
            axes[idx].set_title(f'{metric_name.replace("_", " ").title()}')
            axes[idx].set_ylabel('Distance')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[idx].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                             f'{value:.3f}', ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Metrics plot saved to {save_path}")
        
        return fig

def get_domain_from_filename(filename: str) -> str:
    """Determine domain based on filename prefix"""
    basename = os.path.basename(filename).upper()
    if basename.startswith('P'):
        return 'iSAID'
    elif basename.startswith('D'):
        return 'DeepGlobe'
    elif basename.startswith('L'):
        return 'LoveDA'
    else:
        return 'Unknown'

def collect_images_by_domain(data_dir: str, max_per_domain: int = 50) -> Dict[str, List[str]]:
    """Collect image paths organized by domain"""
    image_paths = glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True)
    
    domain_images = {'iSAID': [], 'DeepGlobe': [], 'LoveDA': []}
    
    for img_path in image_paths:
        domain = get_domain_from_filename(img_path)
        if domain in domain_images and len(domain_images[domain]) < max_per_domain:
            domain_images[domain].append(img_path)
    
    return domain_images

def main():
    parser = argparse.ArgumentParser(description='Test domain style transfer methods')
    parser.add_argument('--dataset_dir', type=str, default='/cfs/home/u035679/datasets/aeriald',
                       help='Path to dataset directory')
    parser.add_argument('--source_domain', type=str, default='iSAID',
                       choices=['iSAID', 'DeepGlobe', 'LoveDA'],
                       help='Source domain to transform from')
    parser.add_argument('--target_domain', type=str, default='DeepGlobe',
                       choices=['iSAID', 'DeepGlobe', 'LoveDA'],
                       help='Target domain to transform to')
    parser.add_argument('--method', type=str, default='all',
                       choices=['histogram', 'color_moment', 'lab_transfer', 'style_bank', 'all'],
                       help='Style transfer method to use')
    parser.add_argument('--output_dir', type=str, default='./style_transfer_results',
                       help='Directory to save results')
    parser.add_argument('--max_images', type=int, default=10,
                       help='Maximum number of images to process per domain')
    parser.add_argument('--build_style_bank', action='store_true',
                       help='Build and save style banks for all domains')
    parser.add_argument('--style_bank_path', type=str, default='./style_bank.json',
                       help='Path to save/load style bank')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Testing style transfer from {args.source_domain} to {args.target_domain}")
    print(f"Dataset directory: {args.dataset_dir}")
    
    # Collect images by domain
    train_dir = os.path.join(args.dataset_dir, 'train', 'images')
    domain_images = collect_images_by_domain(train_dir, args.max_images)
    
    print(f"\nFound images:")
    for domain, images in domain_images.items():
        print(f"  {domain}: {len(images)} images")
    
    if len(domain_images[args.source_domain]) == 0:
        print(f"No images found for source domain {args.source_domain}")
        return
    
    if len(domain_images[args.target_domain]) == 0:
        print(f"No images found for target domain {args.target_domain}")
        return
    
    # Initialize components
    simple_transfer = SimpleStyleTransfer()
    evaluator = StyleTransferEvaluator()
    visualizer = StyleTransferVisualizer()
    style_bank = StyleBank()
    
    # Build style bank if requested
    if args.build_style_bank:
        print("\nBuilding style banks for all domains...")
        for domain, images in domain_images.items():
            if len(images) > 0:
                style_bank.build_style_bank(images, domain)
        style_bank.save_style_bank(args.style_bank_path)
    elif os.path.exists(args.style_bank_path):
        print(f"\nLoading existing style bank from {args.style_bank_path}")
        style_bank.load_style_bank(args.style_bank_path)
    
    # Process test images
    source_images = domain_images[args.source_domain][:5]  # Test with first 5 images
    target_reference = cv2.imread(domain_images[args.target_domain][0])  # Use first target as reference
    
    all_metrics = {}
    
    for i, source_path in enumerate(source_images):
        print(f"\nProcessing image {i+1}/{len(source_images)}: {os.path.basename(source_path)}")
        
        source_img = cv2.imread(source_path)
        if source_img is None:
            continue
        
        # Apply different style transfer methods
        results = {}
        metrics = {}
        
        if args.method in ['histogram', 'all']:
            print("  - Applying histogram matching...")
            results['Histogram Matching'] = simple_transfer.histogram_matching(source_img, target_reference)
        
        if args.method in ['color_moment', 'all']:
            print("  - Applying color moment transfer...")
            results['Color Moment Transfer'] = simple_transfer.color_moment_transfer(source_img, target_reference)
        
        if args.method in ['lab_transfer', 'all']:
            print("  - Applying LAB color transfer...")
            results['LAB Color Transfer'] = simple_transfer.lab_color_transfer(source_img, target_reference)
        
        if args.method in ['style_bank', 'all'] and args.target_domain in style_bank.domain_stats:
            print("  - Applying style bank transfer...")
            results['Style Bank (RGB)'] = style_bank.apply_style_bank(source_img, args.target_domain, 'rgb')
            results['Style Bank (LAB)'] = style_bank.apply_style_bank(source_img, args.target_domain, 'lab')
        
        # Evaluate results
        for method_name, result in results.items():
            hist_dist = evaluator.histogram_distance(result, target_reference)
            color_moments = evaluator.color_moment_distance(result, target_reference)
            
            metrics[method_name] = {
                'histogram_distance': hist_dist,
                'color_moment_distance': color_moments['total_distance']
            }
        
        all_metrics[f'image_{i+1}'] = metrics
        
        # Create visualization
        output_name = f"{args.source_domain}_to_{args.target_domain}_image_{i+1}.png"
        output_path = os.path.join(args.output_dir, output_name)
        
        visualizer.create_comparison_grid(
            source_img, results, target_reference, 
            save_path=output_path
        )
    
    # Create summary metrics plot
    if all_metrics:
        # Average metrics across all images
        method_names = list(next(iter(all_metrics.values())).keys())
        avg_metrics = {}
        
        for method in method_names:
            avg_metrics[method] = {}
            for metric_name in ['histogram_distance', 'color_moment_distance']:
                values = [all_metrics[img][method][metric_name] for img in all_metrics.keys()]
                avg_metrics[method][metric_name] = np.mean(values)
        
        metrics_plot_path = os.path.join(args.output_dir, f"{args.source_domain}_to_{args.target_domain}_metrics.png")
        visualizer.plot_evaluation_metrics(avg_metrics, save_path=metrics_plot_path)
        
        # Save metrics to JSON
        metrics_json_path = os.path.join(args.output_dir, f"{args.source_domain}_to_{args.target_domain}_metrics.json")
        with open(metrics_json_path, 'w') as f:
            json.dump({
                'average_metrics': avg_metrics,
                'per_image_metrics': all_metrics,
                'settings': {
                    'source_domain': args.source_domain,
                    'target_domain': args.target_domain,
                    'num_images_processed': len(source_images)
                }
            }, f, indent=2)
        
        print(f"\nResults saved to {args.output_dir}")
        print("Summary of average metrics:")
        for method, metrics in avg_metrics.items():
            print(f"  {method}:")
            for metric_name, value in metrics.items():
                print(f"    {metric_name}: {value:.4f}")

if __name__ == '__main__':
    main()