import torch
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from pycocotools import mask as mask_utils
from torchmetrics import JaccardIndex
from tqdm import tqdm
import random

from model import SigLipSamSegmentator

def parse_args():
    parser = argparse.ArgumentParser(description='Test SigLIP+SAM segmentation model')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for testing (fixed to 1 for proper IoU calculation)')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--input_size', type=int, default=480, help='Input size for images')
    parser.add_argument('--model_dir', type=str, default='./models', help='Directory with model checkpoint')
    parser.add_argument('--model_name', type=str, required=True, help='Model name for checkpoint loading')
    parser.add_argument('--num_vis', type=int, default=20, help='Number of test images to visualize')
    parser.add_argument('--siglip_model', type=str, default='google/siglip2-so400m-patch14-384', help='SigLIP model name')
    parser.add_argument('--sam_model', type=str, default='facebook/sam-vit-base', help='SAM model name')
    parser.add_argument('--down_spatial_times', type=int, default=2, help='Number of downsampling blocks')
    parser.add_argument('--with_dense_feat', type=bool, default=True, help='Use dense features')
    parser.add_argument('--vis_only', action='store_true', help='Only run visualization without computing metrics')
    parser.add_argument('--dataset_root', type=str, default='./aeriald', help='Root directory of the AERIAL-D dataset')
    parser.add_argument('--dataset_type', type=str, choices=['aeriald', 'rrsisd'], default='aeriald', help='Type of dataset to use for testing')
    parser.add_argument('--rrsisd_root', type=str, default='../datagen/rrsisd', help='Root directory of the RRSISD dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for patch selection')
    
    return parser.parse_args()

def load_model(checkpoint_path, device, args):
    """Load model from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    
    # First initialize model with default parameters
    model = SigLipSamSegmentator(
        siglip_model_name=args.siglip_model,
        sam_model_name=args.sam_model,
        down_spatial_times=args.down_spatial_times,
        with_dense_feat=args.with_dense_feat,
        device='cpu'  # First load to CPU
    )
    
    # Load checkpoint to CPU first
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    
    # Load only trainable parameters
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint['model_state_dict'].items() 
        if k in model_dict
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    missing, unexpected = model.load_state_dict(pretrained_dict, strict=False)
    #print("Missing keys:", missing)
    #print("Unexpected keys:", unexpected)
    # After you've already done: checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # print("All keys in checkpoint['model_state_dict']:\n")
    # for k in checkpoint["model_state_dict"].keys():
    #     print(k)

    # Now transfer to GPU
    model = model.to(device)
    model.eval()
    return model

def calculate_individual_iou(pred, target):
    """Calculate IoU for a single sample at full resolution"""
    # Convert logits to probability
    prob = torch.sigmoid(pred)
    
    # Apply threshold of 0.5
    pred_binary = (prob > 0.5).float()
    
    # Convert target to binary
    target_binary = (target > 0.5).float()
    
    # Calculate IoU
    iou = JaccardIndex(task="binary").to(pred.device)
    iou_score = iou(pred_binary.unsqueeze(0), target_binary.unsqueeze(0))
    
    return iou_score.item(), pred_binary, target_binary

def visualize_predictions(image, mask, pred, text, save_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Convert tensors to numpy arrays
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    print(f"DEBUG VIZ: Original mask shape: {mask.shape}, unique values: {torch.unique(mask)}, sum: {torch.sum(mask)}")
    mask = (mask > 0.5).float().detach().cpu().numpy()
    print(f"DEBUG VIZ: Processed mask shape: {mask.shape}, unique values: {np.unique(mask)}, sum: {np.sum(mask)}")
    
    # Convert logits to probabilities
    prob = torch.sigmoid(pred).detach().cpu().numpy()
    
    # Binary prediction with threshold 0.5
    pred_binary = (prob > 0.5).astype(float)
    
    # Normalize image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std[None, None, :] + mean[None, None, :]
    image = np.clip(image, 0, 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Add text as figure title
    fig.suptitle(f'Expression: "{text}"', wrap=True)
    
    # Plot image, ground truth, probability map, and binary prediction
    axes[0].imshow(image)
    axes[0].set_title('Image')
    
    # Overlay ground truth mask on image
    axes[1].imshow(image)
    axes[1].imshow(mask, cmap='Reds', alpha=0.5)
    axes[1].set_title('Ground Truth Overlay')
    
    # Plot probability map
    prob_plot = axes[2].imshow(prob, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title(f'Probability Map\nmin={prob.min():.3f}, max={prob.max():.3f}')
    fig.colorbar(prob_plot, ax=axes[2])
    
    # Overlay prediction on image
    axes[3].imshow(image)
    axes[3].imshow(pred_binary, cmap='Blues', alpha=0.5)
    axes[3].set_title('Prediction Overlay')
    
    # Turn off axes
    for ax in axes:
        ax.axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig(save_path)
    plt.close()

class SimpleDataset:
    def __init__(self, dataset_root, split='val', input_size=480, max_samples=None, seed=42):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.max_samples = max_samples
        
        # Set random seed
        random.seed(seed)
        
        # Set paths based on split
        self.ann_dir = os.path.join(dataset_root, 'patches', split, 'annotations')
        self.image_dir = os.path.join(dataset_root, 'patches', split, 'images')
        
        # Add transform to match model configuration
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Get list of XML files
        self.xml_files = [f for f in os.listdir(self.ann_dir) if f.endswith('.xml')]
        if self.max_samples is not None:
            # Randomly sample max_samples files
            random.shuffle(self.xml_files)
            self.xml_files = self.xml_files[:self.max_samples]
            
        print(f"\nFound {len(self.xml_files)} XML files in {split} split")
        print("Loading and processing XML files...")
        
        # Store images and their objects separately
        self.images = {}  # filename -> image_path
        self.objects = []  # list of (image_filename, object_data) tuples
        total_objects = 0
        total_groups = 0
        total_expressions = 0
        total_group_expressions = 0
        
        # Use tqdm for progress bar
        for xml_file in tqdm(self.xml_files, desc="Processing XML files"):
            xml_path = os.path.join(self.ann_dir, xml_file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get image filename
            filename = root.find('filename').text
            image_path = os.path.join(self.image_dir, filename)
            
            # Store image path only once
            if filename not in self.images:
                self.images[filename] = image_path
            
            # Create a mapping from object ID to object data for this image
            objects_by_id = {}
            
            # Get all objects with their expressions
            for obj in root.findall('object'):
                # Get object properties
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Get segmentation
                seg = obj.find('segmentation')
                if seg is not None and seg.text:
                    # Parse the segmentation dictionary
                    seg_dict = eval(seg.text)
                    size = seg_dict['size']
                    counts = seg_dict['counts']
                    rle = {'size': size, 'counts': counts}
                else:
                    continue  # Skip objects without segmentation
                
                # Get expressions
                expressions = []
                exp_elem = obj.find('expressions')
                if exp_elem is not None:
                    for exp in exp_elem.findall('expression'):
                        expressions.append(exp.text)
                
                if not expressions:
                    continue  # Skip objects without expressions
                
                total_objects += 1
                total_expressions += len(expressions)
                
                # Get object ID for group processing later
                obj_id = obj.find('id').text if obj.find('id') is not None else None
                
                # Store object data by ID for group processing
                if obj_id is not None:
                    objects_by_id[obj_id] = {
                        'bbox': [xmin, ymin, xmax, ymax],
                        'segmentation': rle,
                        'category': name
                    }
                
                # Add a sample for each expression
                for expression in expressions:
                    self.objects.append({
                        'image_filename': filename,
                        'bbox': [xmin, ymin, xmax, ymax],
                        'segmentation': rle,
                        'expression': expression,
                        'category': name,
                        'type': 'individual'
                    })
            
            # Process groups
            groups_elem = root.find('groups')
            if groups_elem is not None:
                for group in groups_elem.findall('group'):
                    # Get group properties
                    group_id = group.find('id').text
                    instance_ids_text = group.find('instance_ids').text
                    instance_ids = [id.strip() for id in instance_ids_text.split(',')]
                    category = group.find('category').text
                    
                    # Get group expressions
                    group_expressions = []
                    exp_elem = group.find('expressions')
                    if exp_elem is not None:
                        for exp in exp_elem.findall('expression'):
                            group_expressions.append(exp.text)
                    
                    if not group_expressions:
                        continue  # Skip groups without expressions
                    
                    # Collect segmentations for all instances in the group
                    group_segmentations = []
                    group_bboxes = []
                    
                    for instance_id in instance_ids:
                        if instance_id in objects_by_id:
                            group_segmentations.append(objects_by_id[instance_id]['segmentation'])
                            group_bboxes.append(objects_by_id[instance_id]['bbox'])
                    
                    if not group_segmentations:
                        continue  # Skip groups with no valid instances
                    
                    total_groups += 1
                    total_group_expressions += len(group_expressions)
                    
                    # Add a sample for each group expression
                    for expression in group_expressions:
                        self.objects.append({
                            'image_filename': filename,
                            'bbox': group_bboxes,  # List of bboxes for all instances
                            'segmentation': group_segmentations,  # List of segmentations
                            'expression': expression,
                            'category': category,
                            'type': 'group',
                            'instance_ids': instance_ids
                        })
        
        # If in visualization mode, ensure we have unique images
        if self.max_samples is not None:
            # Group objects by image filename
            image_groups = {}
            for obj in self.objects:
                if obj['image_filename'] not in image_groups:
                    image_groups[obj['image_filename']] = []
                image_groups[obj['image_filename']].append(obj)
            
            # Randomly select one object per image (random between groups and individuals)
            self.objects = []
            for filename, objects in image_groups.items():
                # Randomly select one object from this image (groups and individuals together)
                selected_obj = random.choice(objects)
                self.objects.append(selected_obj)
            
            # Shuffle the selected objects
            random.shuffle(self.objects)
            # Limit to max_samples
            self.objects = self.objects[:self.max_samples]
        
        print(f"\nDataset statistics:")
        print(f"- Total patches: {len(self.xml_files)}")
        print(f"- Unique images: {len(self.images)}")
        print(f"- Total individual objects with expressions: {total_objects}")
        print(f"- Total individual expressions: {total_expressions}")
        print(f"- Total groups with expressions: {total_groups}")
        print(f"- Total group expressions: {total_group_expressions}")
        print(f"- Total samples created: {len(self.objects)}")
        if self.max_samples is not None:
            groups_selected = sum(1 for obj in self.objects if obj['type'] == 'group')
            individuals_selected = sum(1 for obj in self.objects if obj['type'] == 'individual')
            print(f"- Selected {len(self.objects)} unique images for visualization:")
            print(f"  - Groups: {groups_selected}")
            print(f"  - Individuals: {individuals_selected}")
    
    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, idx):
        obj = self.objects[idx]
        
        # Load image using stored path
        image = Image.open(self.images[obj['image_filename']]).convert('RGB')
        image = self.transform(image)
        
        # Handle mask creation based on type
        if obj['type'] == 'individual':
            # Single object mask
            binary_mask = mask_utils.decode(obj['segmentation'])
        else:  # group
            # Combine multiple masks for group
            print(f"DEBUG: Processing group with {len(obj['segmentation'])} instances")
            combined_mask = None
            for i, segmentation in enumerate(obj['segmentation']):
                instance_mask = mask_utils.decode(segmentation)
                print(f"DEBUG: Instance {i} mask shape: {instance_mask.shape}, unique values: {np.unique(instance_mask)}, sum: {np.sum(instance_mask)}")
                if combined_mask is None:
                    combined_mask = instance_mask.astype(bool)
                else:
                    # Union of masks (logical OR)
                    combined_mask = np.logical_or(combined_mask, instance_mask.astype(bool))
            binary_mask = combined_mask.astype(np.uint8)
            print(f"DEBUG: Final combined mask shape: {binary_mask.shape}, unique values: {np.unique(binary_mask)}, sum: {np.sum(binary_mask)}")
        
        mask = torch.from_numpy(binary_mask).float()
        mask = T.Resize((self.input_size, self.input_size), antialias=True)(mask.unsqueeze(0)).squeeze(0)
        
        return image, obj['expression'], mask, obj['image_filename'], obj['type']


class RRSISDDataset:
    def __init__(self, dataset_root, split='val', input_size=480, max_samples=None, seed=42):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.max_samples = max_samples
        
        # Set random seed
        random.seed(seed)
        
        # Set paths for RRSISD structure
        self.ann_dir = os.path.join(dataset_root, 'images', 'rrsisd', 'ann_split')
        self.image_dir = os.path.join(dataset_root, 'images', 'rrsisd', 'JPEGImages')
        
        # Add transform to match model configuration
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Get list of XML files
        self.xml_files = [f for f in os.listdir(self.ann_dir) if f.endswith('.xml')]
        
        print(f"\nFound {len(self.xml_files)} XML files in RRSISD dataset")
        print("Loading and processing XML files...")
        
        # Store samples
        self.samples = []
        
        # Use tqdm for progress bar
        for xml_file in tqdm(self.xml_files, desc="Processing RRSISD XML files"):
            xml_path = os.path.join(self.ann_dir, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Get image filename
                filename = root.find('filename').text
                image_path = os.path.join(self.image_dir, filename)
                
                # Check if image exists
                if not os.path.exists(image_path):
                    print(f"Warning: Image {image_path} not found, skipping...")
                    continue
                
                # Get object information
                obj = root.find('object')
                if obj is None:
                    continue
                    
                # Get object properties
                name = obj.find('name')
                name_text = name.text if name is not None else "object"
                
                description = obj.find('description')
                description_text = description.text if description is not None else name_text
                
                # Get segmentation
                seg = obj.find('segmentation')
                if seg is None or not seg.text:
                    print(f"Warning: No segmentation found in {xml_file}, skipping...")
                    continue
                    
                # Parse the segmentation dictionary
                try:
                    seg_dict = eval(seg.text)
                    size = seg_dict['size']
                    counts = seg_dict['counts']
                    rle = {'size': size, 'counts': counts}
                except:
                    print(f"Warning: Failed to parse segmentation in {xml_file}, skipping...")
                    continue
                
                # Add sample
                self.samples.append({
                    'image_path': image_path,
                    'expression': description_text,
                    'segmentation': rle,
                    'category': name_text,
                    'filename': filename
                })
                
            except Exception as e:
                print(f"Warning: Error processing {xml_file}: {e}")
                continue
        
        # If max_samples is specified, randomly sample
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:self.max_samples]
        
        print(f"\nRRSISD Dataset statistics:")
        print(f"- Total valid samples: {len(self.samples)}")
        if self.max_samples is not None:
            print(f"- Selected for processing: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        image = self.transform(image)
        
        # Decode mask from RLE
        binary_mask = mask_utils.decode(sample['segmentation'])
        mask = torch.from_numpy(binary_mask).float()
        mask = T.Resize((self.input_size, self.input_size), antialias=True)(mask.unsqueeze(0)).squeeze(0)
        
        return image, sample['expression'], mask, sample['filename'], 'individual'

def test(model, test_loader, device, output_dir, num_vis=20, vis_only=False):
    model.eval()
    
    # For mIoU calculation
    all_ious = []
    
    # For oIoU calculation - using running counters instead of storing all tensors
    total_intersection = 0
    total_union = 0
    
    vis_count = 0
    
    print("\nTesting model...")
    with torch.no_grad():
        for batch_idx, (images, texts, masks, image_ids, sample_types) in enumerate(test_loader):
            # If in vis_only mode and we've processed enough samples, break
            if vis_only and vis_count >= num_vis:
                break
                
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda'):
                # Forward pass with SigLIP+SAM model
                outputs = model(images, texts)
            
            # Since batch_size=1, we can directly access the first element
            image = images[0]
            mask = masks[0]
            output = outputs[0]
            text = texts[0]
            image_id = image_ids[0]
            sample_type = sample_types[0]
            
            if not vis_only:
                # Calculate IoU for this sample (at full resolution)
                iou_score, pred_binary, target_binary = calculate_individual_iou(output, mask)
                all_ious.append((image_id, iou_score))
                
                # Update running counters for oIoU calculation
                intersection = torch.logical_and(pred_binary, target_binary).sum().item()
                union = torch.logical_or(pred_binary, target_binary).sum().item()
                
                total_intersection += intersection
                total_union += union
            
            # Visualize some predictions
            if vis_count < num_vis:
                save_path = os.path.join(output_dir, f"val_{image_id}.png")
                visualize_predictions(image, mask, output, text, save_path)
                vis_count += 1
            
            # Clean GPU memory after each batch
            del images, masks, outputs, image, mask, output
            if not vis_only:
                del pred_binary, target_binary
            torch.cuda.empty_cache()
            
            if not vis_only:
                print(f"Sample {batch_idx} (ID: {image_id}) - IoU: {iou_score:.4f}")
            else:
                print(f"Visualizing sample {batch_idx} (ID: {image_id}) - Type: {sample_type}")
            
            # If in vis_only mode and we've processed enough samples, break
            if vis_only and vis_count >= num_vis:
                break
    
    if not vis_only:
        # Calculate mIoU (mean of individual IoUs)
        iou_values = [iou for _, iou in all_ious]
        miou = sum(iou_values) / len(iou_values)
        
        # Calculate oIoU using the running counters
        oiou = total_intersection / total_union if total_union > 0 else 0.0
        
        print("\nValidation Results:")
        print(f"mIoU (mean of individual IoUs): {miou:.4f}")
        print(f"oIoU (overall IoU): {oiou:.4f}")
        print(f"Min IoU: {min(iou_values):.4f}")
        print(f"Max IoU: {max(iou_values):.4f}")
        print(f"Median IoU: {np.median(iou_values):.4f}")
        
        # Save results to file
        results_file = os.path.join(output_dir, "validation_results.txt")
        with open(results_file, 'w') as f:
            f.write(f"mIoU (mean of individual IoUs): {miou:.4f}\n")
            f.write(f"oIoU (overall IoU): {oiou:.4f}\n")
            f.write(f"Min IoU: {min(iou_values):.4f}\n")
            f.write(f"Max IoU: {max(iou_values):.4f}\n")
            f.write(f"Median IoU: {np.median(iou_values):.4f}\n\n")
            
            f.write("Individual IoU scores:\n")
            for image_id, iou in sorted(all_ious, key=lambda x: x[0]):
                f.write(f"Image {image_id}: {iou:.4f}\n")
        
        # Plot IoU histogram
        plt.figure(figsize=(10, 6))
        plt.hist(iou_values, bins=20, alpha=0.7)
        plt.axvline(miou, color='r', linestyle='--', label=f'mIoU: {miou:.4f}')
        plt.axvline(oiou, color='b', linestyle='--', label=f'oIoU: {oiou:.4f}')
        plt.axvline(np.median(iou_values), color='g', linestyle='--', label=f'Median IoU: {np.median(iou_values):.4f}')
        plt.xlabel('IoU')
        plt.ylabel('Count')
        plt.title('Distribution of IoU Scores')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "iou_histogram.png"))
        plt.close()
        
        # Create IoU vs. Image ID scatter plot
        plt.figure(figsize=(12, 6))
        image_ids = [id for id, _ in all_ious]
        iou_values = [iou for _, iou in all_ious]
        plt.scatter(image_ids, iou_values, alpha=0.7, s=10)
        plt.axhline(miou, color='r', linestyle='--', label=f'mIoU: {miou:.4f}')
        plt.axhline(oiou, color='b', linestyle='--', label=f'oIoU: {oiou:.4f}')
        plt.xlabel('Image ID')
        plt.ylabel('IoU')
        plt.title('IoU vs. Image ID')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, "iou_by_image.png"))
        plt.close()
        
        return {"miou": miou, "oiou": oiou}
    else:
        print("\nVisualization complete.")
        return None

def main():
    # Parse command line arguments
    args = parse_args()
    
    print("Starting validation script for SigLIP+SAM model")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')
    
    # Force batch size to 1 for proper IoU calculation
    args.batch_size = 1
    
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    print(f"Using device: {device}")
    
    # Create checkpoint path
    checkpoint_path = os.path.join(args.model_dir, args.model_name, 'best.pt')
    
    # Load model
    model = load_model(checkpoint_path, device, args)
    
    # Create validation dataset based on dataset type
    # If in vis_only mode, only load the number of samples we need
    max_samples = args.num_vis if args.vis_only else None
    
    if args.dataset_type == 'aeriald':
        val_dataset = SimpleDataset(
            dataset_root=args.dataset_root,
            split='val',
            input_size=args.input_size,
            max_samples=max_samples,
            seed=args.seed
        )
    elif args.dataset_type == 'rrsisd':
        val_dataset = RRSISDDataset(
            dataset_root=args.rrsisd_root,
            split='val',  # Not used for RRSISD but kept for compatibility
            input_size=args.input_size,
            max_samples=max_samples,
            seed=args.seed
        )
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle since we already randomly sampled files
        pin_memory=True,
        num_workers=4,
        collate_fn=lambda batch: (
            torch.stack([item[0] for item in batch]),  # images
            [item[1] for item in batch],               # texts
            torch.stack([item[2] for item in batch]),  # masks
            [item[3] for item in batch],               # image_ids
            [item[4] for item in batch]                # sample_types
        )
    )
    
    print(f"\nStarting validation with {len(val_dataset)} samples...")
    
    # Create output directory with dataset type
    if args.dataset_type == 'rrsisd':
        output_dir = os.path.join('./results', f'{args.model_name}_rrsisd')
    else:
        output_dir = os.path.join('./results', args.model_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Run testing
    metrics = test(model, val_loader, device, output_dir, args.num_vis, args.vis_only)
    
    if not args.vis_only:
        print(f"\nValidation complete. Results saved to {output_dir}")
        print(f"Final mIoU: {metrics['miou']:.4f}")
        print(f"Final oIoU: {metrics['oiou']:.4f}")
    else:
        print(f"\nVisualization complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 