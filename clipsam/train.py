import torch
import torch.nn as nn
import torch.optim as optim
from model import SigLipSamSegmentator
import os
import json
import shutil
import argparse
from pycocotools import mask as mask_utils
from PIL import Image
import numpy as np
import torchvision.transforms as T
import xml.etree.ElementTree as ET
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler
import torch.nn.functional as F
from datetime import datetime
import random
from tqdm import tqdm

def calculate_grl_lambda(current_iter, total_iters, schedule='linear', max_lambda=1.0):
    """Calculate gradient reversal lambda based on training progress"""
    progress = current_iter / total_iters
    
    if schedule == 'constant':
        return max_lambda
    elif schedule == 'linear':
        return progress * max_lambda
    elif schedule == 'exponential':
        # Exponential schedule: 2 / (1 + exp(-10 * progress)) - 1
        return max_lambda * (2 / (1 + np.exp(-10 * progress)) - 1)
    elif schedule == 'delayed':
        # No GRL for first 70% of training, then gradual ramp
        if progress < 0.7:
            return 0.0
        else:
            return max_lambda * ((progress - 0.7) / 0.3) ** 2
    else:
        return max_lambda

def parse_args():
    parser = argparse.ArgumentParser(description='Train aerial segmentation model with SigLIP+SAM')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--input_size', type=int, default=384, help='Input size for images')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--model_name', type=str, help='Model name for resuming training (required when using --resume)')
    parser.add_argument('--tmp_dir', type=str, default='/tmp/u035679', help='Temporary directory for dataset')
    parser.add_argument('--poly_power', type=float, default=0.9, help='Power factor for polynomial decay')
    parser.add_argument('--grad_accum_steps', type=int, default=2, help='Number of steps to accumulate gradients')
    parser.add_argument('--effective_batch_size', type=int, default=8, help='Target effective batch size (will calculate grad_accum_steps if provided)')
    parser.add_argument('--down_spatial_times', type=int, default=2, help='Number of downsampling blocks')
    parser.add_argument('--with_dense_feat', type=bool, default=True, help='Use dense features')
    # Add LoRA-related arguments
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA for fine-tuning')
    parser.add_argument('--lora_r', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha scaling factor')
    parser.add_argument('--lora_dropout', type=float, default=0.05, help='LoRA dropout rate')
    parser.add_argument('--siglip_model', type=str, default='google/siglip2-so400m-patch14-384', help='SigLIP model name')
    parser.add_argument('--sam_model', type=str, default='facebook/sam-vit-base', help='SAM model name')
    parser.add_argument('--use_historic', action='store_true', default=True, help='Use historic (BW) images instead of normal color images')
    parser.add_argument('--use_transformed', action='store_true', help='Use transformed images (_5, _toD, _toP, _toL) when available')
    
    # Domain adaptation arguments
    parser.add_argument('--enable_domain_adaptation', action='store_true', help='Enable domain adversarial training')
    parser.add_argument('--num_domains', type=int, default=3, help='Number of domains (iSAID=0, DeepGlobe=1, LoveDA=2)')
    parser.add_argument('--domain_loss_weight', type=float, default=0.02, help='Weight for domain adversarial loss')
    parser.add_argument('--grl_lambda_schedule', type=str, default='delayed', choices=['constant', 'linear', 'exponential', 'delayed'], help='GRL lambda scheduling')
    parser.add_argument('--grl_max_lambda', type=float, default=0.5, help='Maximum lambda value for gradient reversal')
    
    # Mid-epoch checkpointing
    parser.add_argument('--save_mid_epoch_checkpoints', action='store_true', help='Save checkpoints at 25%, 50%, 75% of each epoch')
    parser.add_argument('--mid_epoch_intervals', type=int, default=4, help='Number of intervals per epoch to save checkpoints (default: 4 = quarters)')
    
    # Expression filtering
    parser.add_argument('--unique_only', action='store_true', help='Train only on unique expressions (type="unique" in XML). Validation still uses all expressions.')
    parser.add_argument('--one_unique_per_obj', action='store_true', help='If set along with --unique_only, only one unique expression per object/group will be used for training.')
    
    # Balanced batch sampling
    parser.add_argument('--balanced_batch_sampling', action='store_true', help='Enable balanced batch sampling for domain adaptation. Only active if --enable_domain_adaptation is also set.')
    
    # Dataset filtering
    parser.add_argument('--dataset_filter', type=str, choices=['isaid', 'loveda', 'deepglobe'], help='Train only on samples from a specific dataset (isaid, loveda, or deepglobe)')
    
    return parser.parse_args()

def save_checkpoint(model, optimizer, epoch, loss, path):
    """
    Saves a checkpoint with:
      - Only the trained LoRA + prompter weights (excluding big frozen base models)
      - The optimizer state (for resuming training)
      - The current epoch and loss (for logging / resume)
    """
    checkpoint = {
        'model_state_dict': {},
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }

    for k, v in model.state_dict().items():
        # If this key belongs to one of the large base modules but does not belong to LoRA, skip it.
        # LoRA subkeys typically contain 'lora_' or 'lora_A'/'lora_B' in the name.
        if any(x in k for x in ['clip_vision_encoder', 'clip_text_encoder', 'backbone']):
            if 'lora_' not in k:
                continue
        
        # Skip SAM prompt or mask decoder if they are frozen/untrained
        if any(x in k for x in ['sam_prompt_encoder', 'sam_mask_decoder']):
            continue

        # Otherwise, keep this key in the checkpoint.
        checkpoint['model_state_dict'][k] = v
    
    # Finally save
    torch.save(checkpoint, path)


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path)
    
    # Load only trainable parameters
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint['model_state_dict'].items() 
        if k in model_dict
    }
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']

def calculate_metrics(outputs, targets):
    # Apply sigmoid to convert logits to probabilities
    probs = torch.sigmoid(outputs)
    
    # Threshold to get binary predictions
    preds = (probs > 0.5).float()
    
    # Check for NaN or Inf values before IoU calculation
    if torch.isnan(preds).any() or torch.isinf(preds).any():
        print("Warning: NaN or Inf values in predictions")
        # Replace NaN/Inf with zeros
        preds = torch.nan_to_num(preds, nan=0.0, posinf=1.0, neginf=0.0)
    
    if torch.isnan(targets).any() or torch.isinf(targets).any():
        print("Warning: NaN or Inf values in targets")
        targets = torch.nan_to_num(targets, nan=0.0, posinf=1.0, neginf=0.0)
    
    # Ensure binary values
    targets = (targets > 0.5).float()
    
    # Initialize IoU metric
    iou = JaccardIndex(task="binary").to(preds.device)
    
    # Compute IoU safely with try-except
    try:
        iou_score = iou(preds.squeeze(1), targets.squeeze(1))
    except Exception as e:
        print(f"IoU calculation error: {e}")
        print(f"Preds shape: {preds.shape}, min: {preds.min()}, max: {preds.max()}")
        print(f"Targets shape: {targets.shape}, min: {targets.min()}, max: {targets.max()}")
        iou_score = torch.tensor(0.0, device=preds.device)
    
    return {'iou': iou_score}

def visualize_predictions(image, mask, pred, text, epoch, batch_idx, save_dir, sample_type=None):
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert tensors to numpy arrays
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    mask_np = mask.detach().cpu().numpy()
    
    # Ensure pred is the right shape and format
    if len(pred.shape) == 2:
        pred = pred.unsqueeze(0)  # Add batch dimension if needed
    
    # Convert prediction logits to probability map
    prob_map = torch.sigmoid(pred).detach().cpu().squeeze().numpy()
    
    # Binary prediction
    pred_binary = (prob_map > 0.5).astype(float)
    
    # Normalize image for visualization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image * std[None, None, :] + mean[None, None, :]
    image = np.clip(image, 0, 1)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Add text as figure title with type information
    type_info = f" [{sample_type}]" if sample_type else ""
    fig.suptitle(f'Expression{type_info}: "{text}"', wrap=True)
    
    # First row: Image and Ground Truth
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Image')
    axes[0, 1].imshow(mask_np, cmap='gray')
    axes[0, 1].set_title('Ground Truth Mask')
    
    # Second row: Probability map and Binary prediction
    axes[1, 0].imshow(image)
    prob_plot = axes[1, 0].imshow(prob_map, cmap='jet', alpha=0.7, vmin=0, vmax=1)
    axes[1, 0].set_title('Probability Map')
    fig.colorbar(prob_plot, ax=axes[1, 0])
    
    axes[1, 1].imshow(image)
    axes[1, 1].imshow(pred_binary, cmap='gray', alpha=0.7)
    axes[1, 1].set_title('Binary Prediction')
    
    # Turn off axes
    for row in axes:
        for ax in row:
            ax.axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure
    plt.savefig(os.path.join(save_dir, f'epoch_{epoch}_batch_{batch_idx}.png'))
    plt.close()

def plot_losses(train_losses, val_losses, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

def log_epoch_metrics(vis_dir, epoch, avg_train_loss, avg_train_metrics, avg_val_loss, avg_val_metrics):
    """
    Log epoch metrics to the run_details.txt file
    """
    metrics_path = os.path.join(vis_dir, 'run_details.txt')
    
    # If it's the first epoch, add a header
    if epoch == 0:
        with open(metrics_path, 'a') as f:
            f.write("\n\n===== TRAINING PROGRESS =====\n")
            f.write("Epoch | Train Loss | Train IoU | Val Loss | Val IoU\n")
            f.write("-" * 60 + "\n")
    
    # Append the metrics for this epoch
    with open(metrics_path, 'a') as f:
        f.write(f"{epoch:5d} | {avg_train_loss:.6f} | {avg_train_metrics['iou']:.6f} | {avg_val_loss:.6f} | {avg_val_metrics['iou']:.6f}\n")

def train(
    model, 
    train_loader,
    val_loader,
    optimizer,
    device,
    scaler,
    num_epochs=5,
    checkpoint_dir='./models/clip_sam',
    vis_dir='./visualizations/clip_sam',
    resume=False,
    initial_lr=1e-4,
    power=0.9,
    weight_decay=0.01,
    max_grad_norm=1.0,
    grad_accum_steps=1,
    enable_domain_adaptation=False,
    domain_loss_weight=0.1,
    grl_lambda_schedule='linear',
    grl_max_lambda=1.0,
    save_mid_epoch_checkpoints=False,
    mid_epoch_intervals=4
):
    start_epoch = 0
    best_loss = float('inf')
    best_iou = 0.0
    
    # Initialize loss history
    train_losses = []
    val_losses = []
    
    # Calculate total iterations for the polynomial decay
    total_iters = len(train_loader) * num_epochs // grad_accum_steps  # Adjust for gradient accumulation
    current_iter = 0
    
    if resume:
        checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')
        if os.path.exists(checkpoint_path):
            start_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path)
            print(f"Resumed from epoch {start_epoch}")
            # Adjust current_iter if resuming
            current_iter = start_epoch * len(train_loader) // grad_accum_steps
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        epoch_metrics = {'iou': 0.0}
        
        print(f"\nStarting epoch {epoch}")
        
        # For gradient accumulation
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0
        
        # Calculate mid-epoch checkpoint intervals
        total_batches = len(train_loader)
        if save_mid_epoch_checkpoints and total_batches >= mid_epoch_intervals:
            checkpoint_intervals = [int(total_batches * (i + 1) / mid_epoch_intervals) - 1 
                                  for i in range(mid_epoch_intervals)]
            print(f"Mid-epoch checkpoints will be saved at batches: {[i+1 for i in checkpoint_intervals]}")
        else:
            checkpoint_intervals = []
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Handle variable batch returns (with or without domain labels)
            if enable_domain_adaptation:
                images, texts, masks, sample_types, domain_labels = batch_data
                domain_labels = domain_labels.to(device, non_blocking=True)
            else:
                images, texts, masks, sample_types = batch_data
                domain_labels = None
            
            # Update learning rate using polynomial decay
            if batch_idx % grad_accum_steps == 0:
                poly_lr = initial_lr * (1 - current_iter / total_iters) ** power
                for param_group in optimizer.param_groups:
                    param_group['lr'] = poly_lr
                    
                # Update gradient reversal lambda if using domain adaptation
                if enable_domain_adaptation:
                    grl_lambda = calculate_grl_lambda(
                        current_iter, total_iters, 
                        grl_lambda_schedule, grl_max_lambda
                    )
                    model.set_gradient_reversal_lambda(grl_lambda)
            
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda'):
                if enable_domain_adaptation and domain_labels is not None:
                    # Forward pass with domain labels
                    outputs, domain_logits = model(images, texts, domain_labels)
                    
                    # Compute segmentation loss
                    seg_loss = model.compute_loss(outputs, masks, lambda_ce=0.9)
                    
                    # Compute domain loss
                    domain_loss = model.compute_domain_loss(domain_logits, domain_labels)
                    
                    # Combined loss
                    loss = seg_loss + domain_loss_weight * domain_loss
                else:
                    # Standard forward pass
                    outputs = model(images, texts)
                    loss = model.compute_loss(outputs, masks, lambda_ce=0.9)
                    domain_loss = torch.tensor(0.0, device=device)
                
                # Scale the loss by the number of accumulation steps
                loss = loss / grad_accum_steps
            
            # Calculate metrics
            batch_metrics = calculate_metrics(outputs, masks)
            for k, v in batch_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0) + v / grad_accum_steps
            
            # Visualize first batch of each epoch
            if batch_idx == 0:
                visualize_predictions(
                    images[0], masks[0], outputs[0], texts[0],
                    epoch, batch_idx, vis_dir, sample_types[0]
                )
            
            # Backward pass with gradient accumulation
            scaler.scale(loss).backward()
            accumulated_loss += loss.item()
            
            # Only step and update optimizer after accumulating gradients
            if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Apply gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                # Step optimizer and update scaler
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                # Log the accumulated loss
                epoch_loss += accumulated_loss
                if enable_domain_adaptation:
                    print(f"Batch {batch_idx} - Loss: {accumulated_loss:.4f} - IoU: {batch_metrics['iou']:.4f} - LR: {poly_lr:.6f} - GRL λ: {grl_lambda:.3f}")
                else:
                    print(f"Batch {batch_idx} - Loss: {accumulated_loss:.4f} - IoU: {batch_metrics['iou']:.4f} - LR: {poly_lr:.6f}")
                accumulated_loss = 0
                
                # Increment iteration counter
                current_iter += 1
                
                # Save mid-epoch checkpoint if needed
                if batch_idx in checkpoint_intervals:
                    interval_pct = ((checkpoint_intervals.index(batch_idx) + 1) * 100) // mid_epoch_intervals
                    checkpoint_name = f'epoch_{epoch}_checkpoint_{interval_pct}pct.pt'
                    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
                    
                    # Calculate current average metrics for this portion of the epoch
                    current_avg_loss = epoch_loss * grad_accum_steps / (batch_idx + 1)
                    current_avg_metrics = {k: v * grad_accum_steps / (batch_idx + 1) for k, v in epoch_metrics.items()}
                    
                    save_checkpoint(model, optimizer, epoch, current_avg_loss, checkpoint_path)
                    print(f"💾 Saved mid-epoch checkpoint: {checkpoint_name} (Loss: {current_avg_loss:.4f}, IoU: {current_avg_metrics['iou']:.4f})")
                    
                    # Also update latest.pt to this mid-epoch checkpoint
                    latest_checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')
                    save_checkpoint(model, optimizer, epoch, current_avg_loss, latest_checkpoint_path)
        
        avg_loss = epoch_loss * grad_accum_steps / len(train_loader)
        avg_metrics = {k: v * grad_accum_steps / len(train_loader) for k, v in epoch_metrics.items()}
        
        # Validation
        model.eval()
        val_loss = 0
        val_metrics = {'iou': 0.0}
        
        print("\nValidating...")
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(val_loader):
                # Handle variable batch returns (with or without domain labels)
                if enable_domain_adaptation:
                    images, texts, masks, sample_types, domain_labels = batch_data
                else:
                    images, texts, masks, sample_types = batch_data
                
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda'):
                    # For validation, we don't use domain adaptation (no domain labels passed)
                    outputs = model(images, texts)
                    loss = model.compute_loss(outputs, masks, lambda_ce=0.9)
                
                val_loss += loss.item()
                
                # Calculate metrics
                batch_metrics = calculate_metrics(outputs, masks)
                for k, v in batch_metrics.items():
                    val_metrics[k] = val_metrics.get(k, 0) + v
                
                # Visualize first 5 batches
                if batch_idx == 0:  # First batch
                    for i in range(min(5, len(images))):  # Take up to 5 images
                        visualize_predictions(
                            images[i], masks[i], outputs[i], texts[i],
                            epoch, f"{batch_idx}_{i}", os.path.join(vis_dir, 'val'), sample_types[i]
                        )
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_metrics = {k: v / len(val_loader) for k, v in val_metrics.items()}
        
        # Store losses
        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        
        # Plot losses
        plot_losses(train_losses, val_losses, vis_dir)
        
        print(f"\nEpoch {epoch} completed:")
        print(f"Train - Loss: {avg_loss:.4f} - IoU: {avg_metrics['iou']:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f} - IoU: {avg_val_metrics['iou']:.4f}")
        
        # Log metrics to the run_details.txt file
        log_epoch_metrics(vis_dir, epoch, avg_loss, avg_metrics, avg_val_loss, avg_val_metrics)
        
        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')
        save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
        
        # Save best model based on IoU
        if avg_val_metrics['iou'] > best_iou:
            best_iou = avg_val_metrics['iou']
            best_checkpoint_path = os.path.join(checkpoint_dir, 'best.pt')
            save_checkpoint(model, optimizer, epoch, avg_loss, best_checkpoint_path)
            print(f"New best IoU: {best_iou:.4f}")

            # Also log the best model in the run details
            with open(os.path.join(vis_dir, 'run_details.txt'), 'a') as f:
                f.write(f"\n=== NEW BEST MODEL AT EPOCH {epoch} ===\n")
                f.write(f"IoU: {best_iou:.6f}\n")

class DomainBalancedSampler(torch.utils.data.Sampler):
    """
    Custom sampler to ensure each batch contains a balanced number of samples
    from each domain.
    """
    def __init__(self, data_source, batch_size, num_domains):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_domains = num_domains
        
        # Group indices by domain
        self.indices_by_domain = [[] for _ in range(num_domains)]
        for idx, obj in enumerate(data_source.objects):
            domain_label = obj.get('domain_label', 0)
            self.indices_by_domain[domain_label].append(idx)
            
        # Calculate samples per domain for each batch
        self.samples_per_domain = self.batch_size // self.num_domains
        
        # Total number of batches is determined by the largest domain to ensure all samples are seen
        self.num_batches = len(data_source) // self.batch_size
        
        print("\nDomainBalancedSampler initialized:")
        for i in range(num_domains):
            print(f"- Domain {i}: {len(self.indices_by_domain[i])} samples")
        print(f"- Batch size: {self.batch_size}")
        print(f"- Samples per domain per batch: {self.samples_per_domain}")
        print(f"- Total batches per epoch: {self.num_batches}\n")

    def __iter__(self):
        # Shuffle indices for each domain at the start of each epoch
        domain_iters = [iter(torch.randperm(len(indices)).tolist()) for indices in self.indices_by_domain]
        
        for _ in range(self.num_batches):
            batch_indices = []
            for domain_id in range(self.num_domains):
                for _ in range(self.samples_per_domain):
                    try:
                        # Get the next shuffled index
                        shuffled_idx = next(domain_iters[domain_id])
                        # Get the actual dataset index
                        batch_indices.append(self.indices_by_domain[domain_id][shuffled_idx])
                    except StopIteration:
                        # If a domain runs out of samples, reshuffle and start over
                        domain_iters[domain_id] = iter(torch.randperm(len(self.indices_by_domain[domain_id])).tolist())
                        shuffled_idx = next(domain_iters[domain_id])
                        batch_indices.append(self.indices_by_domain[domain_id][shuffled_idx])
            
            # Shuffle the final batch to mix domains
            random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches

def get_domain_from_filename(filename):
    """Determine domain based on annotation filename prefix"""
    filename = filename.upper()
    if filename.startswith('D'):
        return 1  # DeepGlobe
    elif filename.startswith('P'):
        return 0  # iSAID (P for patches)
    elif filename.startswith('L'):
        return 2  # LoveDA
    else:
        # Default to iSAID if can't determine
        print(f"Warning: Could not determine domain from filename {filename}, defaulting to iSAID (0)")
        return 0

def find_transformed_image(image_dir, base_filename):
    """
    Find a transformed version of the image if it exists.
    Checks for various transformation suffixes in priority order.
    
    Args:
        image_dir: Directory containing images
        base_filename: Original filename (e.g., 'L1840_patch_0.png')
    
    Returns:
        tuple: (found_path, used_filename) or (None, None) if no transformed version exists
    """
    if not base_filename:
        return None, None
    
    # Get base name and extension
    base_path, ext = os.path.splitext(base_filename)
    
    # Define transformation suffixes in priority order
    # Historic effects (_5) and domain transfers (_toD, _toP, _toL)
    transform_suffixes = ['_5', '_toD', '_toP', '_toL']
    
    for suffix in transform_suffixes:
        transformed_filename = f"{base_path}{suffix}{ext}"
        transformed_path = os.path.join(image_dir, transformed_filename)
        
        if os.path.exists(transformed_path):
            return transformed_path, transformed_filename
    
    # No transformed version found
    return None, None

class SimpleDataset:
    def __init__(self, dataset_root, split='train', input_size=512, use_historic=False, enable_domain_adaptation=False, unique_only=False, one_unique_per_obj=False, use_transformed=False, dataset_filter=None):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.use_historic = use_historic
        self.enable_domain_adaptation = enable_domain_adaptation
        self.unique_only = unique_only
        self.one_unique_per_obj = one_unique_per_obj
        self.use_transformed = use_transformed
        self.dataset_filter = dataset_filter
        
        # Set paths based on split
        self.ann_dir = os.path.join(dataset_root, split, 'annotations')
        self.image_dir = os.path.join(dataset_root, split, 'images')
        
        # Add transform to match model configuration
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Get list of XML files
        all_xml_files = [f for f in os.listdir(self.ann_dir) if f.endswith('.xml')]
        
        # Apply dataset filter if specified
        if self.dataset_filter:
            filtered_xml_files = []
            for xml_file in all_xml_files:
                domain_id = get_domain_from_filename(xml_file)
                if ((self.dataset_filter == 'isaid' and domain_id == 0) or
                    (self.dataset_filter == 'deepglobe' and domain_id == 1) or
                    (self.dataset_filter == 'loveda' and domain_id == 2)):
                    filtered_xml_files.append(xml_file)
            self.xml_files = filtered_xml_files
            print(f"\nFound {len(all_xml_files)} total XML files in {split} split")
            print(f"Filtered to {len(self.xml_files)} XML files for dataset: {self.dataset_filter}")
        else:
            self.xml_files = all_xml_files
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
            
            # Get image filename and determine domain from XML filename
            filename = root.find('filename').text
            domain_label = get_domain_from_filename(xml_file) if self.enable_domain_adaptation else None
            
            # Determine which image to use based on flags
            if self.use_transformed:
                # Try to find any transformed version first
                transformed_path, transformed_filename = find_transformed_image(self.image_dir, filename)
                if transformed_path:
                    image_path = transformed_path
                    display_filename = transformed_filename
                else:
                    # Fall back to original
                    image_path = os.path.join(self.image_dir, filename)
                    display_filename = filename
            elif self.use_historic:
                # Convert normal filename to historic filename (replace _0.png with _5.png)
                if filename.endswith('_0.png'):
                    historic_filename = filename.replace('_0.png', '_5.png')
                    historic_path = os.path.join(self.image_dir, historic_filename)
                    
                    # Use historic image if it exists, otherwise fall back to normal image
                    if os.path.exists(historic_path):
                        image_path = historic_path
                        display_filename = historic_filename
                    else:
                        # Fall back to normal image
                        image_path = os.path.join(self.image_dir, filename)
                        display_filename = filename
                else:
                    # For non-standard filenames, just use the original
                    image_path = os.path.join(self.image_dir, filename)
                    display_filename = filename
            else:
                image_path = os.path.join(self.image_dir, filename)
                display_filename = filename
            
            # Store image path only once
            if display_filename not in self.images:
                self.images[display_filename] = image_path
            
            # Create a mapping from object ID to object data for this image
            objects_by_id = {}
            
            # Get all objects with their expressions (individual objects)
            for obj in root.findall('object'):
                # Get object properties
                obj_id = obj.find('id').text if obj.find('id') is not None else None
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
                
                # Store object data by ID for group processing (if needed)
                if obj_id is not None:
                    objects_by_id[obj_id] = {
                        'segmentation': rle,
                        'category': name
                    }
                
                # Get expressions for individual objects
                expressions = []
                exp_elem = obj.find('expressions')
                if exp_elem is not None:
                    for exp in exp_elem.findall('expression'):
                        # Filter expressions based on unique_only flag
                        if self.unique_only:
                            # Only include expressions with type="unique"
                            exp_type = exp.get('type')
                            if exp_type == 'unique':
                                expressions.append(exp.text)
                        else:
                            # Include all expressions
                            expressions.append(exp.text)
                
                if expressions:
                    total_objects += 1
                    
                    # If one_unique_per_obj is flagged, take only the first expression
                    if self.unique_only and self.one_unique_per_obj:
                        expressions_to_add = expressions[:1]
                    else:
                        expressions_to_add = expressions
                        
                    total_expressions += len(expressions_to_add)
                    
                    # Add a sample for each expression
                    for expression in expressions_to_add:
                        obj_data = {
                            'image_filename': display_filename,
                            'segmentation': rle,
                            'expression': expression,
                            'category': name,
                            'type': 'individual'
                        }
                        if self.enable_domain_adaptation:
                            obj_data['domain_label'] = domain_label
                        self.objects.append(obj_data)
            
            # Process groups
            groups_elem = root.find('groups')
            if groups_elem is not None:
                for group in groups_elem.findall('group'):
                    # Get group properties with null checks
                    group_id_elem = group.find('id')
                    if group_id_elem is None:
                        continue  # Skip groups without ID
                    group_id = group_id_elem.text
                    
                    category_elem = group.find('category')
                    if category_elem is None:
                        continue  # Skip groups without category
                    category = category_elem.text
                    
                    # Get group expressions
                    group_expressions = []
                    exp_elem = group.find('expressions')
                    if exp_elem is not None:
                        for exp in exp_elem.findall('expression'):
                            # Filter expressions based on unique_only flag
                            if self.unique_only:
                                # Only include expressions with type="unique"
                                exp_type = exp.get('type')
                                if exp_type == 'unique':
                                    group_expressions.append(exp.text)
                            else:
                                # Include all expressions
                                group_expressions.append(exp.text)
                    
                    if not group_expressions:
                        continue  # Skip groups without expressions
                    
                    # Groups have direct segmentation data
                    seg_elem = group.find('segmentation')
                    if seg_elem is None or not seg_elem.text:
                        print(f"WARNING: Group {group_id} in {xml_file} has no segmentation data - this should not happen!")
                        continue  # Skip groups without segmentation
                    
                    try:
                        # Parse the segmentation dictionary
                        seg_dict = eval(seg_elem.text)
                        size = seg_dict['size']
                        counts = seg_dict['counts']
                        rle = {'size': size, 'counts': counts}
                        group_segmentation = rle
                    except Exception as e:
                        print(f"WARNING: Failed to parse segmentation for group {group_id} in {xml_file}: {e}")
                        continue
                    
                    total_groups += 1
                    
                    # If one_unique_per_obj is flagged, take only the first expression
                    if self.unique_only and self.one_unique_per_obj:
                        expressions_to_add = group_expressions[:1]
                    else:
                        expressions_to_add = group_expressions
                        
                    total_group_expressions += len(expressions_to_add)
                    
                    # Add a sample for each group expression
                    for expression in expressions_to_add:
                        obj_data = {
                            'image_filename': display_filename,
                            'segmentation': group_segmentation,  # Single segmentation for group
                            'expression': expression,
                            'category': category,
                            'type': 'group'
                        }
                        if self.enable_domain_adaptation:
                            obj_data['domain_label'] = domain_label
                        self.objects.append(obj_data)
        
        print(f"\nDataset statistics:")
        print(f"- Total patches: {len(self.xml_files)}")
        print(f"- Unique images: {len(self.images)}")
        print(f"- Total individual objects with expressions: {total_objects}")
        print(f"- Total individual expressions: {total_expressions}")
        print(f"- Total groups with expressions: {total_groups}")
        print(f"- Total group expressions: {total_group_expressions}")
        print(f"- Total samples created: {len(self.objects)}")
        if self.unique_only:
            print(f"- Expression filtering: UNIQUE ONLY (type='unique')")
            if self.one_unique_per_obj:
                print(f"- Expression count: ONE PER OBJECT/GROUP")
        else:
            print(f"- Expression filtering: ALL EXPRESSIONS")
        
        # Print domain distribution if domain adaptation is enabled
        if self.enable_domain_adaptation:
            domain_counts = {0: 0, 1: 0, 2: 0}  # iSAID, DeepGlobe, LoveDA
            for obj in self.objects:
                domain_label = obj.get('domain_label', 0)
                domain_counts[domain_label] += 1
            
            domain_names = {0: 'iSAID', 1: 'DeepGlobe', 2: 'LoveDA'}
            print(f"\nDomain distribution:")
            for domain_id, count in domain_counts.items():
                print(f"- {domain_names[domain_id]} (ID {domain_id}): {count} samples")
    
    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, idx):
        obj = self.objects[idx]
        
        # Load image using stored path
        image = Image.open(self.images[obj['image_filename']])
        
        # Convert to RGB (handles both color and grayscale images)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image = self.transform(image)
        
        # Handle mask creation based on type
        if obj['type'] == 'individual':
            # Single object mask
            binary_mask = mask_utils.decode(obj['segmentation'])
        else:  # group
            # Single group mask
            binary_mask = mask_utils.decode(obj['segmentation'])
        
        mask = torch.from_numpy(binary_mask).float()
        mask = T.Resize((self.input_size, self.input_size), antialias=True)(mask.unsqueeze(0)).squeeze(0)
        
        if self.enable_domain_adaptation:
            domain_label = obj.get('domain_label', 0)  # Default to 0 (iSAID) if not found
            return image, obj['expression'], mask, obj['type'], domain_label
        else:
            return image, obj['expression'], mask, obj['type']

def save_run_details(args, run_id, model_name, effective_batch_size, train_dataset, val_dataset, vis_dir):
    """
    Save details of the training run to a text file
    """
    os.makedirs(vis_dir, exist_ok=True)
    details_path = os.path.join(vis_dir, 'run_details.txt')
    
    with open(details_path, 'w') as f:
        f.write(f"===== TRAINING RUN DETAILS =====\n\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"===== MODEL CONFIGURATION =====\n")
        f.write(f"Model Name: {model_name}\n")
        f.write(f"Input Size: {args.input_size}x{args.input_size}\n")
        f.write(f"SigLIP Model: {args.siglip_model}\n")
        f.write(f"SAM Model: {args.sam_model}\n")
        f.write(f"Down Spatial Times: {args.down_spatial_times}\n")
        f.write(f"With Dense Features: {args.with_dense_feat}\n")
        
        if args.use_lora:
            f.write(f"Using LoRA: Yes\n")
            f.write(f"LoRA Rank: {args.lora_r}\n")
            f.write(f"LoRA Alpha: {args.lora_alpha}\n")
            f.write(f"LoRA Dropout: {args.lora_dropout}\n\n")
        else:
            f.write(f"Using LoRA: No\n\n")
        
        f.write(f"===== TRAINING PARAMETERS =====\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Base Batch Size: {args.batch_size}\n")
        f.write(f"Gradient Accumulation Steps: {args.grad_accum_steps}\n")
        f.write(f"Effective Batch Size: {effective_batch_size}\n")
        f.write(f"Learning Rate: {args.lr}\n")
        f.write(f"Weight Decay: {args.weight_decay}\n")
        f.write(f"Polynomial Decay Power: {args.poly_power}\n")
        f.write(f"GPU ID: {args.gpu_id}\n")
        f.write(f"Mid-Epoch Checkpointing: {args.save_mid_epoch_checkpoints}\n")
        if args.save_mid_epoch_checkpoints:
            f.write(f"Checkpoint Intervals: {args.mid_epoch_intervals}\n")
        
        if args.enable_domain_adaptation:
            f.write(f"\n===== DOMAIN ADAPTATION =====\n")
            f.write(f"Enabled: Yes\n")
            f.write(f"Number of Domains: {args.num_domains}\n")
            f.write(f"Domain Loss Weight: {args.domain_loss_weight}\n")
            f.write(f"GRL Lambda Schedule: {args.grl_lambda_schedule}\n")
            f.write(f"GRL Max Lambda: {args.grl_max_lambda}\n\n")
        else:
            f.write(f"\n===== DOMAIN ADAPTATION =====\n")
            f.write(f"Enabled: No\n\n")
        
        f.write(f"===== DATASET INFORMATION =====\n")
        f.write(f"Training Samples: {len(train_dataset)}\n")
        f.write(f"Validation Samples: {len(val_dataset)}\n")
        f.write(f"Using Historic Images: {args.use_historic}\n")
        f.write(f"Using Transformed Images: {args.use_transformed}\n")
        f.write(f"Train on Unique Expressions Only: {args.unique_only}\n")
        if args.unique_only and args.one_unique_per_obj:
            f.write(f"Train on One Unique Expression per Object: Yes\n")
        if args.dataset_filter:
            f.write(f"Dataset Filter: {args.dataset_filter} only\n")
        else:
            f.write(f"Dataset Filter: All datasets\n")
        f.write(f"Dataset Path: ./aeriald\n")
        f.write(f"Images Path: ./aeriald/patches\n")
        f.write(f"Annotations Path: ./aeriald/patches\n")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Validate arguments for resume functionality
    if args.resume and not args.model_name:
        print("Error: --model_name is required when using --resume")
        print("Please specify the model name to resume from (e.g., clip_sam_20241215_143022_epochs5_bs4x2_lr0.0001)")
        return
    
    print("Start")
    torch.cuda.set_device(args.gpu_id)
    device = torch.device(f'cuda:{args.gpu_id}')
    
    # Set dataset path to aeriald root directory
    dataset_path = "/cfs/home/u035679/datasets/aeriald"
    
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    
    print(f"Using device: {device}")
    
    # Calculate gradient accumulation steps if effective batch size is provided
    grad_accum_steps = args.grad_accum_steps
    if args.effective_batch_size is not None:
        grad_accum_steps = max(1, args.effective_batch_size // args.batch_size)
        print(f"Using gradient accumulation with {grad_accum_steps} steps to simulate batch size of {args.batch_size * grad_accum_steps}")
    
    effective_batch_size = args.batch_size * grad_accum_steps
    
    # Generate unique run ID with timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Configure LoRA if enabled
    lora_cfg = None
    if args.use_lora:
        # Define target modules for different components
        lora_cfg = {
            'clip_vision_encoder': {
                'r': args.lora_r,
                'lora_alpha': args.lora_alpha,
                'target_modules': ['q_proj', 'v_proj'],
                'lora_dropout': args.lora_dropout,
                'bias': 'none'
            },
            'clip_text_encoder': {
                'r': args.lora_r,
                'lora_alpha': args.lora_alpha,
                'target_modules': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                'lora_dropout': args.lora_dropout,
                'bias': 'none'
            }
        }
        print(f"Using LoRA with rank {args.lora_r}, alpha {args.lora_alpha}")
    
    # Initialize SigLIP+SAM model
    model = SigLipSamSegmentator(
        siglip_model_name=args.siglip_model,
        sam_model_name=args.sam_model,
        down_spatial_times=args.down_spatial_times,
        with_dense_feat=args.with_dense_feat,
        lora_cfg=lora_cfg,
        device=device,
        enable_domain_adaptation=args.enable_domain_adaptation,
        num_domains=args.num_domains
    ).to(device)
    
    # Get trainable parameters (those with requires_grad=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Initialize optimizer 
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler()
    
    # Create datasets using predefined splits
    train_dataset = SimpleDataset(
        dataset_root=dataset_path,
        split='train',
        input_size=args.input_size,
        use_historic=args.use_historic,
        enable_domain_adaptation=args.enable_domain_adaptation,
        unique_only=args.unique_only,
        one_unique_per_obj=args.one_unique_per_obj,
        use_transformed=args.use_transformed,
        dataset_filter=args.dataset_filter
    )
    
    val_dataset = SimpleDataset(
        dataset_root=dataset_path,
        split='val',
        input_size=args.input_size,
        use_historic=args.use_historic,
        enable_domain_adaptation=args.enable_domain_adaptation,
        unique_only=False,  # Always use all expressions for validation
        one_unique_per_obj=False,
        use_transformed=args.use_transformed,
        dataset_filter=args.dataset_filter
    )
    
    # Initialize train_loader with default settings first
    train_loader_kwargs = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': 4
    }

    # Use the custom balanced sampler if both flags are enabled
    if args.enable_domain_adaptation and args.balanced_batch_sampling:
        print("Using Domain Balanced Sampler for training.")
        # Ensure batch size is divisible by the number of domains
        if args.batch_size % args.num_domains != 0:
            print(f"Warning: Batch size ({args.batch_size}) is not perfectly divisible by the number of domains ({args.num_domains}).")
            print("This may result in slightly unbalanced batches.")

        balanced_sampler = DomainBalancedSampler(
            data_source=train_dataset,
            batch_size=args.batch_size,
            num_domains=args.num_domains
        )
        # When using a batch_sampler, 'batch_size', 'shuffle', 'sampler', and 'drop_last' must be None.
        train_loader_kwargs['batch_sampler'] = balanced_sampler
        # Remove keys that conflict with batch_sampler
        train_loader_kwargs.pop('batch_size', None)
    else:
        print("Using standard random shuffling for training.")
        train_loader_kwargs['shuffle'] = True

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_kwargs)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4
    )
    
    print(f"\nStarting training with {len(train_dataset)} train and {len(val_dataset)} val samples...")
    print(f"Gradient accumulation steps: {grad_accum_steps} (effective batch size: {effective_batch_size})")
    
    # Handle model naming and directory creation based on resume flag
    if args.resume:
        # Use existing model directory for resume
        original_model_name = args.model_name
        original_checkpoint_dir = os.path.join('./models', original_model_name)
        original_vis_dir = os.path.join('./visualizations', original_model_name)
        
        # Check if the original model directory exists
        if not os.path.exists(original_checkpoint_dir):
            print(f"Error: Model directory '{original_checkpoint_dir}' not found!")
            print(f"Available models in ./models/:")
            if os.path.exists('./models'):
                for item in os.listdir('./models'):
                    if os.path.isdir(os.path.join('./models', item)):
                        print(f"  - {item}")
            return
        
        # Check if checkpoint exists
        checkpoint_path = os.path.join(original_checkpoint_dir, 'latest.pt')
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint 'latest.pt' not found in {original_checkpoint_dir}")
            return
        
        # Create restart directory
        restart_model_name = f"{original_model_name}_restart_{run_id}_epochs{args.epochs}"
        model_name = restart_model_name
        checkpoint_dir = os.path.join('./models', restart_model_name)
        vis_dir = os.path.join('./visualizations', restart_model_name)
        
        print(f"Resuming from: {original_model_name}")
        print(f"Creating restart directory: {restart_model_name}")
        
        # Create new directories for restart
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Copy the checkpoint to the new directory as starting point
        shutil.copy2(checkpoint_path, os.path.join(checkpoint_dir, 'latest.pt'))
        
        # Copy run details from original directory to preserve history
        original_details_path = os.path.join(original_vis_dir, 'run_details.txt')
        if os.path.exists(original_details_path):
            shutil.copy2(original_details_path, os.path.join(vis_dir, 'run_details.txt'))
            # Add restart information
            with open(os.path.join(vis_dir, 'run_details.txt'), 'a') as f:
                f.write(f"\n\n===== RESTART INFORMATION =====\n")
                f.write(f"Restart Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Original Model: {original_model_name}\n")
                f.write(f"Restart Model: {restart_model_name}\n")
                f.write(f"Additional Epochs: {args.epochs}\n")
    else:
        # Create new model for fresh training
        model_name = f"clip_sam_{run_id}_epochs{args.epochs}_bs{args.batch_size}x{grad_accum_steps}_lr{args.lr}"
        # Use absolute paths to avoid any path resolution issues
        base_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint_dir = os.path.join(base_dir, 'models', model_name)
        vis_dir = os.path.join(base_dir, 'visualizations', model_name)
        
        # Create directories if they don't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        
        # Save run details
        save_run_details(args, run_id, model_name, effective_batch_size, train_dataset, val_dataset, vis_dir)
    
    train(
        model, 
        train_loader,
        val_loader,
        optimizer, 
        device,
        scaler,
        num_epochs=args.epochs,
        checkpoint_dir=checkpoint_dir,
        vis_dir=vis_dir,
        resume=args.resume,
        initial_lr=args.lr,
        power=args.poly_power,
        weight_decay=args.weight_decay,
        max_grad_norm=1.0,
        grad_accum_steps=grad_accum_steps,
        enable_domain_adaptation=args.enable_domain_adaptation,
        domain_loss_weight=args.domain_loss_weight,
        grl_lambda_schedule=args.grl_lambda_schedule,
        grl_max_lambda=args.grl_max_lambda,
        save_mid_epoch_checkpoints=args.save_mid_epoch_checkpoints,
        mid_epoch_intervals=args.mid_epoch_intervals
    )

if __name__ == '__main__':
    main() 