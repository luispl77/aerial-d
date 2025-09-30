import torch
import torch.nn as nn
import torch.optim as optim
from model import SigLipSamSegmentator
import os
import json
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
import cv2
import html
import sys


# Historic effect functions copied from test.py
def add_film_grain(image: np.ndarray, intensity: float = 0.3) -> np.ndarray:
    """Add film grain noise to simulate old photography."""
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.float32)
    noisy_image = image.astype(np.float32) + noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def adjust_contrast_gamma(image: np.ndarray, contrast: float = 0.8, gamma: float = 1.2) -> np.ndarray:
    """Adjust contrast and gamma to simulate old film characteristics."""
    # Apply gamma correction
    gamma_corrected = np.power(image / 255.0, gamma) * 255.0
    
    # Apply contrast adjustment
    mean_val = np.mean(gamma_corrected)
    contrasted = (gamma_corrected - mean_val) * contrast + mean_val
    
    return np.clip(contrasted, 0, 255).astype(np.uint8)

def apply_sepia(image: np.ndarray) -> np.ndarray:
    """Apply sepia filter using transformation matrix."""
    # Ensure image is 3-channel color
    if len(image.shape) == 2:
        # Convert grayscale to color first
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3 and image.shape[2] == 4:
        # Convert BGRA to BGR
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    # Create sepia filter
    sepia_filter = np.array([[0.272, 0.534, 0.131], 
                            [0.349, 0.686, 0.168], 
                            [0.393, 0.769, 0.189]])
    sepia_image = cv2.transform(image, sepia_filter)
    sepia_image = np.clip(sepia_image, 0, 255)  # Ensure valid range
    
    return sepia_image.astype(np.uint8)

def add_noise(image: np.ndarray) -> np.ndarray:
    """Add random noise to simulate old photography grain."""
    noise = np.random.randint(0, 50, image.shape, dtype='uint8')
    noisy_image = cv2.add(image, noise)
    return noisy_image

def apply_basic_bw_effect(image: np.ndarray) -> tuple:
    """Basic black and white conversion."""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    return gray, "Basic_BW"

def apply_bw_grain_effect(image: np.ndarray) -> tuple:
    """B&W with grain and contrast adjustment."""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Mild contrast adjustment
    adjusted = adjust_contrast_gamma(gray, contrast=0.85, gamma=1.1)
    
    # Light grain
    grainy = add_film_grain(adjusted, intensity=0.1)
    
    return grainy, "BW_Grain"

def apply_sepia_with_noise_effect(image: np.ndarray) -> tuple:
    """Apply sepia tone effect with noise for vintage look."""
    # Apply sepia effect - this should work on the original color image
    sepia_image = apply_sepia(image)
    
    # Add noise
    noisy_sepia = add_noise(sepia_image)
    
    return noisy_sepia, "Sepia_Noise"

def apply_random_historic_effect(image: np.ndarray) -> tuple:
    """Apply one of the three historic effects randomly with equal probability."""
    effects = [
        apply_basic_bw_effect,
        apply_bw_grain_effect,
        apply_sepia_with_noise_effect
    ]
    
    # Choose random effect
    chosen_effect = random.choice(effects)
    return chosen_effect(image)


def parse_args():
    parser = argparse.ArgumentParser(description='Train aerial segmentation model with SigLIP+SAM')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for AdamW')
    parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--input_size', type=int, default=384, help='Input size for images')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--model_name', type=str, help='Optional legacy flag for resume compatibility; prefer --custom_name to point at run folder')
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
    parser.add_argument('--use_historic', action='store_true', default=False, help='Use historic (BW) images instead of normal color images')
    parser.add_argument('--historic_percentage', type=float, default=20.0, help='Percentage of samples (0-100) to apply historic filters during training')
    
    
    # Mid-epoch checkpointing
    parser.add_argument('--save_mid_epoch_checkpoints', action='store_true', help='Save checkpoints at 25%, 50%, 75% of each epoch')
    parser.add_argument('--mid_epoch_intervals', type=int, default=4, help='Number of intervals per epoch to save checkpoints (default: 4 = quarters)')
    
    # Expression filtering
    parser.add_argument('--unique_only', action='store_true', help='Train only on unique expressions (type="unique" in XML). Validation still uses all expressions.')
    parser.add_argument('--original_only', action='store_true', help='Train only on original expressions (those with id attributes in XML). Validation still uses all expressions.')
    parser.add_argument('--enhanced_only', action='store_true', help='Train only on enhanced expressions (type="enhanced" in XML). Validation still uses all expressions.')
    parser.add_argument('--one_unique_per_obj', action='store_true', help='If set along with --unique_only, only one unique expression per object/group will be used for training.')
    
    
    # Dataset filtering
    parser.add_argument('--dataset_filter', type=str, choices=['isaid', 'loveda'], help='Train only on samples from a specific dataset (isaid or loveda)')
    
    # Custom folder naming
    parser.add_argument('--custom_name', type=str, help='Override the default auto-generated folder name for new runs or specify the existing folder when using --resume')
    
    # Multi-dataset training
    parser.add_argument('--use_all_datasets', action='store_true', help='Train on AerialD + 4 additional datasets (RRSISD, RefSegRS, NWPU, Urban1960). Expression filtering flags still apply to AerialD only.')
    parser.add_argument('--exclude_datasets', nargs='*', choices=['rrsisd', 'refsegrs', 'nwpu', 'urban1960'], default=None,
                        help='Datasets to exclude when --use_all_datasets is enabled')
    parser.add_argument('--rrsisd_root', type=str, default='../datagen/rrsisd', help='Root directory of the RRSISD dataset')
    parser.add_argument('--refsegrs_root', type=str, default='../datagen/refsegrs/RefSegRS', help='Root directory of the RefSegRS dataset')
    parser.add_argument('--nwpu_root', type=str, default='../datagen/NWPU-Refer', help='Root directory of the NWPU-Refer dataset')
    parser.add_argument('--urban1960_root', type=str, default='../datagen/Urban1960SatBench', help='Root directory of the Urban1960SatBench dataset')
    
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
    save_mid_epoch_checkpoints=False,
    mid_epoch_intervals=4
):
    start_epoch = 0
    best_loss = float('inf')
    best_iou = 0.0

    # Initialize loss history
    train_losses = []
    val_losses = []

    # Determine training bounds and scheduler state
    end_epoch = num_epochs
    total_epochs_planned = max(num_epochs, 1)
    current_iter = 0

    if resume:
        checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

        last_epoch, best_loss = load_checkpoint(model, optimizer, checkpoint_path)
        start_epoch = last_epoch + 1
        end_epoch = start_epoch + num_epochs
        total_epochs_planned = max(end_epoch, 1)
        current_iter = start_epoch * len(train_loader) // grad_accum_steps
        print(f"Resumed from epoch {last_epoch}. Continuing at epoch {start_epoch} for {num_epochs} additional epochs (target epoch {end_epoch - 1}).")
    else:
        end_epoch = num_epochs
        total_epochs_planned = max(end_epoch, 1)

    # Calculate total iterations for the polynomial decay relative to the overall plan
    total_iters = max(len(train_loader), 1) * total_epochs_planned // max(grad_accum_steps, 1)
    total_iters = max(total_iters, 1)

    if start_epoch >= end_epoch:
        print(f"Nothing to train: start_epoch ({start_epoch}) >= target end epoch ({end_epoch}).")
        return

    poly_lr = initial_lr

    for epoch in range(start_epoch, end_epoch):
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
            # Handle batch data
            images, texts, masks, sample_types = batch_data
            
            # Update learning rate using polynomial decay
            if batch_idx % grad_accum_steps == 0:
                progress = min(max(current_iter / total_iters, 0.0), 1.0)
                poly_lr = initial_lr * (1 - progress) ** power
                for param_group in optimizer.param_groups:
                    param_group['lr'] = poly_lr
            
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda'):
                # Forward pass
                outputs = model(images, texts)
                loss = model.compute_loss(outputs, masks, lambda_ce=0.9)
                
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
                # Handle batch data
                images, texts, masks, sample_types = batch_data
                
                images = images.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda'):
                    # Forward pass
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


def get_domain_from_filename(filename):
    """Determine domain based on annotation filename prefix for dataset filtering"""
    filename = filename.upper()
    if filename.startswith('P'):
        return 0  # iSAID (P for patches)
    elif filename.startswith('L'):
        return 2  # LoveDA
    else:
        # Default to iSAID if can't determine
        print(f"Warning: Could not determine domain from filename {filename}, defaulting to iSAID (0)")
        return 0


class SimpleDataset:
    def __init__(self, dataset_root, split='train', input_size=512, use_historic=False, unique_only=False, original_only=False, enhanced_only=False, one_unique_per_obj=False, dataset_filter=None, historic_percentage=20.0):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.use_historic = use_historic
        self.unique_only = unique_only
        self.original_only = original_only
        self.enhanced_only = enhanced_only
        self.one_unique_per_obj = one_unique_per_obj
        self.dataset_filter = dataset_filter
        self.historic_percentage = historic_percentage
        
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
            
            # Get image filename
            filename = root.find('filename').text
            
            # Always use original image - historic effects will be applied on-the-fly
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
                        # Filter expressions based on filtering flags
                        if self.unique_only:
                            # Only include expressions with type="unique"
                            exp_type = exp.get('type')
                            if exp_type == 'unique':
                                expressions.append(exp.text)
                        elif self.original_only:
                            # Only include expressions with id attributes (original expressions)
                            exp_id = exp.get('id')
                            if exp_id is not None:
                                expressions.append(exp.text)
                        elif self.enhanced_only:
                            # Only include expressions with type="enhanced"
                            exp_type = exp.get('type')
                            if exp_type == 'enhanced':
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
                            # Filter expressions based on filtering flags
                            if self.unique_only:
                                # Only include expressions with type="unique"
                                exp_type = exp.get('type')
                                if exp_type == 'unique':
                                    group_expressions.append(exp.text)
                            elif self.original_only:
                                # Only include expressions with id attributes (original expressions)
                                exp_id = exp.get('id')
                                if exp_id is not None:
                                    group_expressions.append(exp.text)
                            elif self.enhanced_only:
                                # Only include expressions with type="enhanced"
                                exp_type = exp.get('type')
                                if exp_type == 'enhanced':
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
        elif self.original_only:
            print(f"- Expression filtering: ORIGINAL ONLY (with id attributes)")
        elif self.enhanced_only:
            print(f"- Expression filtering: ENHANCED ONLY (type='enhanced')")
        else:
            print(f"- Expression filtering: ALL EXPRESSIONS")
        
    
    def __len__(self):
        return len(self.objects)
    
    def __getitem__(self, idx):
        obj = self.objects[idx]
        
        # Load image using stored path
        image = Image.open(self.images[obj['image_filename']])
        
        # Convert to RGB (handles both color and grayscale images)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply historic effects on-the-fly with specified percentage
        if random.random() < (self.historic_percentage / 100.0):
            # Convert PIL to numpy array for historic effects (OpenCV format - BGR)
            image_array = np.array(image)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Apply random historic effect
            transformed_image_bgr, effect_name = apply_random_historic_effect(image_bgr)
            
            # Handle different output formats from historic effects
            if len(transformed_image_bgr.shape) == 2:
                # Grayscale - convert back to RGB
                transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_GRAY2RGB)
            else:
                # Color - convert BGR back to RGB
                transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert back to PIL
            image = Image.fromarray(transformed_image_rgb)
        
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
        
        return image, obj['expression'], mask, obj['type']


# Additional dataset classes for multi-dataset training
class RRSISDDataset:
    def __init__(self, dataset_root, split='train', input_size=512, historic_percentage=20.0):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.historic_percentage = historic_percentage
        
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
        
        # Load JSON annotations (contains the correct RLE masks)
        json_path = os.path.join(dataset_root, 'instances.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at {json_path}")
            
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Create lookup dict from JSON for fast access
        self.json_annotations = {ann['image_id']: ann for ann in json_data['annotations']}
        
        # Get list of XML files for the specified split
        all_xml_files = [f for f in os.listdir(self.ann_dir) if f.endswith('.xml')]
        
        print(f"\nProcessing RRSISD {split} split from {len(all_xml_files)} XML files...")
        
        # Store samples with proper mask loading
        self.samples = []
        
        for xml_file in tqdm(all_xml_files, desc=f"Processing RRSISD {split} XML files"):
            # Extract image_id from XML filename (e.g., "00001.xml" -> 1)
            try:
                image_id = int(xml_file.replace('.xml', ''))
            except ValueError:
                continue
                
            xml_path = os.path.join(self.ann_dir, xml_file)
            
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Check if this XML is for the requested split
                xml_split = root.find('split').text
                if xml_split != split:
                    continue
                
                # Check if we have JSON annotation for this image
                if image_id not in self.json_annotations:
                    continue
                    
                ann = self.json_annotations[image_id]
                image_path = os.path.join(self.image_dir, f"{image_id:05d}.jpg")
                
                if not os.path.exists(image_path):
                    continue
                
                # Get JSON bbox coordinates [x1, y1, x2, y2]
                json_bbox = ann['bbox']
                
                # Find matching object in XML by exact bbox match
                matched_description = None
                for obj in root.findall('object'):
                    bbox = obj.find('bndbox')
                    desc = obj.find('description')
                    
                    if bbox is not None and desc is not None and desc.text:
                        # Get XML bbox coordinates
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)
                        xml_bbox = [xmin, ymin, xmax, ymax]
                        
                        # Check for exact match
                        if xml_bbox == json_bbox:
                            matched_description = desc.text
                            break
                
                if matched_description is None:
                    # Fallback to generic description
                    matched_description = "the object in the image"
                
                # Get mask from RLE in JSON (this is the correct mask!)
                rle = ann['segmentation'][0]
                
                # Add sample
                sample_data = {
                    'image_path': image_path,
                    'expression': matched_description,
                    'segmentation': rle,  # RLE from JSON, not XML
                    'image_id': image_id,
                    'bbox': json_bbox
                }
                self.samples.append(sample_data)
                
            except Exception as e:
                print(f"Warning: Error processing {xml_file}: {e}")
                continue
        
        print(f"RRSISD Dataset ({split} split):")
        print(f"- Total valid samples: {len(self.samples)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply historic effects on-the-fly with specified percentage
        if random.random() < (self.historic_percentage / 100.0):
            # Convert PIL to numpy array for historic effects (OpenCV format - BGR)
            image_array = np.array(image)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Apply random historic effect
            transformed_image_bgr, effect_name = apply_random_historic_effect(image_bgr)
            
            # Handle different output formats from historic effects
            if len(transformed_image_bgr.shape) == 2:
                # Grayscale - convert back to RGB
                transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_GRAY2RGB)
            else:
                # Color - convert BGR back to RGB
                transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert back to PIL
            image = Image.fromarray(transformed_image_rgb)
        
        image = self.transform(image)
        
        # Decode mask from RLE
        binary_mask = mask_utils.decode(sample['segmentation'])
        mask = torch.from_numpy(binary_mask).float()
        mask = T.Resize((self.input_size, self.input_size), antialias=True)(mask.unsqueeze(0)).squeeze(0)
        
        return image, sample['expression'], mask, 'individual'


class RefSegRSDataset:
    def __init__(self, dataset_root, split='train', input_size=512, historic_percentage=20.0):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.historic_percentage = historic_percentage
        
        # Set paths for RefSegRS structure
        self.images_dir = os.path.join(dataset_root, 'images')
        self.masks_dir = os.path.join(dataset_root, 'masks')
        
        # Add transform to match model configuration
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Load referring expressions from txt files
        self.expressions = self._load_expressions(f'output_phrase_{split}.txt')
        
        print(f"\nRefSegRS Dataset ({split} split):")
        print(f"- Total expressions: {len(self.expressions)}")
        
        # Convert to list of samples
        self.samples = []
        for expr_id, expr_data in self.expressions.items():
            image_id = expr_data['image_id']
            expression = expr_data['expression']
            
            # Check if both image and mask exist
            image_path = os.path.join(self.images_dir, f"{image_id}.tif")
            mask_path = os.path.join(self.masks_dir, f"{image_id}.tif")
            
            if os.path.exists(image_path) and os.path.exists(mask_path):
                sample_data = {
                    'expr_id': expr_id,
                    'image_id': image_id,
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'expression': expression
                }
                self.samples.append(sample_data)
        
        print(f"- Valid samples (with existing files): {len(self.samples)}")
    
    def _load_expressions(self, filename):
        """Load referring expressions from txt file."""
        expressions = {}
        txt_path = os.path.join(self.dataset_root, filename)
        
        if not os.path.exists(txt_path):
            print(f"Warning: {txt_path} not found")
            return expressions
            
        with open(txt_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line and ' ' in line:
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
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image with PIL to handle .tif format
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply historic effects on-the-fly with specified percentage
        if random.random() < (self.historic_percentage / 100.0):
            # Convert PIL to numpy array for historic effects (OpenCV format - BGR)
            image_array = np.array(image)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Apply random historic effect
            transformed_image_bgr, effect_name = apply_random_historic_effect(image_bgr)
            
            # Handle different output formats from historic effects
            if len(transformed_image_bgr.shape) == 2:
                # Grayscale - convert back to RGB
                transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_GRAY2RGB)
            else:
                # Color - convert BGR back to RGB
                transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert back to PIL
            image = Image.fromarray(transformed_image_rgb)
        
        image = self.transform(image)
        
        # Load mask with PIL to handle .tif format
        mask_pil = Image.open(sample['mask_path'])
        mask_array = np.array(mask_pil)
        
        # Convert to grayscale if needed (RefSegRS masks might be 3-channel)
        if len(mask_array.shape) == 3:
            mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)
        
        # Convert to binary (0 or 1)
        mask_array = (mask_array > 127).astype(np.uint8)
        
        # Convert to tensor and resize
        mask = torch.from_numpy(mask_array).float()
        mask = T.Resize((self.input_size, self.input_size), antialias=True)(mask.unsqueeze(0)).squeeze(0)
        
        return image, sample['expression'], mask, 'individual'


class NWPUDataset:
    def __init__(self, dataset_root, split='train', input_size=512, historic_percentage=20.0):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.historic_percentage = historic_percentage
        
        # Import the NWPUReferProcessor from our processing script
        sys.path.append('../datagen/utils')
        try:
            from process_nwpu_refer import NWPUReferProcessor
            
            # Initialize the processor
            self.processor = NWPUReferProcessor(dataset_root)
            
            # Add transform to match model configuration
            self.transform = T.Compose([
                T.Resize((input_size, input_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
            ])
            
            # Get dataset info
            self.dataset_info = self.processor.get_dataset_info()
            
            # Get total samples for the split
            total_samples = self.dataset_info[f'{split}_samples']
            
            # Create list of indices for the split
            self.sample_indices = list(range(total_samples))
            
            print(f"\nNWPU-Refer Dataset ({split} split):")
            print(f"- Total samples in split: {total_samples}")
            print(f"- Samples to process: {len(self.sample_indices)}")
            
        except ImportError as e:
            print(f"Warning: Could not import NWPUReferProcessor: {e}")
            print("NWPU dataset will be empty.")
            self.processor = None
            self.sample_indices = []
    
    def __len__(self):
        return len(self.sample_indices) if self.processor else 0
    
    def __getitem__(self, idx):
        if not self.processor:
            # Return dummy data if processor failed to load
            dummy_image = torch.zeros(3, self.input_size, self.input_size)
            dummy_mask = torch.zeros(self.input_size, self.input_size)
            return dummy_image, "error loading sample", dummy_mask, 'individual'
        
        # Get the actual sample index
        sample_idx = self.sample_indices[idx]
        
        # Get sample from processor
        sample = self.processor.get_sample(self.split, sample_idx)
        
        if sample is None:
            # If sample loading fails, create a dummy sample
            dummy_image = torch.zeros(3, self.input_size, self.input_size)
            dummy_mask = torch.zeros(self.input_size, self.input_size)
            return dummy_image, "error loading sample", dummy_mask, 'individual'
        
        # Convert image to PIL
        image_pil = Image.fromarray(sample['image']).convert('RGB')
        
        # Apply historic effects on-the-fly with specified percentage
        if random.random() < (self.historic_percentage / 100.0):
            # Convert PIL to numpy array for historic effects (OpenCV format - BGR)
            image_array = np.array(image_pil)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Apply random historic effect
            transformed_image_bgr, effect_name = apply_random_historic_effect(image_bgr)
            
            # Handle different output formats from historic effects
            if len(transformed_image_bgr.shape) == 2:
                # Grayscale - convert back to RGB
                transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_GRAY2RGB)
            else:
                # Color - convert BGR back to RGB
                transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert back to PIL
            image_pil = Image.fromarray(transformed_image_rgb)
        
        # Apply transform
        image = self.transform(image_pil)
        
        # Convert mask to tensor and resize
        mask = torch.from_numpy(sample['mask']).float()
        # Normalize mask to 0-1 range
        mask = mask / 255.0 if mask.max() > 1.0 else mask
        mask = T.Resize((self.input_size, self.input_size), antialias=True)(mask.unsqueeze(0)).squeeze(0)
        
        # Get expression
        expression = sample['expression']
        
        # Determine type based on number of objects
        sample_type = 'group' if sample['num_objects'] > 1 else 'individual'
        
        return image, expression, mask, sample_type


class Urban1960Dataset:
    def __init__(self, dataset_root, split='train', input_size=512, historic_percentage=20.0):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.historic_percentage = 0.0  # Always 0 - Urban1960 already contains historic imagery
        
        # Add transform to match model configuration
        self.transform = T.Compose([
            T.Resize((input_size, input_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
        ])
        
        # Class mappings for referring expressions
        self.class_expressions = {
            # From Urban1960SatISP (Impervious Surface Product)
            'ISP_0': "all natural areas and vegetation in the image",
            'ISP_1': "all building and road areas in the image",
            # From Urban1960SatSS (Semantic Segmentation) - skip background class 0
            'SS_1': "all building areas in the image", 
            'SS_2': "all roads in the image",
            'SS_3': "all water in the image"
        }
        
        # Process both ISP and SS subdatasets
        self.samples = []
        
        # Process ISP dataset
        isp_images_dir = os.path.join(dataset_root, 'Urban1960SatISP', 'image')
        isp_masks_dir = os.path.join(dataset_root, 'Urban1960SatISP', 'mask_gt_ISP')
        isp_split_file = os.path.join(dataset_root, 'Urban1960SatISP', f'labelled_{split}.txt')
        
        if os.path.exists(isp_split_file):
            with open(isp_split_file, 'r') as f:
                isp_image_names = [line.strip() for line in f if line.strip()]
            
            print(f"\nUrban1960 Dataset ({split} split):")
            print(f"- ISP images in split: {len(isp_image_names)}")
            
            for image_name in isp_image_names:
                image_path = os.path.join(isp_images_dir, f"{image_name}.png")
                mask_path = os.path.join(isp_masks_dir, f"{image_name}.png")
                
                if not os.path.exists(image_path) or not os.path.exists(mask_path):
                    continue
                
                # Load mask to determine which classes are present
                try:
                    mask_pil = Image.open(mask_path)
                    mask_array = np.array(mask_pil)
                    
                    # Get unique class values in this mask (ISP has classes 0 and 1)
                    unique_classes = np.unique(mask_array)
                    for class_id in unique_classes:
                        if class_id in [0, 1]:  # ISP classes
                            sample_data = {
                                'image_name': image_name,
                                'image_path': image_path,
                                'mask_path': mask_path,
                                'class_id': class_id,
                                'dataset_type': 'ISP',
                                'expression': self.class_expressions[f'ISP_{class_id}']
                            }
                            self.samples.append(sample_data)
                            
                except Exception as e:
                    print(f"Warning: Error processing ISP {image_name}: {e}")
                    continue
        
        # Process SS dataset
        ss_images_dir = os.path.join(dataset_root, 'Urban1960SatSS', 'image')
        ss_masks_dir = os.path.join(dataset_root, 'Urban1960SatSS', 'mask_gt')
        ss_split_file = os.path.join(dataset_root, 'Urban1960SatSS', f'labelled_{split}.txt')
        
        if os.path.exists(ss_split_file):
            with open(ss_split_file, 'r') as f:
                ss_image_names = [line.strip() for line in f if line.strip()]
            
            print(f"- SS images in split: {len(ss_image_names)}")
            
            for image_name in ss_image_names:
                image_path = os.path.join(ss_images_dir, f"{image_name}.png")
                mask_path = os.path.join(ss_masks_dir, f"{image_name}.png")
                
                if not os.path.exists(image_path) or not os.path.exists(mask_path):
                    continue
                
                # Load mask to determine which classes are present
                try:
                    mask_pil = Image.open(mask_path)
                    mask_array = np.array(mask_pil)
                    
                    # Get unique class values in this mask (SS classes 1-3, skip background class 0)
                    unique_classes = np.unique(mask_array)
                    for class_id in unique_classes:
                        if class_id in [1, 2, 3]:  # SS classes, skip background (0)
                            sample_data = {
                                'image_name': image_name,
                                'image_path': image_path,
                                'mask_path': mask_path,
                                'class_id': class_id,
                                'dataset_type': 'SS',
                                'expression': self.class_expressions[f'SS_{class_id}']
                            }
                            self.samples.append(sample_data)
                            
                except Exception as e:
                    print(f"Warning: Error processing SS {image_name}: {e}")
                    continue
        
        print(f"- Total samples (class instances): {len(self.samples)}")
        
        # Show class distribution
        class_counts = {}
        isp_total = 0
        ss_total = 0
        for sample in self.samples:
            cls_key = f"{sample['dataset_type']}_{sample['class_id']}"
            class_counts[cls_key] = class_counts.get(cls_key, 0) + 1
            if sample['dataset_type'] == 'ISP':
                isp_total += 1
            elif sample['dataset_type'] == 'SS':
                ss_total += 1
        
        print("- Dataset type distribution:")
        print(f"  ISP samples: {isp_total}")
        print(f"  SS samples: {ss_total}")
        print("- Class distribution:")
        for cls_key in sorted(class_counts.keys()):
            expression = self.class_expressions[cls_key]
            print(f"  {cls_key}: {class_counts[cls_key]} samples - '{expression}'")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = Image.open(sample['image_path']).convert('RGB')
        
        # Apply historic effects on-the-fly with specified percentage
        if random.random() < (self.historic_percentage / 100.0):
            # Convert PIL to numpy array for historic effects (OpenCV format - BGR)
            image_array = np.array(image)
            image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
            
            # Apply random historic effect
            transformed_image_bgr, effect_name = apply_random_historic_effect(image_bgr)
            
            # Handle different output formats from historic effects
            if len(transformed_image_bgr.shape) == 2:
                # Grayscale - convert back to RGB
                transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_GRAY2RGB)
            else:
                # Color - convert BGR back to RGB
                transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_BGR2RGB)
            
            # Convert back to PIL
            image = Image.fromarray(transformed_image_rgb)
        
        image = self.transform(image)
        
        # Load mask and create binary mask for this specific class
        mask_pil = Image.open(sample['mask_path'])
        mask_array = np.array(mask_pil)
        
        # Create binary mask for the specific class
        class_id = sample['class_id']
        binary_mask = (mask_array == class_id).astype(np.uint8)
        
        # Convert to tensor and resize
        mask = torch.from_numpy(binary_mask).float()
        mask = T.Resize((self.input_size, self.input_size), antialias=True)(mask.unsqueeze(0)).squeeze(0)
        
        return image, sample['expression'], mask, 'individual'


class CombinedDataset:
    """Combined dataset that merges AerialD with additional datasets."""
    def __init__(self, dataset_root, split='train', input_size=512, use_historic=False, 
                 unique_only=False, original_only=False, enhanced_only=False, 
                 one_unique_per_obj=False, dataset_filter=None, 
                 additional_datasets=None, historic_percentage=20.0):
        
        self.datasets = []
        self.dataset_names = []
        self.cumulative_lengths = []
        
        # Always add AerialD dataset first
        print("="*60)
        print("LOADING AERIALD DATASET")
        print("="*60)
        
        aeriald_dataset = SimpleDataset(
            dataset_root=dataset_root,
            split=split,
            input_size=input_size,
            use_historic=use_historic,
            unique_only=unique_only,
            original_only=original_only,
            enhanced_only=enhanced_only,
            one_unique_per_obj=one_unique_per_obj,
            dataset_filter=dataset_filter,
            historic_percentage=historic_percentage
        )
        
        self.datasets.append(aeriald_dataset)
        self.dataset_names.append("AerialD")
        current_length = len(aeriald_dataset)
        self.cumulative_lengths.append(current_length)
        
        # Add additional datasets if provided
        if additional_datasets:
            for dataset_config in additional_datasets:
                dataset_type = dataset_config['type']
                dataset_root = dataset_config['root']
                
                print("="*60)
                print(f"LOADING {dataset_type.upper()} DATASET")
                print("="*60)
                
                try:
                    if dataset_type == 'rrsisd':
                        dataset = RRSISDDataset(
                            dataset_root=dataset_root,
                            split=split,
                            input_size=input_size,
                            historic_percentage=historic_percentage
                        )
                    elif dataset_type == 'refsegrs':
                        dataset = RefSegRSDataset(
                            dataset_root=dataset_root,
                            split=split,
                            input_size=input_size,
                            historic_percentage=historic_percentage
                        )
                    elif dataset_type == 'nwpu':
                        dataset = NWPUDataset(
                            dataset_root=dataset_root,
                            split=split,
                            input_size=input_size,
                            historic_percentage=historic_percentage
                        )
                    elif dataset_type == 'urban1960':
                        dataset = Urban1960Dataset(
                            dataset_root=dataset_root,
                            split=split,
                            input_size=input_size,
                            historic_percentage=historic_percentage
                        )
                    else:
                        print(f"Warning: Unknown dataset type {dataset_type}, skipping...")
                        continue
                    
                    self.datasets.append(dataset)
                    self.dataset_names.append(dataset_type.upper())
                    current_length += len(dataset)
                    self.cumulative_lengths.append(current_length)
                    
                except Exception as e:
                    print(f"Warning: Failed to load {dataset_type} dataset: {e}")
                    print("Continuing with other datasets...")
                    continue
        
        # Print summary
        print("="*60)
        print("COMBINED DATASET SUMMARY")
        print("="*60)
        total_samples = 0
        for i, (name, dataset) in enumerate(zip(self.dataset_names, self.datasets)):
            dataset_length = len(dataset)
            total_samples += dataset_length
            print(f"- {name}: {dataset_length:,} samples")
        
        print(f"\nTotal combined samples: {total_samples:,}")
        print("="*60)
    
    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
    
    def __getitem__(self, idx):
        # Find which dataset this index belongs to
        for i, cum_len in enumerate(self.cumulative_lengths):
            if idx < cum_len:
                # Calculate the local index within the dataset
                local_idx = idx - (self.cumulative_lengths[i-1] if i > 0 else 0)
                return self.datasets[i][local_idx]
        
        # Should never reach here
        raise IndexError(f"Index {idx} out of range for combined dataset of length {len(self)}")

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
        
        
        f.write(f"===== DATASET INFORMATION =====\n")
        f.write(f"Training Samples: {len(train_dataset)}\n")
        f.write(f"Validation Samples: {len(val_dataset)}\n")
        f.write(f"Using Historic Images: {args.use_historic}\n")
        f.write(f"Train on Unique Expressions Only: {args.unique_only}\n")
        f.write(f"Train on Original Expressions Only: {args.original_only}\n")
        f.write(f"Train on Enhanced Expressions Only: {args.enhanced_only}\n")
        if args.unique_only and args.one_unique_per_obj:
            f.write(f"Train on One Unique Expression per Object: Yes\n")
        if args.dataset_filter:
            f.write(f"Dataset Filter: {args.dataset_filter} only\n")
        else:
            f.write(f"Dataset Filter: All datasets\n")
        f.write(f"\n===== MULTI-DATASET INFORMATION =====\n")
        f.write(f"Multi-Dataset Training: {args.use_all_datasets}\n")
        if args.use_all_datasets:
            f.write(f"Datasets Used:\n")
            f.write(f"  - AerialD (Primary): {args.dataset_root if hasattr(args, 'dataset_root') else './aeriald'}\n")
            f.write(f"  - RRSISD: {args.rrsisd_root}\n")
            f.write(f"  - RefSegRS: {args.refsegrs_root}\n")
            f.write(f"  - NWPU: {args.nwpu_root}\n")
            f.write(f"  - Urban1960: {args.urban1960_root}\n")
            f.write(f"Expression Filtering: Applied to AerialD only\n")
            f.write(f"Historic Effects: {args.historic_percentage}% of AerialD, RRSISD, RefSegRS, and NWPU samples (Urban1960 excluded - already historic)\n")
        else:
            f.write(f"Single Dataset Mode - AerialD Only\n")
            f.write(f"Historic Effects: {args.historic_percentage}% of AerialD samples\n")
        f.write(f"Dataset Path: ./aeriald\n")
        f.write(f"Images Path: ./aeriald/patches\n")
        f.write(f"Annotations Path: ./aeriald/patches\n")

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Validate expression filtering arguments are mutually exclusive
    expression_filters = [args.unique_only, args.original_only, args.enhanced_only]
    if sum(expression_filters) > 1:
        print("Error: Only one expression filtering option can be used at a time.")
        print("Choose between --unique_only, --original_only, or --enhanced_only.")
        return
    
    # Validate arguments for resume functionality
    if args.resume and not (args.custom_name or args.model_name):
        print("Error: --custom_name (preferred) or --model_name must be provided when using --resume")
        print("Please point to the existing run folder that owns latest.pt")
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

    # Resolve base directories once so both training and resume share them
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_root = os.path.join(base_dir, 'models')
    vis_root = os.path.join(base_dir, 'visualizations')
    
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
        device=device
    ).to(device)
    
    # Get trainable parameters (those with requires_grad=True)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    
    # Initialize optimizer 
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler()
    
    # Create datasets using predefined splits
    if args.use_all_datasets:
        print("\n" + "="*80)
        print("MULTI-DATASET TRAINING MODE ENABLED")
        print("Training on AerialD + 4 additional datasets")
        print("Expression filtering flags apply to AerialD only")
        print("="*80)
        
        
        # Configure additional datasets
        excluded = set(args.exclude_datasets or [])
        if excluded:
            print(f"Datasets excluded from multi-dataset training: {', '.join(sorted(excluded)).upper()}")
        additional_datasets = [
            {'type': dataset_type, 'root': dataset_root}
            for dataset_type, dataset_root in [
                ('rrsisd', args.rrsisd_root),
                ('refsegrs', args.refsegrs_root),
                ('nwpu', args.nwpu_root),
                ('urban1960', args.urban1960_root)
            ]
            if dataset_type not in excluded
        ]
        
        train_dataset = CombinedDataset(
            dataset_root=dataset_path,
            split='train',
            input_size=args.input_size,
            use_historic=args.use_historic,
            unique_only=args.unique_only,
            original_only=args.original_only,
            enhanced_only=args.enhanced_only,
            one_unique_per_obj=args.one_unique_per_obj,
            dataset_filter=args.dataset_filter,
            additional_datasets=additional_datasets,
            historic_percentage=args.historic_percentage
        )
        
        val_dataset = CombinedDataset(
            dataset_root=dataset_path,
            split='val',
            input_size=args.input_size,
            use_historic=args.use_historic,
            unique_only=False,  # Always use all expressions for validation
            one_unique_per_obj=False,
            dataset_filter=args.dataset_filter,
            additional_datasets=additional_datasets,
            historic_percentage=0.0  # No historic effects for validation
        )
    else:
        print("\nSingle dataset mode - using AerialD only")
        
        train_dataset = SimpleDataset(
            dataset_root=dataset_path,
            split='train',
            input_size=args.input_size,
            use_historic=args.use_historic,
            unique_only=args.unique_only,
            original_only=args.original_only,
            enhanced_only=args.enhanced_only,
            one_unique_per_obj=args.one_unique_per_obj,
            dataset_filter=args.dataset_filter,
            historic_percentage=args.historic_percentage
        )
        
        val_dataset = SimpleDataset(
            dataset_root=dataset_path,
            split='val',
            input_size=args.input_size,
            use_historic=args.use_historic,
            unique_only=False,  # Always use all expressions for validation
            one_unique_per_obj=False,
            dataset_filter=args.dataset_filter,
            historic_percentage=0.0  # No historic effects for validation
        )
    
    # Initialize train_loader with default settings first
    train_loader_kwargs = {
        'batch_size': args.batch_size,
        'pin_memory': True,
        'num_workers': 4
    }

    # Use standard random shuffling for training
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
        # Determine which folder to resume from (prefer custom_name for clarity)
        resume_folder = args.custom_name or args.model_name
        model_name = resume_folder
        checkpoint_dir = os.path.join(models_root, resume_folder)
        vis_dir = os.path.join(vis_root, resume_folder)

        # Check if the original model directory exists
        if not os.path.isdir(checkpoint_dir):
            print(f"Error: Model directory '{checkpoint_dir}' not found!")
            if os.path.isdir(models_root):
                print("Available models:")
                for item in sorted(os.listdir(models_root)):
                    if os.path.isdir(os.path.join(models_root, item)):
                        print(f"  - {item}")
            return

        # Check if checkpoint exists
        checkpoint_path = os.path.join(checkpoint_dir, 'latest.pt')
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint 'latest.pt' not found in {checkpoint_dir}")
            return

        print(f"Resuming from existing run folder: {model_name}")

        # Append resume information to run details so history stays in one file
        details_path = os.path.join(vis_dir, 'run_details.txt')
        os.makedirs(vis_dir, exist_ok=True)
        details_exists = os.path.exists(details_path)
        with open(details_path, 'a') as f:
            if not details_exists:
                f.write("===== TRAINING RUN DETAILS =====\n")
            f.write(f"\n\n===== RESUME INFORMATION =====\n")
            f.write(f"Resume Date and Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Requested Additional Epochs: {args.epochs}\n")
    else:
        # Create new model for fresh training
        if args.custom_name:
            model_name = args.custom_name
            print(f"Using custom folder name: {model_name}")
        else:
            model_name = f"clip_sam_{run_id}_epochs{args.epochs}_bs{args.batch_size}x{grad_accum_steps}_lr{args.lr}"
            print(f"Using auto-generated folder name: {model_name}")
            
        # Use absolute paths to avoid any path resolution issues
        checkpoint_dir = os.path.join(models_root, model_name)
        vis_dir = os.path.join(vis_root, model_name)
        
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
        save_mid_epoch_checkpoints=args.save_mid_epoch_checkpoints,
        mid_epoch_intervals=args.mid_epoch_intervals
    )

if __name__ == '__main__':
    main() 
