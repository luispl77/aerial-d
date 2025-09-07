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
import cv2
import html

from model import SigLipSamSegmentator

# Historic transformation functions from pipeline step 7
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
    parser.add_argument('--dataset_type', type=str, choices=['aeriald', 'rrsisd', 'refsegrs', 'nwpu', 'urban1960'], default='aeriald', help='Type of dataset to use for testing')
    parser.add_argument('--rrsisd_root', type=str, default='../datagen/rrsisd', help='Root directory of the RRSISD dataset')
    parser.add_argument('--refsegrs_root', type=str, default='../datagen/refsegrs/RefSegRS', help='Root directory of the RefSegRS dataset')
    parser.add_argument('--nwpu_root', type=str, default='../datagen/NWPU-Refer', help='Root directory of the NWPU-Refer dataset')
    parser.add_argument('--urban1960_root', type=str, default='../datagen/Urban1960SatBench', help='Root directory of the Urban1960SatBench dataset')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for patch selection')
    parser.add_argument('--historic', action='store_true', help='Apply historic transformations (Basic B&W, B&W + Grain, or Sepia + Noise) randomly to images before testing')
    
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

def visualize_predictions(image, mask, pred, text, save_path, transformed_image=None, effect_name=None):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Always process the original image that the model saw (denormalized)
    original_image = image.detach().cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_image = original_image * std[None, None, :] + mean[None, None, :]
    original_image = np.clip(original_image, 0, 1)
    
    # Use transformed image for the first subplot if available, otherwise use original
    if transformed_image is not None:
        # Use the transformed image (it's already in RGB format from dataset)
        # Need to resize it to match the model input size (same as mask/predictions)
        from PIL import Image as PILImage
        transformed_pil = PILImage.fromarray(transformed_image)
        # Get the size from the original image (which is already at model input size)
        target_size = (original_image.shape[1], original_image.shape[0])  # (width, height)
        transformed_resized = transformed_pil.resize(target_size, PILImage.Resampling.LANCZOS)
        display_image = np.array(transformed_resized).astype(np.float32) / 255.0
        display_image = np.clip(display_image, 0, 1)
    else:
        display_image = original_image
    
    mask = (mask > 0.5).float().detach().cpu().numpy()
    
    # Convert logits to probabilities
    prob = torch.sigmoid(pred).detach().cpu().numpy()
    
    # Binary prediction with threshold 0.5
    pred_binary = (prob > 0.5).astype(float)
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Add text as figure title, including effect name if available
    title = f'Expression: "{text}"'
    if effect_name:
        title += f' | Historic Effect: {effect_name}'
    fig.suptitle(title, wrap=True)
    
    # Plot image, ground truth, probability map, and binary prediction
    axes[0].imshow(display_image)
    axes[0].set_title('Image')
    
    # Overlay ground truth mask on image
    axes[1].imshow(display_image)
    axes[1].imshow(mask, cmap='Reds', alpha=0.5)
    axes[1].set_title('Ground Truth Overlay')
    
    # Plot probability map
    prob_plot = axes[2].imshow(prob, cmap='jet', vmin=0, vmax=1)
    axes[2].set_title(f'Probability Map\nmin={prob.min():.3f}, max={prob.max():.3f}')
    fig.colorbar(prob_plot, ax=axes[2])
    
    # Overlay prediction on image
    axes[3].imshow(display_image)
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
    def __init__(self, dataset_root, split='val', input_size=480, max_samples=None, seed=42, historic=False):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.max_samples = max_samples
        self.historic = historic
        
        # Set random seed
        random.seed(seed)
        
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
                            group_expressions.append(exp.text)
                    
                    if not group_expressions:
                        continue  # Skip groups without expressions
                    
                    # Groups have direct segmentation data
                    seg_elem = group.find('segmentation')
                    if seg_elem is None or not seg_elem.text:
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
                    total_group_expressions += len(group_expressions)
                    
                    # Add a sample for each group expression
                    for expression in group_expressions:
                        self.objects.append({
                            'image_filename': filename,
                            'segmentation': group_segmentation,  # Single segmentation for group
                            'expression': expression,
                            'category': category,
                            'type': 'group'
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
        image_pil = Image.open(self.images[obj['image_filename']]).convert('RGB')
        
        # Apply historic transformations if enabled
        transformed_image = None
        effect_name = None
        if self.historic:
            # Convert PIL to numpy array (RGB format)
            original_image_array = np.array(image_pil)
            # Apply random historic effect to numpy array (in BGR format for OpenCV functions)
            bgr_image = cv2.cvtColor(original_image_array, cv2.COLOR_RGB2BGR)
            transformed_image_bgr, effect_name = apply_random_historic_effect(bgr_image)
            
            # Convert back to RGB for PIL processing
            transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_BGR2RGB)
            transformed_image = transformed_image_rgb.copy()  # Store for visualization
            
            # Use transformed image for model input
            image_pil = Image.fromarray(transformed_image_rgb)
        
        image = self.transform(image_pil)
        
        # Handle mask creation based on type
        if obj['type'] == 'individual':
            # Single object mask
            binary_mask = mask_utils.decode(obj['segmentation'])
        else:  # group
            # Single group mask (already combined in the dataset)
            binary_mask = mask_utils.decode(obj['segmentation'])
        
        mask = torch.from_numpy(binary_mask).float()
        mask = T.Resize((self.input_size, self.input_size), antialias=True)(mask.unsqueeze(0)).squeeze(0)
        
        return image, obj['expression'], mask, obj['image_filename'], obj['type'], transformed_image, effect_name


class RRSISDDataset:
    def __init__(self, dataset_root, split='val', input_size=480, max_samples=None, seed=42, historic=False):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.max_samples = max_samples
        self.historic = historic
        
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
        
        # Load JSON annotations (contains the correct RLE masks)
        json_path = os.path.join(dataset_root, 'instances.json')
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"JSON file not found at {json_path}")
            
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        # Create lookup dict from JSON for fast access
        self.json_annotations = {ann['image_id']: ann for ann in json_data['annotations']}
        
        # Get list of XML files and sample them first (like the old approach)
        all_xml_files = [f for f in os.listdir(self.ann_dir) if f.endswith('.xml')]
        
        # Pre-sample XML files for efficiency
        if self.max_samples is not None and len(all_xml_files) > self.max_samples * 2:
            random.shuffle(all_xml_files)
            xml_files_to_process = all_xml_files[:self.max_samples * 5]  # 5x buffer
        else:
            xml_files_to_process = all_xml_files
        
        print(f"\nProcessing RRSISD {split} split from {len(xml_files_to_process)} sampled XML files...")
        
        # Store samples with proper mask loading
        self.samples = []
        
        for xml_file in tqdm(xml_files_to_process, desc=f"Processing RRSISD {split} XML files"):
            if self.max_samples is not None and len(self.samples) >= self.max_samples:
                break
                
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
                self.samples.append({
                    'image_path': image_path,
                    'expression': matched_description,
                    'segmentation': rle,  # RLE from JSON, not XML
                    'image_id': image_id,
                    'bbox': json_bbox
                })
                
            except Exception as e:
                print(f"Warning: Error processing {xml_file}: {e}")
                continue
            if self.max_samples is not None and len(self.samples) >= self.max_samples:
                break
        
        # Final sampling if we still have too many
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
        image_pil = Image.open(sample['image_path']).convert('RGB')
        
        # Apply historic transformations if enabled
        transformed_image = None
        effect_name = None
        if self.historic:
            # Convert PIL to numpy array (RGB format)
            original_image_array = np.array(image_pil)
            # Apply random historic effect to numpy array (in BGR format for OpenCV functions)
            bgr_image = cv2.cvtColor(original_image_array, cv2.COLOR_RGB2BGR)
            transformed_image_bgr, effect_name = apply_random_historic_effect(bgr_image)
            
            # Convert back to RGB for PIL processing
            transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_BGR2RGB)
            transformed_image = transformed_image_rgb.copy()  # Store for visualization
            
            # Use transformed image for model input
            image_pil = Image.fromarray(transformed_image_rgb)
        
        image = self.transform(image_pil)
        
        # Decode mask from RLE
        binary_mask = mask_utils.decode(sample['segmentation'])
        mask = torch.from_numpy(binary_mask).float()
        mask = T.Resize((self.input_size, self.input_size), antialias=True)(mask.unsqueeze(0)).squeeze(0)
        
        return image, sample['expression'], mask, f"{sample['image_id']:05d}.jpg", 'individual', transformed_image, effect_name


class RefSegRSDataset:
    def __init__(self, dataset_root, split='val', input_size=480, max_samples=None, seed=42, historic=False):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.max_samples = max_samples
        self.historic = historic
        
        # Set random seed
        random.seed(seed)
        
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
                self.samples.append({
                    'expr_id': expr_id,
                    'image_id': image_id,
                    'image_path': image_path,
                    'mask_path': mask_path,
                    'expression': expression
                })
        
        # If max_samples is specified, randomly sample
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:self.max_samples]
        
        print(f"- Valid samples (with existing files): {len(self.samples)}")
        if self.max_samples is not None:
            print(f"- Selected for processing: {len(self.samples)}")
    
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
        image_pil = Image.open(sample['image_path']).convert('RGB')
        
        # Apply historic transformations if enabled
        transformed_image = None
        effect_name = None
        if self.historic:
            # Convert PIL to numpy array (RGB format)
            original_image_array = np.array(image_pil)
            # Apply random historic effect to numpy array (in BGR format for OpenCV functions)
            bgr_image = cv2.cvtColor(original_image_array, cv2.COLOR_RGB2BGR)
            transformed_image_bgr, effect_name = apply_random_historic_effect(bgr_image)
            
            # Convert back to RGB for PIL processing
            transformed_image_rgb = cv2.cvtColor(transformed_image_bgr, cv2.COLOR_BGR2RGB)
            transformed_image = transformed_image_rgb.copy()  # Store for visualization
            
            # Use transformed image for model input
            image_pil = Image.fromarray(transformed_image_rgb)
        
        image = self.transform(image_pil)
        
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
        
        return image, sample['expression'], mask, f"expr{sample['expr_id']}_img{sample['image_id']}", 'individual', transformed_image, effect_name


class NWPUDataset:
    def __init__(self, dataset_root, split='val', input_size=480, max_samples=None, seed=42, historic=False):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.max_samples = max_samples
        self.historic = historic
        
        # Set random seed
        random.seed(seed)
        
        # Import the NWPUReferProcessor from our processing script
        import sys
        sys.path.append('../datagen/utils')
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
        
        # If max_samples is specified, randomly sample indices
        if self.max_samples is not None and len(self.sample_indices) > self.max_samples:
            random.shuffle(self.sample_indices)
            self.sample_indices = self.sample_indices[:self.max_samples]
        
        print(f"\nNWPU-Refer Dataset ({split} split):")
        print(f"- Total samples in split: {total_samples}")
        print(f"- Samples to process: {len(self.sample_indices)}")
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        # Get the actual sample index
        sample_idx = self.sample_indices[idx]
        
        # Get sample from processor
        sample = self.processor.get_sample(self.split, sample_idx)
        
        if sample is None:
            # If sample loading fails, create a dummy sample
            dummy_image = torch.zeros(3, self.input_size, self.input_size)
            dummy_mask = torch.zeros(self.input_size, self.input_size)
            return dummy_image, "error loading sample", dummy_mask, f"error_{idx}", 'individual'
        
        # Apply historic transformations if enabled
        original_image_array = sample['image']
        if self.historic:
            # Apply random historic effect to numpy array (in BGR format for OpenCV functions)
            bgr_image = cv2.cvtColor(original_image_array, cv2.COLOR_RGB2BGR)
            transformed_image, effect_name = apply_random_historic_effect(bgr_image)
            
            # Convert back to RGB for PIL processing
            if len(transformed_image.shape) == 2:
                # Grayscale to RGB
                rgb_image = cv2.cvtColor(transformed_image, cv2.COLOR_GRAY2RGB)
            else:
                # BGR to RGB
                rgb_image = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
            
            # Store the transformed image for visualization
            self._last_transformed_image = rgb_image
            self._last_effect_name = effect_name
        else:
            rgb_image = original_image_array
            self._last_transformed_image = None
            self._last_effect_name = None
        
        # Convert image to PIL and apply transform
        image_pil = Image.fromarray(rgb_image).convert('RGB')
        image = self.transform(image_pil)
        
        # Convert mask to tensor and resize
        mask = torch.from_numpy(sample['mask']).float()
        # Normalize mask to 0-1 range
        mask = mask / 255.0 if mask.max() > 1.0 else mask
        mask = T.Resize((self.input_size, self.input_size), antialias=True)(mask.unsqueeze(0)).squeeze(0)
        
        # Get expression
        expression = sample['expression']
        
        # Create a filename for identification (avoid long filenames with many refs)
        if len(sample['ref_ids']) > 5:
            filename = f"img{sample['image_id']}_refs{len(sample['ref_ids'])}items"
        else:
            filename = f"img{sample['image_id']}_refs{'_'.join(map(str, sample['ref_ids']))}"
        
        # Determine type based on number of objects
        sample_type = 'group' if sample['num_objects'] > 1 else 'individual'
        
        # Store transformed image for visualization if historic flag is enabled
        if self.historic and hasattr(self, '_last_transformed_image'):
            return image, expression, mask, filename, sample_type, self._last_transformed_image, self._last_effect_name
        else:
            return image, expression, mask, filename, sample_type, None, None


class Urban1960Dataset:
    def __init__(self, dataset_root, split='val', input_size=480, max_samples=None, seed=42, historic=False):
        self.dataset_root = dataset_root
        self.split = split
        self.input_size = input_size
        self.max_samples = max_samples
        self.historic = historic  # Always False for Urban1960 (already historic imagery)
        
        # Set random seed
        random.seed(seed)
        
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
            'SS_1': "all buildings in the image", 
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
                # Early termination if we have enough samples for testing
                if self.max_samples is not None and len(self.samples) >= self.max_samples * 2:
                    print(f"  Early termination at {len(self.samples)} samples for ISP processing")
                    break
                    
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
                    isp_classes_added = 0
                    for class_id in unique_classes:
                        if class_id in [0, 1]:  # ISP classes
                            self.samples.append({
                                'image_name': image_name,
                                'image_path': image_path,
                                'mask_path': mask_path,
                                'class_id': class_id,
                                'dataset_type': 'ISP',
                                'expression': self.class_expressions[f'ISP_{class_id}']
                            })
                            isp_classes_added += 1
                    if isp_classes_added > 0:
                        print(f"    Added {isp_classes_added} ISP classes from {image_name}: {unique_classes}")
                            
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
                # Early termination if we have enough samples for testing
                if self.max_samples is not None and len(self.samples) >= self.max_samples * 2:
                    print(f"  Early termination at {len(self.samples)} samples for SS processing")
                    break
                    
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
                            self.samples.append({
                                'image_name': image_name,
                                'image_path': image_path,
                                'mask_path': mask_path,
                                'class_id': class_id,
                                'dataset_type': 'SS',
                                'expression': self.class_expressions[f'SS_{class_id}']
                            })
                            
                except Exception as e:
                    print(f"Warning: Error processing SS {image_name}: {e}")
                    continue
        
        # If max_samples is specified, randomly sample
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            random.shuffle(self.samples)
            self.samples = self.samples[:self.max_samples]
        
        print(f"- Total samples (class instances): {len(self.samples)}")
        if self.max_samples is not None:
            print(f"- Selected for processing: {len(self.samples)}")
        
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
        image_pil = Image.open(sample['image_path']).convert('RGB')
        
        # No historic transformations (already historic imagery)
        transformed_image = None
        effect_name = None
        
        image = self.transform(image_pil)
        
        # Load mask and create binary mask for this specific class
        mask_pil = Image.open(sample['mask_path'])
        mask_array = np.array(mask_pil)
        
        # Create binary mask for the specific class
        class_id = sample['class_id']
        dataset_type = sample['dataset_type']
        binary_mask = (mask_array == class_id).astype(np.uint8)
        
        # Convert to tensor and resize
        mask = torch.from_numpy(binary_mask).float()
        mask = T.Resize((self.input_size, self.input_size), antialias=True)(mask.unsqueeze(0)).squeeze(0)
        
        # Create filename for identification
        filename = f"{sample['image_name']}_{dataset_type}_class{class_id}"
        
        return image, sample['expression'], mask, filename, 'individual', transformed_image, effect_name


def test(model, test_loader, device, output_dir, num_vis=20, vis_only=False):
    model.eval()
    
    # For mIoU calculation
    all_ious = []
    
    # For oIoU calculation - using running counters instead of storing all tensors
    total_intersection = 0
    total_union = 0
    
    # For pass@k metrics
    pass_thresholds = [0.5, 0.7, 0.9]
    pass_counts = {thresh: 0 for thresh in pass_thresholds}
    
    vis_count = 0
    visualized_images = set()  # Track which images we've already visualized
    
    print("\nTesting model...")
    with torch.no_grad():
        for batch_idx, (images, texts, masks, image_ids, sample_types, transformed_images, effect_names) in enumerate(test_loader):
                
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
                
                # Update pass@k counters
                for thresh in pass_thresholds:
                    if iou_score >= thresh:
                        pass_counts[thresh] += 1
                
                # Update running counters for oIoU calculation
                intersection = torch.logical_and(pred_binary, target_binary).sum().item()
                union = torch.logical_or(pred_binary, target_binary).sum().item()
                
                total_intersection += intersection
                total_union += union
            
            # Visualize first sample from each unique image - up to 20 different images
            if vis_count < num_vis and image_id not in visualized_images:
                print(f"Visualizing image {vis_count + 1}/20: {image_id}")
                # Create unique filename using vis_count to avoid overwriting
                save_path = os.path.join(output_dir, f"val_{vis_count:03d}_{image_id}.png")
                # Pass transformed image and effect name if available
                transformed_img = transformed_images[0] if transformed_images[0] is not None else None
                effect_name = effect_names[0] if effect_names[0] is not None else None
                visualize_predictions(image, mask, output, text, save_path, transformed_img, effect_name)
                visualized_images.add(image_id)
                vis_count += 1
            
            # Clean GPU memory after each batch
            del images, masks, outputs, image, mask, output
            if not vis_only:
                del pred_binary, target_binary
            torch.cuda.empty_cache()
            
            if not vis_only:
                vis_msg = f" - Visualized!" if (vis_count <= num_vis and image_id in visualized_images) else ""
                print(f"Sample {batch_idx} (ID: {image_id}) - IoU: {iou_score:.4f}{vis_msg}")
            else:
                print(f"Visualizing sample {batch_idx} (ID: {image_id}) - Type: {sample_type}")
            
            # If we've found enough unique images for visualization, and we're in vis_only mode, break
            if vis_only and vis_count >= num_vis:
                break
    
    if not vis_only:
        # Calculate mIoU (mean of individual IoUs)
        iou_values = [iou for _, iou in all_ious]
        miou = sum(iou_values) / len(iou_values)
        
        # Calculate oIoU using the running counters
        oiou = total_intersection / total_union if total_union > 0 else 0.0
        
        # Calculate pass@k metrics
        total_samples = len(iou_values)
        pass_metrics = {}
        for thresh in pass_thresholds:
            pass_rate = pass_counts[thresh] / total_samples * 100
            pass_metrics[thresh] = pass_rate
        
        print("\nValidation Results:")
        print(f"mIoU (mean of individual IoUs): {miou:.4f}")
        print(f"oIoU (overall IoU): {oiou:.4f}")
        for thresh in pass_thresholds:
            print(f"Pass@{thresh}: {pass_metrics[thresh]:.2f}% ({pass_counts[thresh]}/{total_samples})")
        print(f"Min IoU: {min(iou_values):.4f}")
        print(f"Max IoU: {max(iou_values):.4f}")
        print(f"Median IoU: {np.median(iou_values):.4f}")
        
        # Save results to file
        results_file = os.path.join(output_dir, "validation_results.txt")
        with open(results_file, 'w') as f:
            f.write(f"mIoU (mean of individual IoUs): {miou:.4f}\n")
            f.write(f"oIoU (overall IoU): {oiou:.4f}\n")
            for thresh in pass_thresholds:
                f.write(f"Pass@{thresh}: {pass_metrics[thresh]:.2f}% ({pass_counts[thresh]}/{total_samples})\n")
            f.write(f"Min IoU: {min(iou_values):.4f}\n")
            f.write(f"Max IoU: {max(iou_values):.4f}\n")
            f.write(f"Median IoU: {np.median(iou_values):.4f}\n\n")
            
            f.write("Individual IoU scores:\n")
            for image_id, iou in sorted(all_ious, key=lambda x: x[0]):
                f.write(f"Image {image_id}: {iou:.4f}\n")
        
        # Plot IoU histogram with pass@k thresholds
        plt.figure(figsize=(12, 6))
        plt.hist(iou_values, bins=20, alpha=0.7, color='lightblue')
        plt.axvline(miou, color='r', linestyle='--', label=f'mIoU: {miou:.4f}')
        plt.axvline(oiou, color='b', linestyle='--', label=f'oIoU: {oiou:.4f}')
        plt.axvline(np.median(iou_values), color='g', linestyle='--', label=f'Median IoU: {np.median(iou_values):.4f}')
        
        # Add pass@k threshold lines
        colors = ['orange', 'purple', 'red']
        for i, thresh in enumerate(pass_thresholds):
            plt.axvline(thresh, color=colors[i], linestyle=':', alpha=0.8, 
                       label=f'Pass@{thresh} threshold ({pass_metrics[thresh]:.2f}%)')
        
        plt.xlabel('IoU')
        plt.ylabel('Count')
        plt.title('Distribution of IoU Scores with Pass@k Thresholds')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "iou_histogram.png"), dpi=300, bbox_inches='tight')
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
        
        # Return all metrics including pass@k
        metrics_dict = {
            "miou": miou, 
            "oiou": oiou,
            **{f"pass_{thresh}": pass_metrics[thresh] for thresh in pass_thresholds}
        }
        return metrics_dict
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
            seed=args.seed,
            historic=args.historic
        )
    elif args.dataset_type == 'rrsisd':
        val_dataset = RRSISDDataset(
            dataset_root=args.rrsisd_root,
            split='val',  # Not used for RRSISD but kept for compatibility
            input_size=args.input_size,
            max_samples=max_samples,
            seed=args.seed,
            historic=args.historic
        )
    elif args.dataset_type == 'refsegrs':
        val_dataset = RefSegRSDataset(
            dataset_root=args.refsegrs_root,
            split='val',
            input_size=args.input_size,
            max_samples=max_samples,
            seed=args.seed,
            historic=args.historic
        )
    elif args.dataset_type == 'nwpu':
        val_dataset = NWPUDataset(
            dataset_root=args.nwpu_root,
            split='val',
            input_size=args.input_size,
            max_samples=max_samples,
            seed=args.seed,
            historic=args.historic
        )
    elif args.dataset_type == 'urban1960':
        val_dataset = Urban1960Dataset(
            dataset_root=args.urban1960_root,
            split='val',
            input_size=args.input_size,
            max_samples=max_samples,
            seed=args.seed,
            historic=False  # Always False for Urban1960 (already historic imagery)
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
            [item[4] for item in batch],               # sample_types
            [item[5] for item in batch],               # transformed_images (can be None)
            [item[6] for item in batch]                # effect_names (can be None)
        )
    )
    
    print(f"\nStarting validation with {len(val_dataset)} samples...")
    
    # Display historic transformations info if enabled
    if args.historic:
        print("Historic transformations enabled:")
        print("  - Random selection of Basic B&W, B&W + Grain, or Sepia + Noise effects")
        print("  - Applied with equal probability to each test image")
        print("  - Visualizations will show transformed images")
    
    # Create output directory with dataset type
    historic_suffix = '_historic' if args.historic else ''
    
    if args.dataset_type == 'rrsisd':
        output_dir = os.path.join('./results', f'{args.model_name}_rrsisd{historic_suffix}')
    elif args.dataset_type == 'refsegrs':
        output_dir = os.path.join('./results', f'{args.model_name}_refsegrs{historic_suffix}')
    elif args.dataset_type == 'nwpu':
        output_dir = os.path.join('./results', f'{args.model_name}_nwpu{historic_suffix}')
    elif args.dataset_type == 'urban1960':
        output_dir = os.path.join('./results', f'{args.model_name}_urban1960{historic_suffix}')
    else:  # aeriald dataset
        output_dir = os.path.join('./results', f'{args.model_name}_aeriald{historic_suffix}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Run testing
    metrics = test(model, val_loader, device, output_dir, args.num_vis, args.vis_only)
    
    if not args.vis_only:
        print(f"\nValidation complete. Results saved to {output_dir}")
        print(f"Final mIoU: {metrics['miou']:.4f}")
        print(f"Final oIoU: {metrics['oiou']:.4f}")
        for thresh in [0.5, 0.7, 0.9]:
            print(f"Final Pass@{thresh}: {metrics[f'pass_{thresh}']:.2f}%")
    else:
        print(f"\nVisualization complete. Results saved to {output_dir}")

if __name__ == '__main__':
    main() 