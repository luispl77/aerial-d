#!/usr/bin/env python3
import os
import random
import argparse
import json
import xml.etree.ElementTree as ET
import shutil
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
import torch
import time
import numpy as np
from pycocotools import mask as mask_utils
from scipy import ndimage
from skimage.measure import label, regionprops

# Completely disable torch compilation and dynamo
os.environ["TORCH_COMPILE"] = "0"
torch._dynamo.config.disable = True
torch.backends.cudnn.enabled = False  # Disable cudnn compilation
torch.set_float32_matmul_precision('high')  # Suppress the warning

def decode_rle_and_get_bbox(segmentation_str):
    """Decode RLE segmentation and compute bounding box."""
    try:
        # Parse the segmentation string (it's a dict-like string)
        seg_dict = eval(segmentation_str)
        
        # Create RLE dict in the format expected by pycocotools
        rle = {
            'size': seg_dict['size'],
            'counts': seg_dict['counts'].encode('utf-8')
        }
        
        # Decode RLE to binary mask
        mask = mask_utils.decode(rle)
        
        # Find bounding box from mask
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        bbox = {
            'xmin': int(cmin),
            'ymin': int(rmin),
            'xmax': int(cmax + 1),  # +1 because we want inclusive bbox
            'ymax': int(rmax + 1)
        }
        
        return bbox
        
    except Exception as e:
        print(f"Error decoding RLE segmentation: {e}")
        return None

def decode_rle_and_get_individual_bboxes(segmentation_str, min_area=10):
    """Decode RLE segmentation and compute individual bounding boxes for each connected component."""
    try:
        # Parse the segmentation string (it's a dict-like string)
        seg_dict = eval(segmentation_str)
        
        # Create RLE dict in the format expected by pycocotools
        rle = {
            'size': seg_dict['size'],
            'counts': seg_dict['counts'].encode('utf-8')
        }
        
        # Decode RLE to binary mask
        mask = mask_utils.decode(rle)
        
        # Find connected components
        labeled_mask = label(mask, connectivity=2)  # 8-connected
        regions = regionprops(labeled_mask)
        
        bboxes = []
        for region in regions:
            # Filter out very small components (noise)
            if region.area < min_area:
                continue
                
            # Get bounding box coordinates (minr, minc, maxr, maxc)
            minr, minc, maxr, maxc = region.bbox
            
            bbox = {
                'xmin': int(minc),
                'ymin': int(minr),
                'xmax': int(maxc),
                'ymax': int(maxr)
            }
            bboxes.append(bbox)
        
        return bboxes if bboxes else None
        
    except Exception as e:
        print(f"Error decoding RLE segmentation for individual objects: {e}")
        return None

# Constants for the enhancement
NUM_ENHANCED = 1  # Number of enhanced expressions per original
NUM_UNIQUE = 2    # Number of unique expressions to generate

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhance aerial image annotations using fine-tuned Gemma 3')
    parser.add_argument('--dataset_path', type=str, default='/cfs/home/u035679/datasets/old/aeriald_old2/train',
                        help='Path to the dataset directory')
    parser.add_argument('--model_path', type=str, default='./gemma-aerial-12b',
                        help='Path to the fine-tuned Gemma 3 model')
    parser.add_argument('--use_lora', action='store_true',
                        help='Use LoRA adapter with base model instead of merged model')
    parser.add_argument('--base_model', type=str, default='google/gemma-3-12b-it',
                        help='Base model to use when loading LoRA adapter')
    parser.add_argument('--lora_path', type=str, default='./gemma-aerial-referring-12b-lora',
                        help='Path to LoRA adapter weights')
    parser.add_argument('--output_dir', type=str, default='./enhanced_gemma_annotations_dual',
                        help='Directory to save enhanced annotations')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--specific_file', type=str, default=None,
                        help='Process a specific file instead of random selection (filename without extension)')
    parser.add_argument('--num_objects', type=int, default=1,
                        help='Number of random objects to process (ignored if specific_file is provided)')
    parser.add_argument('--clean', action='store_true',
                        help='Clear the output directory before processing')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--dataset_filter', type=str, choices=['deepglobe', 'isaid', 'loveda'], default=None,
                        help='Filter samples by source dataset: deepglobe (D*), isaid (P*), loveda (L*)')
    parser.add_argument('--crop_fraction', type=float, default=0.5,
                        help='Fraction of original image size to use for focused crop (default: 0.5 for half image)')
    return parser.parse_args()

def get_random_files(args, max_files=100):
    """Get random image files and their corresponding annotation files.
    
    Returns up to max_files random file pairs to sample objects from.
    """
    random.seed(args.random_seed)
    
    images_dir = os.path.join(args.dataset_path, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    
    # Apply dataset filter if specified
    if args.dataset_filter:
        if args.dataset_filter == 'deepglobe':
            prefix = 'D'
            dataset_name = 'DeepGlobe'
        elif args.dataset_filter == 'isaid':
            prefix = 'P'
            dataset_name = 'iSAID'
        elif args.dataset_filter == 'loveda':
            prefix = 'L'
            dataset_name = 'LoveDA'
        
        # Filter files based on prefix
        filtered_files = [f for f in image_files if f.startswith(prefix)]
        print(f"Dataset filter applied: {dataset_name} ({prefix}*)")
        print(f"Files before filtering: {len(image_files)}")
        print(f"Files after filtering: {len(filtered_files)}")
        
        if not filtered_files:
            print(f"Warning: No files found with prefix '{prefix}' for dataset {dataset_name}")
            return []
        
        image_files = filtered_files
    
    # Select random images without replacement, but cap at max_files to avoid scanning entire dataset
    selected_images = random.sample(image_files, min(max_files, len(image_files)))
    
    file_pairs = []
    for image_file in selected_images:
        image_path = os.path.join(images_dir, image_file)
        annotation_file = image_file.replace('.png', '.xml')
        annotation_path = os.path.join(args.dataset_path, 'annotations', annotation_file)
        
        # Check if annotation file exists
        if os.path.exists(annotation_path):
            file_pairs.append((image_path, annotation_path))
        else:
            print(f"Warning: Annotation file not found for {image_file}, skipping...")
    
    return file_pairs

def get_random_objects(args, num_objects=1):
    """Get random objects and groups from random images."""
    random.seed(args.random_seed)
    
    # Get a pool of file pairs to sample from (limiting to avoid scanning entire dataset)
    file_pairs = get_random_files(args, max_files=100)
    
    if not file_pairs:
        print("No valid image-annotation pairs found!")
        return []
    
    # Collect all objects and groups with expressions
    all_items = []
    for image_path, annotation_path in file_pairs:
        annotation_data = parse_annotations(annotation_path)
        
        # Add individual objects
        for obj in annotation_data['objects']:
            if obj['expressions']:
                all_items.append({
                    'image_path': image_path,
                    'annotation_path': annotation_path,
                    'object_data': obj
                })
        
        # Add groups
        for group in annotation_data['groups']:
            if group['expressions']:
                all_items.append({
                    'image_path': image_path,
                    'annotation_path': annotation_path,
                    'object_data': group
                })
    
    # Sample random items (objects or groups)
    if not all_items:
        print("No objects or groups with expressions found in the sampled images!")
        return []
    
    selected_items = random.sample(all_items, min(num_objects, len(all_items)))
    
    objects_count = sum(1 for item in selected_items if item['object_data']['type'] == 'object')
    groups_count = sum(1 for item in selected_items if item['object_data']['type'] == 'group')
    
    print(f"Selected {len(selected_items)} random items from {len(file_pairs)} images:")
    print(f"  - {objects_count} individual objects")
    print(f"  - {groups_count} groups")
    
    return selected_items

def parse_annotations(annotation_path):
    """Extract annotations from XML file."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    objects = []
    groups = []
    total_objects = 0
    objects_with_expressions = 0
    total_groups = 0
    groups_with_expressions = 0
    
    # Parse individual objects
    for obj in root.findall('object'):
        total_objects += 1
        # Get category
        category = None
        if obj.find('name') is not None:
            category = obj.find('name').text
        elif obj.find('n') is not None:
            category = obj.find('n').text
        else:
            category = "unknown"
        
        # Get object ID
        obj_id = obj.find('id').text if obj.find('id') is not None else "unknown"
        
        # Get bounding box
        bbox = {}
        if obj.find('bndbox') is not None:
            bbox = {
                'xmin': int(obj.find('bndbox/xmin').text),
                'ymin': int(obj.find('bndbox/ymin').text),
                'xmax': int(obj.find('bndbox/xmax').text),
                'ymax': int(obj.find('bndbox/ymax').text)
            }
        
        # Initialize expressions list
        expressions = []
        
        # Only check for explicit expressions tag - don't generate any new ones
        if obj.find('expressions') is not None:
            for expr in obj.findall('expressions/expression'):
                if expr.text is not None and expr.text.strip():  # Make sure it's not empty
                    expressions.append(expr.text.strip())
        
        # Only add to objects list if we have a bounding box AND existing expressions
        if bbox and expressions:
            objects_with_expressions += 1
            obj_dict = {
                'id': obj_id,
                'category': category,
                'bbox': bbox,
                'expressions': expressions,
                'type': 'object'
            }
            objects.append(obj_dict)
    
    # Parse groups
    groups_elem = root.find('groups')
    if groups_elem is not None:
        for group in groups_elem.findall('group'):
            total_groups += 1
            
            # Get group ID
            group_id = group.find('id').text if group.find('id') is not None else "unknown"
            
            # Get category
            category = group.find('category').text if group.find('category') is not None else "unknown"
            
            # Get size
            size = group.find('size').text if group.find('size') is not None else "unknown"
            
            # Get centroid (optional, for metadata)
            centroid = {}
            centroid_elem = group.find('centroid')
            if centroid_elem is not None:
                x_elem = centroid_elem.find('x')
                y_elem = centroid_elem.find('y')
                if x_elem is not None and y_elem is not None:
                    centroid = {
                        'x': float(x_elem.text),
                        'y': float(y_elem.text)
                    }
            
            # Get segmentation and compute individual bounding boxes from RLE mask
            bboxes = []
            segmentation_elem = group.find('segmentation')
            if segmentation_elem is not None and segmentation_elem.text:
                individual_bboxes = decode_rle_and_get_individual_bboxes(segmentation_elem.text)
                if individual_bboxes:
                    bboxes = individual_bboxes
                else:
                    # Fallback to single bbox if component separation fails
                    single_bbox = decode_rle_and_get_bbox(segmentation_elem.text)
                    if single_bbox:
                        bboxes = [single_bbox]
            
            # Initialize expressions list
            expressions = []
            if group.find('expressions') is not None:
                for expr in group.findall('expressions/expression'):
                    if expr.text is not None and expr.text.strip():
                        expressions.append(expr.text.strip())
            
            # Only add to groups list if we have valid bboxes AND existing expressions
            if bboxes and expressions:
                groups_with_expressions += 1
                group_dict = {
                    'id': group_id,
                    'category': category,
                    'size': size,
                    'centroid': centroid,
                    'bboxes': bboxes,  # List of individual bboxes computed from RLE segmentation mask
                    'expressions': expressions,
                    'type': 'group'
                }
                groups.append(group_dict)
    
    print(f"Parsed {objects_with_expressions} objects and {groups_with_expressions} groups with expressions")
    
    return {
        'filename': filename,
        'objects': objects,
        'groups': groups
    }

def create_prompt(object_name, original_expressions, image_mode="bbox_dual"):
    """Create the detailed prompt for Gemma 3 with dual image support."""
    formatted_expressions = "\n".join([f"- {expr}" for expr in original_expressions])
    
    # Different intro text based on image mode
    if image_mode == "mask_dual":
        intro_text = (
            "You are an expert at creating natural language descriptions for objects and groups in aerial imagery. "
            "Your task is to help create diverse and precise referring expressions for the target. "
            "The target is a group/collection of multiple objects with semantic segmentation.\n\n"
            
            "I am providing you with TWO images:\n"
            "1. MASK IMAGE: Full aerial scene with the target region highlighted by a red mask overlay\n"
            "2. CLEAN IMAGE: The same full aerial scene without any highlighting\n\n"
            
            "IMPORTANT: The red highlighting in the first image indicates the target area, but does NOT mean the objects are actually red in color. "
            "Look at the clean image (second image) to understand the true appearance and colors of the target objects.\n\n"
            
            "Use BOTH images to understand the target and its context better.\n\n"
        )
    else:  # bbox_dual mode
        intro_text = (
            "You are an expert at creating natural language descriptions for objects and groups in aerial imagery. "
            "Your task is to help create diverse and precise referring expressions for the target. "
            "The target may be a single object or a group/collection of multiple objects.\n\n"
            
            "I am providing you with TWO images:\n"
            "1. CONTEXT IMAGE: Full aerial scene with the target highlighted by red bounding box(es)\n"
            "2. FOCUSED IMAGE: Close-up crop centered on the target area without bounding boxes\n\n"
            
            "Use BOTH images to understand the target and its context better.\n\n"
        )
    
    system_prompt = (
        intro_text +
        "IMPORTANT GUIDELINES:\n"
        "- If the original expressions refer to 'all', 'group of', or multiple objects, maintain this collective reference\n"
        "- If working with a group, use plural forms and consider the spatial distribution of the entire collection\n"
        "- If working with a single object, focus on that specific instance\n"
        "- Always preserve the scope and meaning of the original expressions\n"
        "- NEVER reference red boxes, masks, or markings in your expressions\n\n"
        
        "You have three tasks:\n\n"
        
        f"TASK 1: For each original expression listed below, create EXACTLY {NUM_ENHANCED} language variation that:\n"
        "1. MUST PRESERVE ALL SPATIAL INFORMATION from the original expression:\n"
        "   - Absolute positions (e.g., \"in the top right\", \"near the center\")\n"
        "   - Relative positions (e.g., \"to the right of\", \"below\")\n"
        "   - Collective scope (e.g., \"all\", \"group of\", individual references)\n"
        "2. Use natural, everyday language that a regular person would use\n"
        "   - Avoid overly formal or technical vocabulary\n"
        "   - Use common synonyms (e.g., \"car\" instead of \"automobile\")\n"
        "   - Keep the tone conversational and straightforward\n"
        "3. Ensure the variation uniquely identifies the target to avoid ambiguity\n"
        "4. Maintain the same scope as the original (single object vs. group/collection)\n\n"
        
        "TASK 2: Analyze the target's context and uniqueness factors:\n"
        "1. Examine the immediate surroundings of the target\n"
        "2. Identify distinctive features that could be used to uniquely identify the target:\n"
        "   - Nearby objects and their relationships\n"
        "   - Visual characteristics that distinguish it from similar objects\n"
        "   - Environmental context (roads, buildings, terrain) that provide reference points\n"
        "   - For groups: spatial distribution and arrangement patterns\n"
        "3. Consider how the original automated expressions could be improved\n"
        "4. Focus on features that would help someone locate this specific target without ambiguity\n\n"
        
        f"TASK 3: Generate EXACTLY {NUM_UNIQUE} new expressions that:\n"
        "1. MUST be based on one of the original expressions or their variations\n"
        "2. Add visual details ONLY when you are highly confident about them\n"
        "3. Each expression must uniquely identify the target\n"
        "4. Focus on describing the target's relationship with its immediate surroundings\n"
        "5. Maintain the core spatial information from the original expression\n"
        "6. Preserve the same scope as the original (individual vs. collective reference)\n\n"
        
        f"ORIGINAL EXPRESSIONS TO ENHANCE:\n{formatted_expressions}\n\n"
        
        "You must return your output in the following JSON format:\n"
        "{\n"
        "  \"enhanced_expressions\": [\n"
        "    {\n"
        "      \"original_expression\": \"<original expression>\",\n"
        "      \"variation\": \"<single language variation>\"\n"
        "    },\n"
        "    ...\n"
        "  ],\n"
        "  \"unique_description\": \"<detailed analysis of spatial context and uniqueness factors>\",\n"
        "  \"unique_expressions\": [\n"
        "    \"<new expression based on original 1>\",\n"
        "    \"<new expression based on original 2>\"\n"
        "  ]\n"
        "}\n"
        "Only return the JSON object, no other text or comments.\n"
        "Write all the expressions using lowercase letters and no punctuation."
    )
    
    user_prompt = (
        f"Create language variations of the provided expressions while preserving spatial information, "
        f"analyze the spatial context for uniqueness factors, and generate new unique expressions for this {object_name}. "
        f"Use both images to understand the target better."
    )
    
    return system_prompt, user_prompt

def visualize_and_save_object(image_path, bboxes, output_path):
    """Visualize object(s) with bounding box(es) and save to file.
    
    Args:
        image_path: Path to the image file
        bboxes: Either a single bbox dict or a list of bbox dicts
        output_path: Where to save the visualization
    """
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # Handle both single bbox and list of bboxes
    if isinstance(bboxes, dict):
        bboxes = [bboxes]
    
    # Draw all bounding boxes
    for bbox in bboxes:
        rect = patches.Rectangle(
            (bbox['xmin'], bbox['ymin']),
            bbox['xmax'] - bbox['xmin'],
            bbox['ymax'] - bbox['ymin'],
            linewidth=1,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
    
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def setup_model_and_processor(args):
    """Setup the model and processor with either merged model or LoRA adapter."""
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    if args.use_lora:
        print(f"Loading base model: {args.base_model}")
        print(f"Loading LoRA adapter from: {args.lora_path}")
        
        # Load base model
        model = AutoModelForImageTextToText.from_pretrained(
            args.base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )
        
        # Load LoRA adapter
        model = PeftModel.from_pretrained(model, args.lora_path)
        
        # Load processor from base model
        processor = AutoProcessor.from_pretrained(args.base_model)
        
        print("Base model and LoRA adapter loaded successfully")
    else:
        print(f"Loading merged fine-tuned model from: {args.model_path}")
        
        # Load merged model
        model = AutoModelForImageTextToText.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="eager"
        )
        
        # Load processor from fine-tuned model path
        processor = AutoProcessor.from_pretrained(args.model_path)
        
        print("Merged model loaded successfully")
    
    return model, processor

def compute_centroid_from_bboxes(bboxes):
    """Compute centroid from a list of bounding boxes or a single bbox."""
    if isinstance(bboxes, dict):
        # Single bbox
        center_x = (bboxes['xmin'] + bboxes['xmax']) / 2
        center_y = (bboxes['ymin'] + bboxes['ymax']) / 2
        return center_x, center_y
    elif isinstance(bboxes, list):
        # Multiple bboxes - compute center of all bbox centers
        total_x = 0
        total_y = 0
        for bbox in bboxes:
            center_x = (bbox['xmin'] + bbox['xmax']) / 2
            center_y = (bbox['ymin'] + bbox['ymax']) / 2
            total_x += center_x
            total_y += center_y
        return total_x / len(bboxes), total_y / len(bboxes)
    else:
        raise ValueError("bboxes must be a dict or list of dicts")

def create_focused_crop(image_path, bboxes, crop_size=384, image_fraction=0.5):
    """Create a focused square crop around the target area with black padding if needed."""
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # Compute centroid
    center_x, center_y = compute_centroid_from_bboxes(bboxes)
    
    # Use a fixed fraction of the original image size
    # Make it square by using the smaller dimension to ensure we don't exceed image bounds
    min_dimension = min(img_width, img_height)
    target_size = min_dimension * image_fraction
    
    # Compute crop coordinates centered on the centroid
    half_size = target_size / 2
    crop_left = int(center_x - half_size)
    crop_right = int(center_x + half_size)
    crop_top = int(center_y - half_size)
    crop_bottom = int(center_y + half_size)
    
    # Handle cases where crop goes outside image boundaries
    # Calculate how much padding we need on each side
    pad_left = max(0, -crop_left)
    pad_right = max(0, crop_right - img_width)
    pad_top = max(0, -crop_top)
    pad_bottom = max(0, crop_bottom - img_height)
    
    # Adjust crop coordinates to stay within image bounds
    crop_left = max(0, crop_left)
    crop_right = min(img_width, crop_right)
    crop_top = max(0, crop_top)
    crop_bottom = min(img_height, crop_bottom)
    
    # Extract the crop
    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    # Create a square canvas with black background
    square_size = int(target_size)
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        # Create black canvas
        canvas = Image.new('RGB', (square_size, square_size), (0, 0, 0))
        
        # Calculate where to paste the cropped image
        paste_x = pad_left
        paste_y = pad_top
        
        canvas.paste(cropped, (paste_x, paste_y))
        result = canvas
    else:
        result = cropped
    
    # Resize to final target size
    result = result.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    
    return result

def create_mask_overlay(image_path, segmentation_str, output_path, mask_color=(255, 0, 0), mask_alpha=0.3):
    """Create an image with red mask overlay from RLE segmentation data."""
    # Load the original image
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    
    # Decode the RLE mask
    mask = decode_rle_and_get_mask(segmentation_str)
    if mask is None:
        print(f"Error: Could not decode segmentation mask for {image_path}")
        # Fallback: save original image
        img.save(output_path)
        return
    
    # Convert mask to numpy array and ensure it's binary
    mask_array = np.array(mask, dtype=bool)
    
    # Create a colored mask overlay
    mask_overlay = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    mask_overlay[mask_array] = mask_color
    
    # Convert images to numpy arrays
    img_array = np.array(img)
    
    # Blend the original image with the mask overlay
    # Only apply overlay where mask is True
    blended = img_array.copy()
    for c in range(3):  # RGB channels
        blended[mask_array, c] = (
            (1 - mask_alpha) * img_array[mask_array, c] + 
            mask_alpha * mask_overlay[mask_array, c]
        ).astype(np.uint8)
    
    # Convert back to PIL Image and save
    result_img = Image.fromarray(blended)
    result_img.save(output_path)

def process_single_object(object_info, model_processor_tuple, args):
    """Process a single object or group and return results."""
    model, processor = model_processor_tuple
    image_path = object_info['image_path']
    obj = object_info['object_data']
    
    image_name = os.path.basename(image_path).split('.')[0]
    item_type = obj.get('type', 'object')
    
    # Determine processing mode based on dataset and item type
    if image_name.startswith('D'):  # DeepGlobe - only has groups
        mode = 'mask_dual'
        mode_desc = 'DeepGlobe group with mask overlay'
    elif image_name.startswith('L'):  # LoveDA
        if item_type == 'group':
            mode = 'mask_dual'
            mode_desc = 'LoveDA group with mask overlay'
        else:
            mode = 'bbox_dual'
            mode_desc = 'LoveDA instance with bbox and focused crop'
    elif image_name.startswith('P'):  # iSAID
        mode = 'bbox_dual'
        mode_desc = 'iSAID with bbox and focused crop'
    else:
        mode = 'bbox_dual'
        mode_desc = 'Unknown dataset - using bbox and focused crop'
    
    if item_type == 'group':
        print(f"\nProcessing group {obj['id']} (size: {obj['size']}) from image: {image_name}")
        output_dir = os.path.join(args.output_dir, f"{image_name}_group_{obj['id']}")
    else:
        print(f"\nProcessing object {obj['id']} from image: {image_name}")
        output_dir = os.path.join(args.output_dir, f"{image_name}_obj_{obj['id']}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Using {mode_desc}")
    
    # Set up paths for images
    if mode == 'mask_dual':
        # For mask overlay mode (DeepGlobe and LoveDA groups)
        mask_path = os.path.join(output_dir, f"{image_name}_mask.png")
        clean_path = os.path.join(output_dir, f"{image_name}_clean.png")
        
        # Create mask overlay
        if 'segmentation' not in obj:
            print("Error: No segmentation data found for mask overlay mode")
            return None
        
        # Save clean image first (just a copy)
        shutil.copy2(image_path, clean_path)
        
        # Create and save mask overlay
        create_mask_overlay(image_path, obj['segmentation'], mask_path)
        print("  Created mask overlay and clean images")
        
        main_image_path = mask_path  # First image will be the mask overlay
        second_image_path = clean_path  # Second image will be the clean version
        
    else:  # bbox_dual mode
        # For bbox mode (iSAID and LoveDA instances)
        bbox_path = os.path.join(output_dir, f"{image_name}_bbox.png")
        focused_path = os.path.join(output_dir, f"{image_name}_focused.png")
        
        # Create visualization with bounding box(es)
        if item_type == 'group':
            bboxes_to_visualize = obj['bboxes']
            print(f"  Drawing {len(bboxes_to_visualize)} individual bounding boxes for group")
        else:
            bboxes_to_visualize = obj['bbox']
        
        visualize_and_save_object(image_path, bboxes_to_visualize, bbox_path)
        
        # Create focused crop
        print(f"  Creating focused crop (fraction: {args.crop_fraction})...")
        focused_crop = create_focused_crop(image_path, bboxes_to_visualize, image_fraction=args.crop_fraction)
        focused_crop.save(focused_path)
        print(f"  Focused crop saved to: {focused_path}")
        
        main_image_path = bbox_path  # First image will be the bbox visualization
        second_image_path = focused_path  # Second image will be the focused crop
    
    try:
        # Create prompts with mode-specific text
        system_prompt, user_prompt = create_prompt(obj['category'], obj['expressions'], image_mode=mode)
        
        # Create messages format that matches training
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": main_image_path},
                    {"type": "image", "url": second_image_path},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # Process inputs using the processor
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to("cuda")
        
        # Time the generation
        start_time = time.time()
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.8)
        end_time = time.time()
        
        # Calculate metrics
        generation_time = end_time - start_time
        generated_content = processor.decode(output[0], skip_special_tokens=True)
        
        # Extract only the new content (remove the input prompt)
        input_length = len(processor.decode(inputs['input_ids'][0], skip_special_tokens=True))
        generated_content = generated_content[input_length:].strip()
        
        print(f"Generation time: {generation_time:.2f} seconds")
        print(f"Generated content:\n{generated_content}\n")
        
        # Try to parse JSON response with better error handling
        def extract_and_parse_json(content):
            """Extract JSON from content that might be wrapped in code blocks."""
            import re
            
            # First try to parse as-is
            try:
                return json.loads(content.strip())
            except json.JSONDecodeError:
                pass
            
            # Try to extract JSON from markdown code blocks
            json_patterns = [
                r'```json\s*(.*?)\s*```',  # ```json ... ```
                r'```\s*(.*?)\s*```',      # ``` ... ```
                r'`(.*?)`',                # `...`
            ]
            
            for pattern in json_patterns:
                matches = re.findall(pattern, content, re.DOTALL)
                for match in matches:
                    try:
                        return json.loads(match.strip())
                    except json.JSONDecodeError:
                        continue
            
            # Try to find JSON-like content (starts with { and ends with })
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group().strip())
                except json.JSONDecodeError:
                    pass
            
            # If all else fails, return None
            return None
        
        try:
            enhanced_data = extract_and_parse_json(generated_content)
            if enhanced_data is None:
                raise json.JSONDecodeError("Could not extract valid JSON", generated_content, 0)
            success = True
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            print(f"Raw generated content:\n{generated_content[:500]}...")
            enhanced_data = {
                "enhanced_expressions": [],
                "unique_description": f"Parsing error: {e}",
                "unique_expressions": []
            }
            success = False
        
        if success:
            if item_type == 'group':
                print(f"Successfully enhanced group {obj['id']} ({obj['category']}, size: {obj['size']})")
            else:
                print(f"Successfully enhanced object {obj['id']} ({obj['category']})")
            print(f"Used {mode} mode ({mode_desc})")
            if 'enhanced_expressions' in enhanced_data:
                print(f"Generated {len(enhanced_data['enhanced_expressions'])} enhanced expression groups")
            if 'unique_expressions' in enhanced_data:
                print(f"Generated {len(enhanced_data['unique_expressions'])} unique expressions")
            
    except Exception as e:
        enhanced_data = {
            "enhanced_expressions": [],
            "unique_description": f"Processing error: {e}",
            "unique_expressions": []
        }
        print(f"Error processing object {obj['id']}: {e}")
    
    result = {
        "image": os.path.basename(image_path),
        "item_id": obj['id'],
        "item_type": item_type,
        "category": obj['category'],
        "original_expressions": obj['expressions'],
        "enhanced_data": enhanced_data,
        "processing_mode": mode,
        "processing_description": mode_desc
    }
    
    # Add type-specific information
    if item_type == 'group':
        result["group_size"] = obj['size']
        result["centroid"] = obj['centroid']
        result["num_bboxes"] = len(obj['bboxes'])
        result["bboxes"] = obj['bboxes']
    else:
        result["object_id"] = obj['id']  # Keep for backward compatibility
        result["bbox"] = obj['bbox']
    
    # Save result
    output_file = os.path.join(output_dir, "enhanced_expressions.json")
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Enhancement completed. Results saved to {output_file}")
    return output_file

def main():
    """Main function."""
    args = parse_arguments()
    
    # Clean output directory if requested
    if args.clean and os.path.exists(args.output_dir):
        print(f"Cleaning output directory: {args.output_dir}")
        shutil.rmtree(args.output_dir)
        print("Output directory cleared.")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Show dataset filter information
    if args.dataset_filter:
        dataset_mapping = {
            'deepglobe': 'DeepGlobe (D*)',
            'isaid': 'iSAID (P*)',
            'loveda': 'LoveDA (L*)'
        }
        print(f"Dataset filter: {dataset_mapping[args.dataset_filter]}")
    else:
        print("Dataset filter: None (all datasets)")
    
    # Setup model and processor
    model_processor_tuple = setup_model_and_processor(args)
    
    # Get objects and groups to process
    if args.specific_file:
        image_filename = f"{args.specific_file}.png"
        image_path = os.path.join(args.dataset_path, 'images', image_filename)
        annotation_path = os.path.join(args.dataset_path, 'annotations', f"{args.specific_file}.xml")
        
        annotation_data = parse_annotations(annotation_path)
        objects_to_process = []
        
        # Add individual objects
        for obj in annotation_data['objects']:
            if obj['expressions']:
                objects_to_process.append({
                    'image_path': image_path,
                    'annotation_path': annotation_path,
                    'object_data': obj
                })
        
        # Add groups
        for group in annotation_data['groups']:
            if group['expressions']:
                objects_to_process.append({
                    'image_path': image_path,
                    'annotation_path': annotation_path,
                    'object_data': group
                })
        
        objects_count = sum(1 for item in objects_to_process if item['object_data']['type'] == 'object')
        groups_count = sum(1 for item in objects_to_process if item['object_data']['type'] == 'group')
        print(f"Found {len(objects_to_process)} items in {args.specific_file}:")
        print(f"  - {objects_count} individual objects")
        print(f"  - {groups_count} groups")
    else:
        objects_to_process = get_random_objects(args, args.num_objects)
    
    if not objects_to_process:
        print("No valid objects found!")
        return
    
    print(f"Processing {len(objects_to_process)} object(s)...")
    
    successful_processes = 0
    for i, object_info in enumerate(objects_to_process, 1):
        print(f"\n{'='*50}")
        print(f"Processing object {i}/{len(objects_to_process)}")
        print(f"{'='*50}")
        
        try:
            result = process_single_object(object_info, model_processor_tuple, args)
            if result:
                successful_processes += 1
        except Exception as e:
            print(f"Error processing object {object_info['object_data']['id']} from {os.path.basename(object_info['image_path'])}: {e}")
            continue
    
    print(f"\n{'='*50}")
    print(f"Processing completed!")
    print(f"Successfully processed: {successful_processes}/{len(objects_to_process)} objects")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 