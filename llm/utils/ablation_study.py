#!/usr/bin/env python3
"""
Ablation Study Script for LLM Enhancement Comparison

This script compares three LLM models for aerial image annotation enhancement:
1. OpenAI O3 model (via API)
2. Gemma3 model (via vLLM server)
3. Gemma3-distilled model (via vLLM server)

The script uses dual image prompting logic and tracks token usage for all models.
Results are saved in the debug folder without parsing JSON responses.
"""

import os
import sys
import random
import argparse
import json
import xml.etree.ElementTree as ET
import shutil
import time
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import base64
from io import BytesIO
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pycocotools import mask as mask_utils
from skimage.measure import label, regionprops

# Add the parent directory to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from openai import OpenAI
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install the required packages:")
    print("pip install openai python-dotenv pillow matplotlib pycocotools scikit-image")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Constants
NUM_ENHANCED = 1  # Number of enhanced expressions per original
NUM_UNIQUE = 2    # Number of unique expressions to generate

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run ablation study comparing three LLM models for aerial image annotation enhancement')
    
    # Dataset and sampling
    parser.add_argument('--dataset_root', type=str, default='../datagen/dataset',
                        help='Root path to the AerialD dataset directory')
    parser.add_argument('--split', type=str, choices=['train', 'val'], default='train',
                        help='Dataset split to process')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of random samples to process')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    # Model configuration
    parser.add_argument('--gemma3_model_name', type=str, default='./gemma-aerial-12b',
                        help='Model name for Gemma3 vLLM server (you can switch between distilled and normal)')
    
    # Processing options
    parser.add_argument('--crop_fraction', type=float, default=0.5,
                        help='Fraction of original image size to use for focused crop (default: 0.5)')
    parser.add_argument('--temp_dir', type=str, default='./temp_ablation',
                        help='Temporary directory for image processing')
    parser.add_argument('--output_dir', type=str, default='./debug',
                        help='Output directory for results')
    parser.add_argument('--max_retries', type=int, default=3,
                        help='Maximum number of retries per model per sample')
    
    # Filtering options
    parser.add_argument('--dataset_filter', type=str, choices=['loveda', 'isaid'], default=None,
                        help='Filter samples by source dataset: loveda (L*), isaid (P*)')
    
    return parser.parse_args()

def decode_rle_and_get_bbox(segmentation_str):
    """Decode RLE segmentation and compute bounding box."""
    try:
        seg_dict = eval(segmentation_str)
        rle = {
            'size': seg_dict['size'],
            'counts': seg_dict['counts'].encode('utf-8')
        }
        mask = mask_utils.decode(rle)
        
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return None
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        bbox = {
            'xmin': int(cmin),
            'ymin': int(rmin),
            'xmax': int(cmax + 1),
            'ymax': int(rmax + 1)
        }
        
        return bbox
        
    except Exception as e:
        print(f"Error decoding RLE segmentation: {e}")
        return None

def decode_rle_and_get_individual_bboxes(segmentation_str, min_area=10):
    """Decode RLE segmentation and compute individual bounding boxes for each connected component."""
    try:
        seg_dict = eval(segmentation_str)
        rle = {
            'size': seg_dict['size'],
            'counts': seg_dict['counts'].encode('utf-8')
        }
        mask = mask_utils.decode(rle)
        
        labeled_mask = label(mask, connectivity=2)
        regions = regionprops(labeled_mask)
        
        bboxes = []
        for region in regions:
            if region.area < min_area:
                continue
                
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

def decode_rle_and_get_mask(segmentation_str):
    """Decode RLE segmentation string to binary mask."""
    try:
        seg_dict = eval(segmentation_str)
        rle = {
            'size': seg_dict['size'],
            'counts': seg_dict['counts'].encode('utf-8')
        }
        mask = mask_utils.decode(rle)
        return mask
        
    except Exception as e:
        print(f"Error decoding RLE segmentation to mask: {e}")
        return None

def create_prompt(object_name, original_expressions, image_mode="bbox_dual"):
    """Create the detailed prompt for LLM models with dual image support."""
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
    """Create visualization with bounding box(es) and save to file."""
    try:
        image = Image.open(image_path).convert('RGB')
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 12))
        ax.imshow(image)
        
        # Handle both single bbox and list of bboxes
        if isinstance(bboxes, dict):
            bboxes = [bboxes]
        
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax']
            width, height = x2 - x1, y2 - y1
            
            rect = patches.Rectangle((x1, y1), width, height, 
                                   linewidth=1, edgecolor='red', facecolor='none')
            ax.add_patch(rect)
        
        ax.set_xlim(0, image.width)
        ax.set_ylim(image.height, 0)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        # Create a simple fallback image
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.text(0.5, 0.5, f"Error loading image:\n{e}", 
                ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)

def create_mask_overlay(image_path, segmentation_str, output_path, mask_color=(255, 0, 0), mask_alpha=0.3):
    """Create an image with red mask overlay from RLE segmentation data."""
    img = Image.open(image_path).convert('RGB')
    img_width, img_height = img.size
    
    mask = decode_rle_and_get_mask(segmentation_str)
    if mask is None:
        print(f"Error: Could not decode segmentation mask for {image_path}")
        img.save(output_path)
        return
    
    mask_array = np.array(mask, dtype=bool)
    mask_overlay = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    mask_overlay[mask_array] = mask_color
    
    img_array = np.array(img)
    blended = img_array.copy()
    for c in range(3):  # RGB channels
        blended[mask_array, c] = (
            (1 - mask_alpha) * img_array[mask_array, c] + 
            mask_alpha * mask_overlay[mask_array, c]
        ).astype(np.uint8)
    
    result_img = Image.fromarray(blended)
    result_img.save(output_path)

def compute_centroid_from_bboxes(bboxes):
    """Compute centroid from a list of bounding boxes or a single bbox."""
    if isinstance(bboxes, dict):
        center_x = (bboxes['xmin'] + bboxes['xmax']) / 2
        center_y = (bboxes['ymin'] + bboxes['ymax']) / 2
        return center_x, center_y
    elif isinstance(bboxes, list):
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
    
    center_x, center_y = compute_centroid_from_bboxes(bboxes)
    
    min_dimension = min(img_width, img_height)
    target_size = min_dimension * image_fraction
    
    half_size = target_size / 2
    crop_left = int(center_x - half_size)
    crop_right = int(center_x + half_size)
    crop_top = int(center_y - half_size)
    crop_bottom = int(center_y + half_size)
    
    pad_left = max(0, -crop_left)
    pad_right = max(0, crop_right - img_width)
    pad_top = max(0, -crop_top)
    pad_bottom = max(0, crop_bottom - img_height)
    
    crop_left = max(0, crop_left)
    crop_right = min(img_width, crop_right)
    crop_top = max(0, crop_top)
    crop_bottom = min(img_height, crop_bottom)
    
    cropped = img.crop((crop_left, crop_top, crop_right, crop_bottom))
    
    square_size = int(target_size)
    if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
        canvas = Image.new('RGB', (square_size, square_size), (0, 0, 0))
        paste_x = pad_left
        paste_y = pad_top
        canvas.paste(cropped, (paste_x, paste_y))
        result = canvas
    else:
        result = cropped
    
    result = result.resize((crop_size, crop_size), Image.Resampling.LANCZOS)
    
    return result

def encode_base64_content_from_file(image_path):
    """Encode image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

def setup_clients(args):
    """Setup OpenAI clients for both models."""
    clients = {}
    
    # Setup OpenAI O3 client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Warning: No OpenAI API key found in .env file. O3 model will be skipped.")
        clients['o3'] = None
    else:
        clients['o3'] = OpenAI(
            api_key=api_key,
            base_url="https://api.openai.com/v1",
        )
    
    # Setup Gemma3 client (hardcoded URL)
    clients['gemma3'] = OpenAI(
        base_url="http://0.0.0.0:8000/v1",
        api_key="token-abc123",  # vLLM doesn't require real API key
    )
    
    return clients

def parse_annotations(annotation_path):
    """Extract annotations from XML file and return objects/groups with expressions."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    items_to_process = []
    
    # Parse individual objects
    for obj in root.findall('object'):
        category = None
        if obj.find('name') is not None:
            category = obj.find('name').text
        elif obj.find('n') is not None:
            category = obj.find('n').text
        else:
            category = "unknown"
        
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
        
        # Get existing expressions
        expressions = []
        if obj.find('expressions') is not None:
            for expr in obj.findall('expressions/expression'):
                if expr.text is not None and expr.text.strip():
                    expressions.append(expr.text.strip())
        
        # Only process objects with bbox and expressions
        if bbox and expressions:
            items_to_process.append({
                'id': obj_id,
                'category': category,
                'bbox': bbox,
                'expressions': expressions,
                'type': 'object'
            })
    
    # Parse groups
    groups_elem = root.find('groups')
    if groups_elem is not None:
        for group in groups_elem.findall('group'):
            group_id = group.find('id').text if group.find('id') is not None else "unknown"
            category = group.find('category').text if group.find('category') is not None else "unknown"
            size = group.find('size').text if group.find('size') is not None else "unknown"
            
            # Get segmentation data
            segmentation = None
            segmentation_elem = group.find('segmentation')
            if segmentation_elem is not None and segmentation_elem.text:
                segmentation = segmentation_elem.text
            
            # Get bboxes from segmentation
            bboxes = []
            if segmentation:
                individual_bboxes = decode_rle_and_get_individual_bboxes(segmentation)
                if individual_bboxes:
                    bboxes = individual_bboxes
                else:
                    single_bbox = decode_rle_and_get_bbox(segmentation)
                    if single_bbox:
                        bboxes = [single_bbox]
            
            # Get existing expressions
            expressions = []
            if group.find('expressions') is not None:
                for expr in group.findall('expressions/expression'):
                    if expr.text is not None and expr.text.strip():
                        expressions.append(expr.text.strip())
            
            # Only process groups with segmentation/bboxes and expressions
            if segmentation and expressions:
                items_to_process.append({
                    'id': group_id,
                    'category': category,
                    'size': size,
                    'segmentation': segmentation,
                    'bboxes': bboxes,
                    'expressions': expressions,
                    'type': 'group'
                })
    
    return {
        'filename': filename,
        'items': items_to_process
    }

def get_random_samples(args):
    """Get random samples by picking random files, not parsing everything."""
    random.seed(args.seed)
    
    # Build paths
    dataset_root = os.path.abspath(args.dataset_root)
    images_dir = os.path.join(dataset_root, 'patches', args.split, 'images')
    annotations_dir = os.path.join(dataset_root, 'patches_rules_expressions_unique', args.split, 'annotations')
    
    if not os.path.exists(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not os.path.exists(annotations_dir):
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    
    # Get all annotation files
    annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    
    # Apply dataset filter if specified
    if args.dataset_filter:
        if args.dataset_filter == 'loveda':
            prefix = 'L'
            dataset_name = 'LoveDA'
        elif args.dataset_filter == 'isaid':
            prefix = 'P'
            dataset_name = 'iSAID'
        
        filtered_files = [f for f in annotation_files if f.startswith(prefix)]
        print(f"Dataset filter applied: {dataset_name} ({prefix}*)")
        print(f"Files before filtering: {len(annotation_files)}")
        print(f"Files after filtering: {len(filtered_files)}")
        
        annotation_files = filtered_files
    
    print(f"Found {len(annotation_files)} annotation files")
    
    # Shuffle files and pick enough to get our samples
    random.shuffle(annotation_files)
    
    selected_items = []
    files_checked = 0
    
    for annotation_file in annotation_files:
        if len(selected_items) >= args.num_samples:
            break
            
        files_checked += 1
        annotation_path = os.path.join(annotations_dir, annotation_file)
        image_path = os.path.join(images_dir, annotation_file.replace('.xml', '.png'))
        
        if not os.path.exists(image_path):
            continue
        
        # Parse only this one file
        annotation_data = parse_annotations(annotation_path)
        
        # Pick first valid item from this file
        for item in annotation_data['items']:
            selected_items.append({
                'item': item,
                'image_path': image_path,
                'annotation_path': annotation_path,
                'filename': annotation_data['filename']
            })
            break  # Only take one item per file
    
    print(f"Checked {files_checked} files, selected {len(selected_items)} items for processing")
    
    return selected_items

def call_model_api(client, model_name, system_prompt, user_prompt, main_image_base64, second_image_base64, is_o3=False):
    """Make API call to a model and return response with token usage."""
    try:
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{main_image_base64}"},
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{second_image_base64}"},
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ]
        
        # Different parameters for O3 vs vLLM
        if is_o3:
            completion = client.chat.completions.create(
                model="o3",
                temperature=1.0,
                messages=messages,
                reasoning_effort="high",
            )
        else:
            completion = client.chat.completions.create(
                messages=messages,
                model=model_name,
                max_tokens=2048,
                temperature=0.8,
            )
        
        # Extract response and token usage
        response_text = completion.choices[0].message.content.strip()
        
        # Extract token usage information
        usage_info = {}
        if hasattr(completion, 'usage') and completion.usage:
            if hasattr(completion.usage, 'prompt_tokens'):
                usage_info['input_tokens'] = completion.usage.prompt_tokens
            if hasattr(completion.usage, 'completion_tokens'):
                usage_info['output_tokens'] = completion.usage.completion_tokens
            if hasattr(completion.usage, 'total_tokens'):
                usage_info['total_tokens'] = completion.usage.total_tokens
        
        return {
            'raw_output': response_text,
            'success': True,
            'error': None,
            **usage_info
        }
        
    except Exception as e:
        return {
            'raw_output': None,
            'success': False,
            'error': str(e),
            'input_tokens': None,
            'output_tokens': None,
            'total_tokens': None
        }

def process_single_sample(sample_data, clients, args, sample_idx):
    """Process a single sample through all three models."""
    item = sample_data['item']
    image_path = sample_data['image_path']
    filename = sample_data['filename']
    
    print(f"\nProcessing sample {sample_idx + 1}/{args.num_samples}")
    print(f"  Image: {filename}")
    print(f"  Target: {item['type']} {item['id']} ({item['category']})")
    
    # Determine processing mode based on image name and item type
    image_name = os.path.basename(image_path).split('.')[0]
    if image_name.startswith('L'):  # LoveDA
        if item['type'] == 'group':
            mode = 'mask_dual'
            mode_desc = 'LoveDA group with mask overlay'
        else:
            mode = 'bbox_dual'
            mode_desc = 'LoveDA object with bbox and focused crop'
    elif image_name.startswith('P'):  # iSAID
        mode = 'bbox_dual'
        mode_desc = 'iSAID with bbox and focused crop'
    else:
        mode = 'bbox_dual'
        mode_desc = 'Unknown dataset - using bbox and focused crop'
    
    print(f"  Mode: {mode_desc}")
    
    # Create unique timestamp for this sample
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sample_id = f"sample_{timestamp}_{image_name}_{item['id']}"
    
    # Create temporary visualization paths
    viz_filename = f"{sample_id}_main.png"
    second_filename = f"{sample_id}_second.png"
    viz_path = os.path.join(args.temp_dir, viz_filename)
    second_image_path = os.path.join(args.temp_dir, second_filename)
    
    try:
        # Create visualizations based on mode
        if mode == 'mask_dual':
            # Save clean image first (just a copy)
            shutil.copy2(image_path, second_image_path)
            
            # Create and save mask overlay
            if 'segmentation' not in item:
                raise Exception("No segmentation data found for mask overlay mode")
            create_mask_overlay(image_path, item['segmentation'], viz_path)
            
        else:  # bbox_dual mode
            # Create visualization with bounding box(es)
            if item['type'] == 'group':
                bboxes_to_visualize = item['bboxes']
            else:
                bboxes_to_visualize = item['bbox']
            
            # Create bbox visualization
            visualize_and_save_object(image_path, bboxes_to_visualize, viz_path)
            
            # Create focused crop
            focused_crop = create_focused_crop(image_path, bboxes_to_visualize, image_fraction=args.crop_fraction)
            focused_crop.save(second_image_path)
        
        # Create prompts
        system_prompt, user_prompt = create_prompt(item['category'], item['expressions'], image_mode=mode)
        
        # Convert both images to base64
        main_image_base64 = encode_base64_content_from_file(viz_path)
        second_image_base64 = encode_base64_content_from_file(second_image_path)
        
        if main_image_base64 is None or second_image_base64 is None:
            raise Exception("Failed to encode images to base64")
        
        # Process with both models
        results = {}
        
        # Model 1: OpenAI O3
        print("  Processing with O3...")
        if clients['o3'] is not None:
            for attempt in range(args.max_retries):
                result = call_model_api(
                    clients['o3'], "o3", system_prompt, user_prompt,
                    main_image_base64, second_image_base64, is_o3=True
                )
                if result['success']:
                    break
                print(f"    O3 attempt {attempt + 1} failed: {result['error']}")
            results['o3_result'] = result
        else:
            results['o3_result'] = {
                'raw_output': None,
                'success': False,
                'error': 'O3 client not available (no API key)',
                'input_tokens': None,
                'output_tokens': None,
                'total_tokens': None
            }
        
        # Model 2: Gemma3 (configurable model name)
        print(f"  Processing with Gemma3 ({args.gemma3_model_name})...")
        for attempt in range(args.max_retries):
            result = call_model_api(
                clients['gemma3'], args.gemma3_model_name, system_prompt, user_prompt,
                main_image_base64, second_image_base64, is_o3=False
            )
            if result['success']:
                break
            print(f"    Gemma3 attempt {attempt + 1} failed: {result['error']}")
        results['gemma3_result'] = result
        
        # Move visualization files to output directory
        output_viz_path = os.path.join(args.output_dir, viz_filename)
        output_second_path = os.path.join(args.output_dir, second_filename)
        shutil.move(viz_path, output_viz_path)
        shutil.move(second_image_path, output_second_path)
        
        # Prepare final result
        final_result = {
            'sample_id': sample_id,
            'timestamp': timestamp,
            'image_info': {
                'image_path': os.path.relpath(image_path, args.dataset_root),
                'filename': filename,
                'dataset_type': 'LoveDA' if image_name.startswith('L') else 'iSAID'
            },
            'target_info': {
                'id': item['id'],
                'type': item['type'],
                'category': item['category'],
                'expressions': item['expressions']
            },
            'processing_mode': mode,
            'mode_description': mode_desc,
            'visualization_files': {
                'main_image': viz_filename,
                'second_image': second_filename
            },
            **results
        }
        
        # Add type-specific target information
        if item['type'] == 'group':
            final_result['target_info']['size'] = item['size']
            final_result['target_info']['num_bboxes'] = len(item['bboxes'])
        else:
            final_result['target_info']['bbox'] = item['bbox']
        
        # Save result to JSON file
        output_json_path = os.path.join(args.output_dir, f"{sample_id}.json")
        with open(output_json_path, 'w') as f:
            json.dump(final_result, f, indent=2)
        
        print(f"  ✓ Completed sample {sample_idx + 1}")
        print(f"    Results saved: {sample_id}.json")
        
        # Print success status for each model
        for model_name, result_key in [('O3', 'o3_result'), ('Gemma3', 'gemma3_result')]:
            result = results[result_key]
            if result['success']:
                tokens_info = ""
                if result.get('input_tokens') and result.get('output_tokens'):
                    tokens_info = f" (in: {result['input_tokens']}, out: {result['output_tokens']})"
                print(f"    {model_name}: ✓ Success{tokens_info}")
            else:
                print(f"    {model_name}: ❌ Failed - {result['error']}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Sample {sample_idx + 1} failed: {e}")
        return False
    
    finally:
        # Clean up temporary files if they still exist
        for temp_file in [viz_path, second_image_path]:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass

def main():
    """Main function."""
    args = parse_arguments()
    
    print("🚀 Starting Ablation Study")
    print(f"Dataset: {args.dataset_root}")
    print(f"Split: {args.split}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Gemma3 model: {args.gemma3_model_name}")
    print(f"Random seed: {args.seed}")
    print(f"Output directory: {args.output_dir}")
    
    # Create directories
    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\nSetting up model clients...")
    clients = setup_clients(args)
    
    # Test client connections
    print("Testing model connections:")
    for model_name, client in clients.items():
        if client is not None:
            print(f"  {model_name}: ✓ Connected")
        else:
            print(f"  {model_name}: ❌ Not available")
    
    print(f"\nGetting random samples...")
    samples = get_random_samples(args)
    
    if not samples:
        print("❌ No samples found! Check dataset path and filter settings.")
        return
    
    # Process samples
    successful_samples = 0
    failed_samples = 0
    
    print(f"\n🔄 Processing {len(samples)} samples...")
    
    for i, sample in enumerate(samples):
        if process_single_sample(sample, clients, args, i):
            successful_samples += 1
        else:
            failed_samples += 1
    
    # Final summary
    print(f"\n🎉 Ablation Study Complete!")
    print(f"   Total samples: {len(samples)}")
    print(f"   Successful: {successful_samples}")
    print(f"   Failed: {failed_samples}")
    print(f"   Results saved in: {args.output_dir}")
    
    # Clean up temp directory
    if os.path.exists(args.temp_dir):
        shutil.rmtree(args.temp_dir)
        print(f"   Cleaned up temporary directory")

if __name__ == "__main__":
    main()