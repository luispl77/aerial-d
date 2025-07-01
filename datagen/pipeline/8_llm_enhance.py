#!/usr/bin/env python3
"""
Step 8: LLM Enhancement using Fine-tuned Gemma 3

This script processes the entire dataset and uses the fine-tuned Gemma 3 model
to enhance expressions in the XML annotations directly. It modifies the XML files
in-place by adding enhanced expressions to existing objects and groups.
"""

import os
import sys
import json
import xml.etree.ElementTree as ET
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
from tqdm import tqdm
import tempfile
import shutil
import numpy as np
import random
import multiprocessing as mp
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pycocotools import mask as mask_utils
from skimage.measure import label, regionprops

# Add the LLM directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'llm'))

try:
    from transformers import AutoProcessor, AutoModelForImageTextToText
    import torch
    import time
except ImportError as e:
    print(f"Error importing transformers/torch: {e}")
    print("Please install the required packages:")
    print("pip install torch transformers pillow matplotlib pycocotools scikit-image")
    sys.exit(1)

# Disable torch compilation warnings
os.environ["TORCH_COMPILE"] = "0"
torch._dynamo.config.disable = True
torch.backends.cudnn.enabled = False
torch.set_float32_matmul_precision('high')

# Enhancement constants
NUM_ENHANCED = 1  # Number of enhanced expressions per original
NUM_UNIQUE = 2    # Number of unique expressions to generate

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhance dataset expressions using fine-tuned Gemma 3')
    parser.add_argument('--dataset_root', type=str, default='./dataset',
                        help='Root path to the dataset directory')
    parser.add_argument('--model_path', type=str, default='../llm/merged_model_4b',
                        help='Path to the fine-tuned Gemma 3 model')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default='both',
                        help='Which dataset split to process')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of annotation files to process (for testing)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--temp_dir', type=str, default='.',
                        help='Temporary directory for visualization images')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic file ordering (default: 42)')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel workers (default: 6)')
    parser.add_argument('--dry_run', action='store_true',
                        help='Run without actually modifying the XML files')
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

def create_prompt(object_name, original_expressions):
    """Create the detailed prompt for Gemma 3."""
    formatted_expressions = "\n".join([f"- {expr}" for expr in original_expressions])
    
    system_prompt = (
        "You are an expert at creating natural language descriptions for objects and groups in aerial imagery. "
        "Your task is to help create diverse and precise referring expressions for the target highlighted with a red bounding box. "
        "The target may be a single object or a group/collection of multiple objects.\n\n"
        
        "IMPORTANT GUIDELINES:\n"
        "- If the original expressions refer to 'all', 'group of', or multiple objects, maintain this collective reference\n"
        "- If working with a group, use plural forms and consider the spatial distribution of the entire collection\n"
        "- If working with a single object, focus on that specific instance\n"
        "- Always preserve the scope and meaning of the original expressions\n"
        "- NEVER reference red boxes or markings in your expressions\n\n"
        
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
        f"analyze the spatial context for uniqueness factors, and generate new unique expressions for the target "
        "(highlighted in red)."
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

def get_available_gpus():
    """Get list of available CUDA devices."""
    if not torch.cuda.is_available():
        return [0]  # Fallback to CPU or single device
    
    num_gpus = torch.cuda.device_count()
    return list(range(num_gpus))

def setup_model_and_processor(model_path, gpu_id=0):
    """Load the fine-tuned model and processor on specific GPU."""
    print(f"Loading fine-tuned model from: {model_path} on GPU {gpu_id}")
    
    # Set the device for this process
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(gpu_id)
    
    # Load model without device_map first, then move to device
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=None,  # Don't use device_map in multiprocessing
        attn_implementation="sdpa"
    )
    
    # Move model to the specific device
    model = model.to(device)
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    print(f"Model loaded successfully on GPU {gpu_id}")
    return model, processor

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
    
    return None

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
                'element': obj,
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
            
            # Get bboxes from segmentation
            bboxes = []
            segmentation_elem = group.find('segmentation')
            if segmentation_elem is not None and segmentation_elem.text:
                individual_bboxes = decode_rle_and_get_individual_bboxes(segmentation_elem.text)
                if individual_bboxes:
                    bboxes = individual_bboxes
                else:
                    single_bbox = decode_rle_and_get_bbox(segmentation_elem.text)
                    if single_bbox:
                        bboxes = [single_bbox]
            
            # Get existing expressions
            expressions = []
            if group.find('expressions') is not None:
                for expr in group.findall('expressions/expression'):
                    if expr.text is not None and expr.text.strip():
                        expressions.append(expr.text.strip())
            
            # Only process groups with bboxes and expressions
            if bboxes and expressions:
                items_to_process.append({
                    'element': group,
                    'id': group_id,
                    'category': category,
                    'size': size,
                    'bboxes': bboxes,
                    'expressions': expressions,
                    'type': 'group'
                })
    
    return {
        'filename': filename,
        'tree': tree,
        'root': root,
        'items': items_to_process
    }

def process_item_with_llm(item, image_path, model, processor, temp_dir, max_retries=3):
    """Process a single item (object or group) with the LLM, with retry logic."""
    retry_count = 0
    
    try:
        # Create temporary visualization
        viz_filename = f"temp_viz_{item['id']}_{item['type']}.png"
        viz_path = os.path.join(temp_dir, viz_filename)
        
        # Create visualization with bounding box(es)
        if item['type'] == 'group':
            bboxes_to_visualize = item['bboxes']
        else:
            bboxes_to_visualize = item['bbox']
        
        visualize_and_save_object(image_path, bboxes_to_visualize, viz_path)
        
        # Create prompt
        system_prompt, user_prompt = create_prompt(item['category'], item['expressions'])
        
        # Create messages format
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": viz_path},
                    {"type": "text", "text": user_prompt}
                ]
            }
        ]
        
        # Process inputs
        device = next(model.parameters()).device
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(device)
        
        # Retry loop for generation and parsing
        for attempt in range(max_retries + 1):  # 0, 1, 2, 3 (4 total attempts)
            try:
                # Generate response
                with torch.no_grad():
                    # Vary temperature slightly on retries to get different outputs
                    temperature = 0.8 + (attempt * 0.1)
                    output = model.generate(**inputs, max_new_tokens=2048, do_sample=True, temperature=temperature)
                
                # Decode response
                generated_content = processor.decode(output[0], skip_special_tokens=True)
                input_length = len(processor.decode(inputs['input_ids'][0], skip_special_tokens=True))
                generated_content = generated_content[input_length:].strip()
                
                # Parse JSON response
                enhanced_data = extract_and_parse_json(generated_content)
                if enhanced_data is not None:
                    # Success! Clean up and return
                    try:
                        os.remove(viz_path)
                    except:
                        pass
                    
                    if attempt > 0:
                        print(f"    ✓ Succeeded on retry {attempt}")
                    
                    return enhanced_data, retry_count
                
                # Parsing failed, increment retry count and try again
                if attempt < max_retries:
                    retry_count += 1
                    print(f"    ⚠ JSON parsing failed, retrying ({attempt + 1}/{max_retries})...")
                else:
                    # Final attempt failed
                    print(f"  ❌ Failed to parse JSON after {max_retries} retries for {item['type']} {item['id']}")
                    print(f"  Final response content:")
                    print("-" * 80)
                    print(generated_content)
                    print("-" * 80)
                    
            except Exception as gen_e:
                print(f"  Error during generation attempt {attempt + 1}: {gen_e}")
                if attempt < max_retries:
                    retry_count += 1
                    continue
        
        # Clean up visualization
        try:
            os.remove(viz_path)
        except:
            pass
        
        return None, retry_count
        
    except Exception as e:
        print(f"  Error processing {item['type']} {item['id']}: {e}")
        return None, retry_count

def is_file_already_processed(annotation_path):
    """Check if all objects/groups in the file already have enhanced and unique expressions."""
    try:
        annotation_data = parse_annotations(annotation_path)
        
        if not annotation_data['items']:
            return True  # No items to process, consider it processed
        
        for item in annotation_data['items']:
            expressions_elem = item['element'].find('expressions')
            if expressions_elem is None:
                return False  # No expressions at all
            
            # Check for enhanced and unique expressions
            has_enhanced = False
            has_unique = False
            
            for expr in expressions_elem.findall('expression'):
                expr_type = expr.get('type', '')
                if expr_type == 'enhanced':
                    has_enhanced = True
                elif expr_type == 'unique':
                    has_unique = True
            
            # If any item lacks both enhanced and unique expressions, file is not processed
            if not (has_enhanced and has_unique):
                return False
        
        return True  # All items have both enhanced and unique expressions
        
    except Exception as e:
        print(f"  Error checking if file is processed: {e}")
        return False  # Assume not processed if we can't check

def add_enhanced_expressions_to_xml(item, enhanced_data):
    """Add enhanced expressions to the XML element."""
    expressions_elem = item['element'].find('expressions')
    if expressions_elem is None:
        expressions_elem = ET.SubElement(item['element'], 'expressions')
    
    # Add enhanced expressions (variations of originals)
    if 'enhanced_expressions' in enhanced_data:
        for enhanced in enhanced_data['enhanced_expressions']:
            if 'variation' in enhanced and enhanced['variation'].strip():
                expr_elem = ET.SubElement(expressions_elem, 'expression')
                expr_elem.text = enhanced['variation'].strip()
                expr_elem.set('type', 'enhanced')
    
    # Add unique expressions
    if 'unique_expressions' in enhanced_data:
        for unique_expr in enhanced_data['unique_expressions']:
            if unique_expr.strip():
                expr_elem = ET.SubElement(expressions_elem, 'expression')
                expr_elem.text = unique_expr.strip()
                expr_elem.set('type', 'unique')

def process_annotation_file(annotation_path, images_dir, model, processor, temp_dir, dry_run=False):
    """Process a single annotation file."""
    try:
        # Parse annotations
        annotation_data = parse_annotations(annotation_path)
        
        if not annotation_data['items']:
            return 0, 0, 0, 0  # No items to process
        
        # Get corresponding image
        image_filename = annotation_data['filename']
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"  Warning: Image not found: {image_path}")
            return 0, 0, 0, 0
        
        processed_count = 0
        enhanced_count = 0
        total_retries = 0
        failed_items = 0
        
        # Process each item with expressions
        for item in annotation_data['items']:
            print(f"  Processing {item['type']} {item['id']} ({item['category']})")
            
            result = process_item_with_llm(item, image_path, model, processor, temp_dir)
            
            if result is None:
                enhanced_data, retry_count = None, 0
            else:
                enhanced_data, retry_count = result
            
            total_retries += retry_count
            
            if enhanced_data:
                if not dry_run:
                    add_enhanced_expressions_to_xml(item, enhanced_data)
                
                # Count enhancements
                num_enhanced = len(enhanced_data.get('enhanced_expressions', []))
                num_unique = len(enhanced_data.get('unique_expressions', []))
                enhanced_count += num_enhanced + num_unique
                
                print(f"    Added {num_enhanced} enhanced + {num_unique} unique expressions")
            else:
                failed_items += 1
                print(f"    ❌ Failed to process item after retries")
            
            processed_count += 1
        
        # Save modified XML
        if not dry_run and enhanced_count > 0:
            annotation_data['tree'].write(annotation_path, encoding='utf-8', xml_declaration=True)
            print(f"  Updated XML file with {enhanced_count} new expressions")
        
        return processed_count, enhanced_count, total_retries, failed_items
        
    except Exception as e:
        print(f"  Error processing annotation file: {e}")
        return 0, 0, 0, 0

def worker_process_files(worker_id, gpu_id, model, processor, annotation_file_paths, images_dir, temp_dir, dry_run):
    """Worker function to process a batch of annotation files on a specific GPU."""
    try:
        print(f"Worker {worker_id} starting processing on GPU {gpu_id}")
        
        # Create worker-specific temp directory
        worker_temp_dir = os.path.join(temp_dir, f"worker_{worker_id}")
        os.makedirs(worker_temp_dir, exist_ok=True)
        
        worker_stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'items_processed': 0,
            'expressions_enhanced': 0,
            'total_retries': 0,
            'failed_items': 0
        }
        
        for annotation_path in annotation_file_paths:
            annotation_file = os.path.basename(annotation_path)
            
            # Check if file is already fully processed
            if is_file_already_processed(annotation_path):
                worker_stats['files_skipped'] += 1
                print(f"Worker {worker_id}: Skipping {annotation_file} (already processed)")
                continue
            
            print(f"Worker {worker_id}: Processing {annotation_file}")
            items_processed, expressions_enhanced, retries, failed = process_annotation_file(
                annotation_path, images_dir, model, processor, worker_temp_dir, dry_run
            )
            
            if items_processed > 0:
                worker_stats['files_processed'] += 1
                worker_stats['items_processed'] += items_processed
                worker_stats['expressions_enhanced'] += expressions_enhanced
                worker_stats['total_retries'] += retries
                worker_stats['failed_items'] += failed
        
        # Cleanup worker temp directory
        try:
            shutil.rmtree(worker_temp_dir)
        except:
            pass
        
        print(f"Worker {worker_id} completed processing on GPU {gpu_id}")
        return worker_stats
        
    except Exception as e:
        print(f"Worker {worker_id} error: {e}")
        return {'files_processed': 0, 'files_skipped': 0, 'items_processed': 0, 'expressions_enhanced': 0, 'total_retries': 0, 'failed_items': 0}

def main():
    """Main processing function."""
    args = parse_arguments()
    
    # Set random seed for deterministic file ordering
    random.seed(args.seed)
    print(f"Using random seed: {args.seed}")
    
    # Setup paths
    dataset_root = os.path.abspath(args.dataset_root)
    model_path = os.path.abspath(args.model_path)
    
    print(f"Dataset root: {dataset_root}")
    print(f"Model path: {model_path}")
    
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset root not found: {dataset_root}")
        return
    
    if not os.path.exists(model_path):
        print(f"Error: Model path not found: {model_path}")
        return
    
    # Setup temporary directory
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp()
    
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Get available GPUs and setup workers
        available_gpus = get_available_gpus()
        num_workers = args.workers
        workers_per_gpu = num_workers / len(available_gpus)
        print(f"\nAvailable GPUs: {available_gpus}")
        print(f"Using {num_workers} workers ({workers_per_gpu:.1f} workers per GPU)")
        
        # Pre-load all models on their respective GPUs
        print(f"\n🔄 Pre-loading {num_workers} model instances...")
        worker_models = {}
        
        for worker_id in range(num_workers):
            gpu_id = available_gpus[worker_id % len(available_gpus)]
            print(f"Loading model for Worker {worker_id} on GPU {gpu_id}...")
            model, processor = setup_model_and_processor(model_path, gpu_id)
            worker_models[worker_id] = (model, processor)
        
        print(f"\n✅ All {num_workers} models loaded successfully!")
        print(f"⏳ Waiting 10 seconds for you to check nvidia-smi...")
        print("   You can run 'nvidia-smi' in another terminal to see GPU memory usage")
        
        for i in range(10, 0, -1):
            print(f"   Starting processing in {i} seconds...", end='\r')
            time.sleep(1)
        print("\n🚀 Starting processing!")
        
        # Determine splits to process
        splits = ['train', 'val'] if args.split == 'both' else [args.split]
        
        total_files = 0
        total_items = 0
        total_enhanced = 0
        total_retries = 0
        total_failures = 0
        
        for split in splits:
            print(f"\nProcessing {split} split...")
            
            # Setup paths for this split
            images_dir = os.path.join(dataset_root, 'patches', split, 'images')
            annotations_dir = os.path.join(dataset_root, 'patches_rules_expressions_unique', split, 'annotations')
            
            if not os.path.exists(images_dir):
                print(f"Warning: Images directory not found: {images_dir}")
                continue
            
            if not os.path.exists(annotations_dir):
                print(f"Warning: Annotations directory not found: {annotations_dir}")
                continue
            
            # Get annotation files and shuffle them for random processing
            annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
            random.shuffle(annotation_files)  # Randomize processing order
            
            if args.limit:
                annotation_files = annotation_files[:args.limit]
            
            print(f"Found {len(annotation_files)} annotation files in {split} (shuffled randomly)")
            
            # Create full paths for annotation files
            annotation_file_paths = [os.path.join(annotations_dir, f) for f in annotation_files]
            
            # Distribute files across workers
            files_per_worker = len(annotation_file_paths) // num_workers
            worker_file_batches = []
            
            for i in range(num_workers):
                start_idx = i * files_per_worker
                if i == num_workers - 1:  # Last worker gets remaining files
                    end_idx = len(annotation_file_paths)
                else:
                    end_idx = (i + 1) * files_per_worker
                
                worker_file_batches.append(annotation_file_paths[start_idx:end_idx])
            
            print(f"Distributing files across {num_workers} workers:")
            for i, batch in enumerate(worker_file_batches):
                print(f"  Worker {i}: {len(batch)} files")
            
            # Process files in parallel using ThreadPoolExecutor
            split_stats = {'files_processed': 0, 'files_skipped': 0, 'items_processed': 0, 'expressions_enhanced': 0, 'total_retries': 0, 'failed_items': 0}
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                # Submit worker tasks
                future_to_worker = {}
                for i in range(num_workers):
                    if worker_file_batches[i]:  # Only submit if worker has files to process
                        gpu_id = available_gpus[i % len(available_gpus)]
                        model, processor = worker_models[i]
                        future = executor.submit(
                            worker_process_files,
                            i, gpu_id, model, processor, worker_file_batches[i],
                            images_dir, temp_dir, args.dry_run
                        )
                        future_to_worker[future] = i
                
                # Collect results
                for future in as_completed(future_to_worker):
                    worker_id = future_to_worker[future]
                    try:
                        worker_stats = future.result()
                        split_stats['files_processed'] += worker_stats['files_processed']
                        split_stats['files_skipped'] += worker_stats['files_skipped']
                        split_stats['items_processed'] += worker_stats['items_processed']
                        split_stats['expressions_enhanced'] += worker_stats['expressions_enhanced']
                        split_stats['total_retries'] += worker_stats['total_retries']
                        split_stats['failed_items'] += worker_stats['failed_items']
                        print(f"Worker {worker_id} completed: {worker_stats}")
                    except Exception as e:
                        print(f"Worker {worker_id} failed: {e}")
            
            print(f"\n{split.capitalize()} split summary:")
            print(f"  Files processed: {split_stats['files_processed']}")
            print(f"  Files skipped (already processed): {split_stats['files_skipped']}")
            print(f"  Items processed: {split_stats['items_processed']}")
            print(f"  Expressions enhanced: {split_stats['expressions_enhanced']}")
            print(f"  Total retries: {split_stats['total_retries']}")
            print(f"  Failed items (after retries): {split_stats['failed_items']}")
            
            total_files += split_stats['files_processed']
            total_items += split_stats['items_processed']
            total_enhanced += split_stats['expressions_enhanced']
            total_retries += split_stats['total_retries']
            total_failures += split_stats['failed_items']
        
        print(f"\nTotal summary:")
        print(f"  Files processed: {total_files}")
        print(f"  Items processed: {total_items}")
        print(f"  Expressions enhanced: {total_enhanced}")
        print(f"  Total retries performed: {total_retries}")
        print(f"  Failed items (after max retries): {total_failures}")
        
        # Calculate success/retry statistics
        if total_items > 0:
            success_rate = ((total_items - total_failures) / total_items) * 100
            retry_rate = (total_retries / total_items) * 100
            print(f"\nRetry Statistics:")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Retry rate: {retry_rate:.1f}% (avg {retry_rate/100:.2f} retries per item)")
            if total_retries > 0:
                recovery_rate = ((total_retries - total_failures) / total_retries) * 100 if total_retries > 0 else 0
                print(f"  Recovery rate: {recovery_rate:.1f}% (retries that succeeded)")
        
        if args.dry_run:
            print("\nDry run completed - no files were modified")
        else:
            print("\nEnhancement completed successfully!")
    
    finally:
        # Clean up temporary directory if we created it (worker temp dirs are cleaned up individually)
        pass

if __name__ == "__main__":
    main() 