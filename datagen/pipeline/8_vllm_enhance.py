#!/usr/bin/env python3
"""
Step 8: LLM Enhancement using Fine-tuned Gemma 3 with vLLM Server

This script processes the entire dataset and uses the fine-tuned Gemma 3 model
via OpenAI-compatible vLLM server for fast inference to enhance expressions in the XML annotations 
directly. It modifies the XML files in-place by adding enhanced expressions to 
existing objects and groups.
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
import time
from datetime import datetime, timedelta
from pycocotools import mask as mask_utils
from skimage.measure import label, regionprops
import base64
from io import BytesIO
from typing import List, Dict, Any, Optional, Tuple
import concurrent.futures
import threading

# Add the LLM directory to Python path for imports
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'llm'))

try:
    from openai import OpenAI
    import requests
    import time
except ImportError as e:
    print(f"Error importing required packages: {e}")
    print("Please install the required packages:")
    print("pip install openai requests pillow matplotlib pycocotools scikit-image")
    sys.exit(1)

# Enhancement constants
NUM_ENHANCED = 1  # Number of enhanced expressions per original
NUM_UNIQUE = 2    # Number of unique expressions to generate

# Thread-safe print lock
print_lock = threading.Lock()

def thread_safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)

class ProgressTracker:
    """Simple progress tracker for monitoring processing."""
    
    def __init__(self, total_files):
        self.total_files = total_files
        self.start_time = time.time()
        self.stats = {
            'files_processed': 0,
            'files_skipped': 0,
            'items_processed': 0,
            'expressions_enhanced': 0,
            'total_retries': 0,
            'failed_items': 0
        }
        
    def update_stats(self, files_processed=0, files_skipped=0, items_processed=0, 
                    expressions_enhanced=0, total_retries=0, failed_items=0):
        """Update progress statistics."""
        self.stats['files_processed'] += files_processed
        self.stats['files_skipped'] += files_skipped
        self.stats['items_processed'] += items_processed
        self.stats['expressions_enhanced'] += expressions_enhanced
        self.stats['total_retries'] += total_retries
        self.stats['failed_items'] += failed_items
    
    def get_progress_report(self):
        """Get current progress report with rates and ETA."""
        current_time = time.time()
        elapsed_time = current_time - self.start_time
        elapsed_minutes = elapsed_time / 60.0
        
        files_completed = self.stats['files_processed'] + self.stats['files_skipped']
        files_remaining = self.total_files - files_completed
        
        # Calculate rates (only based on actually processed files, not skipped)
        if elapsed_minutes > 0:
            processed_files_per_min = self.stats['files_processed'] / elapsed_minutes
            items_per_min = self.stats['items_processed'] / elapsed_minutes
        else:
            processed_files_per_min = 0
            items_per_min = 0
        
        # Calculate ETA based on actual processing rate
        eta_minutes = 0
        eta_str = "Calculating..."
        
        # All remaining files need processing (skipped files were already processed in previous runs)
        if processed_files_per_min > 0 and files_remaining > 0:
            eta_minutes = files_remaining / processed_files_per_min
            eta_time = datetime.now() + timedelta(minutes=eta_minutes)
            eta_str = eta_time.strftime("%H:%M:%S")
        elif files_remaining <= 0:
            eta_str = "Complete!"
        
        # Calculate skip ratio just for informational purposes (about past work)
        total_checked = self.stats['files_processed'] + self.stats['files_skipped']
        if total_checked > 0:
            skip_ratio = self.stats['files_skipped'] / total_checked
        else:
            skip_ratio = 0.0
        
        # Calculate completion percentage
        completion_pct = (files_completed / self.total_files * 100) if self.total_files > 0 else 0
        
        return {
            'elapsed_minutes': elapsed_minutes,
            'files_completed': files_completed,
            'files_remaining': files_remaining,
            'completion_pct': completion_pct,
            'processed_files_per_min': processed_files_per_min,
            'items_per_min': items_per_min,
            'eta_minutes': eta_minutes,
            'eta_str': eta_str,
            'skip_ratio': skip_ratio,
            'stats': self.stats
        }
    
    def print_progress(self):
        """Print current progress."""
        report = self.get_progress_report()
        
        print(f"\n📊 Progress Update [{datetime.now().strftime('%H:%M:%S')}]:")
        print(f"   Files: {report['files_completed']}/{self.total_files} "
              f"({report['completion_pct']:.1f}%) - {report['files_remaining']} remaining")
        print(f"   Processed: {report['stats']['files_processed']:,} | "
              f"Skipped: {report['stats']['files_skipped']:,}")
        print(f"   Items processed: {report['stats']['items_processed']:,}")
        print(f"   Expressions enhanced: {report['stats']['expressions_enhanced']:,}")
        print(f"   Processing rate: {report['processed_files_per_min']:.1f} files/min, "
              f"{report['items_per_min']:.1f} items/min")
        print(f"   Elapsed: {report['elapsed_minutes']:.1f} min | "
              f"ETA: {report['eta_str']} ({report['eta_minutes']:.1f} min remaining)")
        if report['skip_ratio'] > 0:
            print(f"   Already processed: {report['skip_ratio']*100:.1f}% of files checked so far")
        if report['stats']['total_retries'] > 0:
            success_rate = ((report['stats']['items_processed'] - report['stats']['failed_items']) / 
                          report['stats']['items_processed'] * 100) if report['stats']['items_processed'] > 0 else 100
            print(f"   Success rate: {success_rate:.1f}% | Retries: {report['stats']['total_retries']:,} | "
                  f"Failures: {report['stats']['failed_items']:,}")
        print("─" * 70)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhance dataset expressions using fine-tuned Gemma 3 with vLLM Server')
    parser.add_argument('--dataset_root', type=str, default='./dataset',
                        help='Root path to the dataset directory')
    parser.add_argument('--server_url', type=str, default='http://localhost:8000/v1',
                        help='URL of the vLLM server (default: http://localhost:8000/v1)')
    parser.add_argument('--model_name', type=str, default='../llm/merged_model_4b/',
                        help='Model name to use with the server')
    parser.add_argument('--split', type=str, choices=['train', 'val', 'both'], default='both',
                        help='Which dataset split to process')
    parser.add_argument('--limit', type=int, default=None,
                        help='Limit the number of annotation files to process (for testing)')
    parser.add_argument('--temp_dir', type=str, default='.',
                        help='Temporary directory for visualization images')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for deterministic file ordering (default: 42)')
    parser.add_argument('--progress_interval', type=int, default=3,
                        help='Progress update interval in files (default: 50). Use smaller values for more frequent updates.')
    parser.add_argument('--threads', type=int, default=2,
                        help='Number of threads to use for processing items within each file (default: 2)')
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



def setup_openai_client(server_url):
    """Initialize OpenAI client for vLLM server."""
    print(f"Setting up OpenAI client for server: {server_url}")
    
    client = OpenAI(
        base_url=server_url,
        api_key="token-abc123",  # vLLM server doesn't require a real API key
    )
    
    print("OpenAI client setup successfully!")
    return client

def encode_base64_content_from_file(image_path):
    """Encode image file to base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"Error encoding image to base64: {e}")
        return None

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

def process_single_item(client, item, image_path, temp_dir, model_name, max_retries=3):
    """Process a single item with OpenAI client."""
    viz_path = None
    
    try:
        # Create temporary visualization
        viz_filename = f"temp_viz_{item['id']}_{item['type']}_{time.time()}.png"
        viz_path = os.path.join(temp_dir, viz_filename)
        
        # Create visualization with bounding box(es)
        if item['type'] == 'group':
            bboxes_to_visualize = item['bboxes']
        else:
            bboxes_to_visualize = item['bbox']
        
        visualize_and_save_object(image_path, bboxes_to_visualize, viz_path)
        
        # Create prompt
        system_prompt, user_prompt = create_prompt(item['category'], item['expressions'])
        
        # Convert image to base64
        image_base64 = encode_base64_content_from_file(viz_path)
        if image_base64 is None:
            raise Exception("Failed to encode image to base64")
        
        # Process with retries
        for attempt in range(max_retries + 1):
            try:
                # Format messages for OpenAI API
                messages = [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{image_base64}"},
                            },
                        ],
                    }
                ]
                
                # Make API call
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model_name,
                    max_tokens=2048,
                    temperature=0.8,
                )
                
                generated_text = chat_completion.choices[0].message.content.strip()
                
                # Parse JSON response
                enhanced_data = extract_and_parse_json(generated_text)
                if enhanced_data:
                    # Success
                    return {
                        'item': item,
                        'enhanced_data': enhanced_data,
                        'success': True,
                        'retries': attempt
                    }
                else:
                    raise Exception(f"Failed to parse JSON: {generated_text[:200]}...")
                
            except Exception as e:
                thread_safe_print(f"    Error processing {item['type']} {item['id']} (attempt {attempt + 1}): {e}")
                if attempt == max_retries:
                    # Final attempt failed
                    return {
                        'item': item,
                        'enhanced_data': None,
                        'success': False,
                        'retries': attempt,
                        'error': str(e)
                    }
                time.sleep(1)  # Brief delay before retry
    
    except Exception as e:
        thread_safe_print(f"    Error setting up processing for {item['type']} {item['id']}: {e}")
        return {
            'item': item,
            'enhanced_data': None,
            'success': False,
            'retries': 0,
            'error': str(e)
        }
    
    finally:
        # Always clean up the temporary visualization file
        if viz_path and os.path.exists(viz_path):
            try:
                os.remove(viz_path)
            except:
                pass  # Ignore cleanup errors

def process_annotation_file(annotation_path, images_dir, client, model_name, temp_dir, num_threads=2, dry_run=False):
    """Process a single annotation file using OpenAI client with threading."""
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
        
        # Process items using ThreadPoolExecutor
        items = annotation_data['items']
        
        print(f"  Processing {len(items)} items with {num_threads} threads")
        
        # Use ThreadPoolExecutor to process items concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            future_to_item = {
                executor.submit(process_single_item, client, item, image_path, temp_dir, model_name): item 
                for item in items
            }
            
            # Process completed tasks as they finish
            for i, future in enumerate(concurrent.futures.as_completed(future_to_item)):
                item = future_to_item[future]
                
                try:
                    result = future.result()
                    
                    total_retries += result['retries']
                    
                    if result['success'] and result['enhanced_data']:
                        if not dry_run:
                            add_enhanced_expressions_to_xml(item, result['enhanced_data'])
                        
                        # Count enhancements
                        num_enhanced = len(result['enhanced_data'].get('enhanced_expressions', []))
                        num_unique = len(result['enhanced_data'].get('unique_expressions', []))
                        enhanced_count += num_enhanced + num_unique
                        
                        thread_safe_print(f"    ✓ {item['type']} {item['id']}: Added {num_enhanced} enhanced + {num_unique} unique expressions")
                    else:
                        failed_items += 1
                        thread_safe_print(f"    ❌ {item['type']} {item['id']}: {result.get('error', 'Unknown error')}")
                    
                    processed_count += 1
                    
                except Exception as e:
                    failed_items += 1
                    thread_safe_print(f"    ❌ {item['type']} {item['id']}: Thread execution error: {e}")
                    processed_count += 1
        
        # Save modified XML
        if not dry_run and enhanced_count > 0:
            annotation_data['tree'].write(annotation_path, encoding='utf-8', xml_declaration=True)
            print(f"  Updated XML file with {enhanced_count} new expressions")
        
        return processed_count, enhanced_count, total_retries, failed_items
        
    except Exception as e:
        print(f"  Error processing annotation file: {e}")
        return 0, 0, 0, 0

def main():
    """Main processing function."""
    args = parse_arguments()
    
    # Set random seed for deterministic file ordering
    random.seed(args.seed)
    print(f"Using random seed: {args.seed}")
    
    # Setup paths
    dataset_root = os.path.abspath(args.dataset_root)
    server_url = args.server_url
    model_name = args.model_name
    
    print(f"Dataset root: {dataset_root}")
    print(f"Server URL: {server_url}")
    print(f"Model name: {model_name}")
    print(f"Threads per file: {args.threads}")
    
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset root not found: {dataset_root}")
        return
    
    # Setup temporary directory
    if args.temp_dir:
        temp_dir = args.temp_dir
        os.makedirs(temp_dir, exist_ok=True)
    else:
        temp_dir = tempfile.mkdtemp()
    
    print(f"Using temporary directory: {temp_dir}")
    
    try:
        # Setup OpenAI client
        print(f"\n🔄 Setting up OpenAI client...")
        client = setup_openai_client(server_url)
        
        # Determine splits to process
        splits = ['train', 'val'] if args.split == 'both' else [args.split]
        
        # Count total annotation files
        print(f"\n🔍 Getting total annotation file count...")
        total_files = 0
        
        for split in splits:
            annotations_dir = os.path.join(dataset_root, 'patches_rules_expressions_unique', split, 'annotations')
            if os.path.exists(annotations_dir):
                annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
                if args.limit:
                    annotation_files = annotation_files[:args.limit]
                
                total_files += len(annotation_files)
                print(f"   {split}: {len(annotation_files)} total files")
        
        print(f"📋 Total annotation files: {total_files}")
        
        if total_files == 0:
            print("\n🎉 No files to process!")
            return
        
        # Initialize progress tracker
        progress_tracker = ProgressTracker(total_files)
        
        print(f"\n🚀 Starting enhancement process...")
        print(f"   Progress updates every {args.progress_interval} files")
        print(f"   Using {args.threads} threads per file (items within each file processed concurrently)")
        print(f"   ETA will be calculated after processing the first few files")
        print("─" * 70)
        
        files_processed_count = 0
        
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
            
            split_stats = {'files_processed': 0, 'files_skipped': 0, 'items_processed': 0, 'expressions_enhanced': 0, 'total_retries': 0, 'failed_items': 0}
            
            # Process files sequentially
            for i, annotation_file in enumerate(annotation_files):
                annotation_path = os.path.join(annotations_dir, annotation_file)
                
                # Print immediate progress for first file
                if files_processed_count == 0:
                    print(f"\n🔄 Starting file processing...")
                
                # Check if file is already fully processed
                if is_file_already_processed(annotation_path):
                    split_stats['files_skipped'] += 1
                    progress_tracker.update_stats(files_skipped=1)
                    print(f"Skipping {annotation_file} (already processed)")
                    continue
                
                print(f"Processing {annotation_file} ({i+1}/{len(annotation_files)})")
                
                items_processed, expressions_enhanced, retries, failed = process_annotation_file(
                    annotation_path, images_dir, client, model_name, temp_dir, args.threads, args.dry_run
                )
                
                if items_processed > 0:
                    split_stats['files_processed'] += 1
                    split_stats['items_processed'] += items_processed
                    split_stats['expressions_enhanced'] += expressions_enhanced
                    split_stats['total_retries'] += retries
                    split_stats['failed_items'] += failed
                    
                    # Update progress tracker
                    progress_tracker.update_stats(
                        files_processed=1,
                        items_processed=items_processed,
                        expressions_enhanced=expressions_enhanced,
                        total_retries=retries,
                        failed_items=failed
                    )
                
                files_processed_count += 1
                
                # Print progress periodically
                if files_processed_count % args.progress_interval == 0:
                    progress_tracker.print_progress()
                
                # Also print progress for the first few files to give immediate feedback
                elif files_processed_count <= 3:
                    progress_tracker.print_progress()
            
            print(f"\n{split.capitalize()} split summary:")
            print(f"  Files processed: {split_stats['files_processed']}")
            print(f"  Files skipped (already processed): {split_stats['files_skipped']}")
            print(f"  Items processed: {split_stats['items_processed']}")
            print(f"  Expressions enhanced: {split_stats['expressions_enhanced']}")
            print(f"  Total retries: {split_stats['total_retries']}")
            print(f"  Failed items (after retries): {split_stats['failed_items']}")
        
        # Print final progress update
        print(f"\n📊 Final Progress Update:")
        progress_tracker.print_progress()
        
        # Final statistics
        final_report = progress_tracker.get_progress_report()
        
        print(f"\n🎉 Processing Complete!")
        print(f"📊 Final Summary:")
        print(f"  Files processed: {progress_tracker.stats['files_processed']:,}")
        print(f"  Files skipped (already processed): {progress_tracker.stats['files_skipped']:,}")
        print(f"  Items processed: {progress_tracker.stats['items_processed']:,}")
        print(f"  Expressions enhanced: {progress_tracker.stats['expressions_enhanced']:,}")
        print(f"  Total retries performed: {progress_tracker.stats['total_retries']:,}")
        print(f"  Failed items (after max retries): {progress_tracker.stats['failed_items']:,}")
        print()
        print(f"⏱️  Timing & Performance:")
        print(f"  Total elapsed time: {final_report['elapsed_minutes']:.1f} minutes ({final_report['elapsed_minutes']/60:.1f} hours)")
        print(f"  Average processing rate: {final_report['processed_files_per_min']:.1f} files/min (actual work)")
        print(f"  Average item processing rate: {final_report['items_per_min']:.1f} items/min")
        if final_report['skip_ratio'] > 0:
            print(f"  Files already processed: {final_report['skip_ratio']*100:.1f}% of files checked were already done")
        
        # Calculate success/retry statistics
        if progress_tracker.stats['items_processed'] > 0:
            success_rate = ((progress_tracker.stats['items_processed'] - progress_tracker.stats['failed_items']) / progress_tracker.stats['items_processed']) * 100
            retry_rate = (progress_tracker.stats['total_retries'] / progress_tracker.stats['items_processed']) * 100
            print(f"\n📈 Performance Statistics:")
            print(f"  Success rate: {success_rate:.1f}%")
            print(f"  Retry rate: {retry_rate:.1f}% (avg {retry_rate/100:.2f} retries per item)")
            if progress_tracker.stats['total_retries'] > 0:
                recovery_rate = ((progress_tracker.stats['total_retries'] - progress_tracker.stats['failed_items']) / progress_tracker.stats['total_retries']) * 100
                print(f"  Recovery rate: {recovery_rate:.1f}% (retries that succeeded)")
        
        if args.dry_run:
            print("\n🔍 Dry run completed - no files were modified")
        else:
            print("\n✅ Enhancement completed successfully!")
    
    finally:
        # Clean up temporary directory if we created it
        pass

if __name__ == "__main__":
    main() 