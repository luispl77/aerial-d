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
    parser.add_argument('--temp_dir', type=str, default=None,
                        help='Temporary directory for visualization images')
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
                                   linewidth=3, edgecolor='red', facecolor='none')
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

def setup_model_and_processor(model_path):
    """Load the fine-tuned model and processor."""
    print(f"Loading fine-tuned model from: {model_path}")
    
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )
    
    processor = AutoProcessor.from_pretrained(model_path)
    
    print("Model loaded successfully")
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

def process_item_with_llm(item, image_path, model, processor, temp_dir):
    """Process a single item (object or group) with the LLM."""
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
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to("cuda")
        
        # Generate response
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=1024, do_sample=True, temperature=0.8)
        
        # Decode response
        generated_content = processor.decode(output[0], skip_special_tokens=True)
        input_length = len(processor.decode(inputs['input_ids'][0], skip_special_tokens=True))
        generated_content = generated_content[input_length:].strip()
        
        # Parse JSON response
        enhanced_data = extract_and_parse_json(generated_content)
        if enhanced_data is None:
            print(f"  Warning: Could not parse JSON response for {item['type']} {item['id']}")
            return None
        
        # Clean up visualization
        try:
            os.remove(viz_path)
        except:
            pass
        
        return enhanced_data
        
    except Exception as e:
        print(f"  Error processing {item['type']} {item['id']}: {e}")
        return None

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
            return 0, 0  # No items to process
        
        # Get corresponding image
        image_filename = annotation_data['filename']
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            print(f"  Warning: Image not found: {image_path}")
            return 0, 0
        
        processed_count = 0
        enhanced_count = 0
        
        # Process each item with expressions
        for item in annotation_data['items']:
            print(f"  Processing {item['type']} {item['id']} ({item['category']})")
            
            enhanced_data = process_item_with_llm(item, image_path, model, processor, temp_dir)
            
            if enhanced_data:
                if not dry_run:
                    add_enhanced_expressions_to_xml(item, enhanced_data)
                
                # Count enhancements
                num_enhanced = len(enhanced_data.get('enhanced_expressions', []))
                num_unique = len(enhanced_data.get('unique_expressions', []))
                enhanced_count += num_enhanced + num_unique
                
                print(f"    Added {num_enhanced} enhanced + {num_unique} unique expressions")
            
            processed_count += 1
        
        # Save modified XML
        if not dry_run and enhanced_count > 0:
            annotation_data['tree'].write(annotation_path, encoding='utf-8', xml_declaration=True)
            print(f"  Updated XML file with {enhanced_count} new expressions")
        
        return processed_count, enhanced_count
        
    except Exception as e:
        print(f"  Error processing annotation file: {e}")
        return 0, 0

def main():
    """Main processing function."""
    args = parse_arguments()
    
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
        # Load model
        print("\nLoading model...")
        model, processor = setup_model_and_processor(model_path)
        
        # Determine splits to process
        splits = ['train', 'val'] if args.split == 'both' else [args.split]
        
        total_files = 0
        total_items = 0
        total_enhanced = 0
        
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
            
            # Get annotation files
            annotation_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
            
            if args.limit:
                annotation_files = annotation_files[:args.limit]
            
            print(f"Found {len(annotation_files)} annotation files in {split}")
            
            # Process each annotation file
            split_files = 0
            split_items = 0
            split_enhanced = 0
            
            for annotation_file in tqdm(annotation_files, desc=f"Processing {split}"):
                annotation_path = os.path.join(annotations_dir, annotation_file)
                
                print(f"\nProcessing: {annotation_file}")
                items_processed, expressions_enhanced = process_annotation_file(
                    annotation_path, images_dir, model, processor, temp_dir, args.dry_run
                )
                
                if items_processed > 0:
                    split_files += 1
                    split_items += items_processed
                    split_enhanced += expressions_enhanced
            
            print(f"\n{split.capitalize()} split summary:")
            print(f"  Files processed: {split_files}")
            print(f"  Items processed: {split_items}")
            print(f"  Expressions enhanced: {split_enhanced}")
            
            total_files += split_files
            total_items += split_items
            total_enhanced += split_enhanced
        
        print(f"\nTotal summary:")
        print(f"  Files processed: {total_files}")
        print(f"  Items processed: {total_items}")
        print(f"  Expressions enhanced: {total_enhanced}")
        
        if args.dry_run:
            print("\nDry run completed - no files were modified")
        else:
            print("\nEnhancement completed successfully!")
    
    finally:
        # Clean up temporary directory if we created it
        if not args.temp_dir:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

if __name__ == "__main__":
    main() 