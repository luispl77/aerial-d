#!/usr/bin/env python3
"""
Step 9: Convert XML annotations to RefCOCO JSON format

This script converts the XML annotations with enhanced expressions to RefCOCO JSON format,
preserving RLE segmentations and creating a standard format for referring segmentation training.
"""

import os
import json
import xml.etree.ElementTree as ET
import argparse
from tqdm import tqdm
from pycocotools import mask as mask_utils
import numpy as np
from PIL import Image
import base64
from datetime import datetime

def setup_args():
    parser = argparse.ArgumentParser(description='Convert XML annotations to RefCOCO JSON format')
    parser.add_argument('--dataset_dir', type=str, default='/cfs/home/u035679/aerialseg/datagen/dataset',
                       help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: dataset_dir/refcoco_format)')
    parser.add_argument('--splits', nargs='+', default=['train', 'val'],
                       help='Dataset splits to process')
    parser.add_argument('--include_groups', action='store_true', default=True,
                       help='Include group annotations')
    parser.add_argument('--include_enhanced', action='store_true', default=True,
                       help='Include enhanced expressions')
    parser.add_argument('--include_unique', action='store_true', default=True,
                       help='Include unique expressions')
    return parser.parse_args()

def get_category_mapping():
    """Define category mapping for aerial image classes"""
    categories = {
        'ship': 1, 'storage_tank': 2, 'baseball_diamond': 3, 'tennis_court': 4,
        'basketball_court': 5, 'ground_track_field': 6, 'bridge': 7, 'large_vehicle': 8,
        'small_vehicle': 9, 'helicopter': 10, 'swimming_pool': 11, 'roundabout': 12,
        'soccer_ball_field': 13, 'plane': 14, 'harbor': 15, 'building': 16,
        'road': 17, 'water': 18, 'barren': 19, 'forest': 20, 'agricultural': 21
    }
    return categories

def decode_rle_mask(rle_string, height, width):
    """Decode RLE string to binary mask"""
    try:
        # Handle dictionary format from XML: "{'size': [480, 480], 'counts': '...'}"
        if isinstance(rle_string, str):
            # Try to parse as dictionary string first
            try:
                import ast
                rle_dict = ast.literal_eval(rle_string)
                if isinstance(rle_dict, dict) and 'counts' in rle_dict and 'size' in rle_dict:
                    # This is the XML format with dict
                    rle = {
                        'size': rle_dict['size'],
                        'counts': rle_dict['counts'].encode('utf-8') if isinstance(rle_dict['counts'], str) else rle_dict['counts']
                    }
                    return mask_utils.decode(rle)
            except (ValueError, SyntaxError):
                # Fall back to base64 decoding
                rle_bytes = base64.b64decode(rle_string)
                rle = {'size': [height, width], 'counts': rle_bytes}
        else:
            rle = rle_string
        return mask_utils.decode(rle)
    except Exception as e:
        print(f"Error decoding RLE mask: {e}")
        print(f"RLE string: {rle_string[:100]}..." if len(str(rle_string)) > 100 else f"RLE string: {rle_string}")
        return None

def mask_to_rle_coco_format(mask):
    """Convert binary mask to COCO RLE format"""
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    # Convert bytes to string for JSON serialization
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle

def get_bbox_from_mask(mask):
    """Extract bounding box from binary mask"""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 0, 0]
    
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    # Convert to COCO format: [x, y, width, height]
    return [int(cmin), int(rmin), int(cmax - cmin + 1), int(rmax - rmin + 1)]

def process_xml_file(xml_path, image_info, category_mapping, args):
    """Process a single XML file and extract annotations and referring expressions"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    annotations = []
    refs = []
    
    # Get image dimensions
    height = int(root.find('size/height').text)
    width = int(root.find('size/width').text)
    
    ann_id = 1
    ref_id = 1
    sent_id = 1
    
    # Process individual objects
    for obj in root.findall('object'):
        obj_id = obj.find('id').text
        category_name = obj.find('name').text
        category_id = category_mapping.get(category_name, 0)
        
        # Get RLE mask
        segmentation_elem = obj.find('segmentation')
        if segmentation_elem is None:
            continue
            
        rle_string = segmentation_elem.text
        if not rle_string:
            continue
            
        # Decode mask to get bbox and area
        mask = decode_rle_mask(rle_string, height, width)
        if mask is None:
            continue
            
        bbox = get_bbox_from_mask(mask)
        area = float(np.sum(mask))
        
        # Convert mask back to COCO RLE format for JSON
        coco_rle = mask_to_rle_coco_format(mask)
        
        # Create annotation
        annotation = {
            'id': ann_id,
            'image_id': image_info['id'],
            'category_id': category_id,
            'segmentation': coco_rle,
            'bbox': bbox,
            'area': area,
            'iscrowd': 0,
            'original_obj_id': obj_id
        }
        annotations.append(annotation)
        
        # Collect all expressions for this object
        all_expressions = []
        
        # Original expressions
        expressions_elem = obj.find('expressions')
        if expressions_elem is not None:
            for expr in expressions_elem.findall('expression'):
                if expr.text and expr.text.strip():
                    # Determine type from attributes
                    expr_type = expr.get('type', 'original')
                    if expr.get('id') is not None and expr_type == 'original':
                        expr_type = 'original'
                    all_expressions.append({
                        'text': expr.text.strip(),
                        'type': expr_type
                    })
        
        # Note: Enhanced and unique expressions are already included in the main expressions element
        # with their type attributes, so no additional processing needed
        
        # Create referring expression entry
        if all_expressions:
            sentences = []
            for expr_info in all_expressions:
                sentences.append({
                    'sent_id': sent_id,
                    'sent': expr_info['text'],
                    'raw': expr_info['text'],
                    'type': expr_info['type']
                })
                sent_id += 1
            
            ref = {
                'ref_id': ref_id,
                'ann_id': ann_id,
                'image_id': image_info['id'],
                'category_id': category_id,
                'sentences': sentences,
                'split': 'train',  # Will be updated based on actual split
                'original_obj_id': obj_id
            }
            refs.append(ref)
            ref_id += 1
        
        ann_id += 1
    
    # Process groups if enabled
    if args.include_groups:
        groups_elem = root.find('groups')
        if groups_elem is not None:
            for group in groups_elem.findall('group'):
                group_id = group.find('id').text
                category_name = group.find('category').text if group.find('category') is not None else 'group'
                category_id = category_mapping.get(category_name, 0)
                
                # Get instance IDs for this group
                instance_ids_elem = group.find('instance_ids')
                if instance_ids_elem is None or not instance_ids_elem.text:
                    continue
                
                instance_ids = [id.strip() for id in instance_ids_elem.text.split(',')]
                
                # Find masks of member objects to create group mask
                member_masks = []
                for member_obj in root.findall('object'):
                    member_id = member_obj.find('id').text
                    if member_id in instance_ids:
                        seg_elem = member_obj.find('segmentation')
                        if seg_elem is not None and seg_elem.text:
                            mask = decode_rle_mask(seg_elem.text, height, width)
                            if mask is not None:
                                member_masks.append(mask)
                
                if not member_masks:
                    continue
                
                # Combine member masks to create group mask
                group_mask = np.zeros((height, width), dtype=np.uint8)
                for mask in member_masks:
                    group_mask = np.logical_or(group_mask, mask)
                
                bbox = get_bbox_from_mask(group_mask)
                area = float(np.sum(group_mask))
                coco_rle = mask_to_rle_coco_format(group_mask)
                
                # Create group annotation
                annotation = {
                    'id': ann_id,
                    'image_id': image_info['id'],
                    'category_id': category_id,
                    'segmentation': coco_rle,
                    'bbox': bbox,
                    'area': area,
                    'iscrowd': 0,
                    'original_group_id': group_id,
                    'member_ids': instance_ids
                }
                annotations.append(annotation)
                
                # Collect group expressions
                all_expressions = []
                
                # Original group expressions
                expressions_elem = group.find('expressions')
                if expressions_elem is not None:
                    for expr in expressions_elem.findall('expression'):
                        if expr.text and expr.text.strip():
                            # Determine type from attributes
                            expr_type = expr.get('type', 'original')
                            if expr.get('id') is not None and expr_type == 'original':
                                expr_type = 'original'
                            all_expressions.append({
                                'text': expr.text.strip(),
                                'type': expr_type
                            })
                
                # Note: Enhanced and unique expressions are already included in the main expressions element
                # with their type attributes, so no additional processing needed
                
                # Create referring expression for group
                if all_expressions:
                    sentences = []
                    for expr_info in all_expressions:
                        sentences.append({
                            'sent_id': sent_id,
                            'sent': expr_info['text'],
                            'raw': expr_info['text'],
                            'type': expr_info['type']
                        })
                        sent_id += 1
                    
                    ref = {
                        'ref_id': ref_id,
                        'ann_id': ann_id,
                        'image_id': image_info['id'],
                        'category_id': category_id,
                        'sentences': sentences,
                        'split': 'train',  # Will be updated based on actual split
                        'original_group_id': group_id,
                        'member_ids': instance_ids
                    }
                    refs.append(ref)
                    ref_id += 1
                
                ann_id += 1
    
    return annotations, refs

def convert_split(dataset_dir, split, output_dir, category_mapping, args):
    """Convert a single split to RefCOCO format"""
    print(f"Processing {split} split...")
    
    # Paths
    annotations_dir = os.path.join(dataset_dir, 'patches_rules_expressions_unique', split, 'annotations')
    images_dir = os.path.join(dataset_dir, 'patches', split, 'images')
    
    if not os.path.exists(annotations_dir):
        print(f"Warning: Annotations directory not found: {annotations_dir}")
        return
    
    # Initialize RefCOCO structure
    refcoco_data = {
        'info': {
            'description': 'AerialSeg Dataset in RefCOCO format',
            'version': '1.0',
            'year': datetime.now().year,
            'contributor': 'AerialSeg Pipeline',
            'date_created': datetime.now().isoformat()
        },
        'images': [],
        'annotations': [],
        'categories': [],
        'refs': []
    }
    
    # Add categories
    for cat_name, cat_id in category_mapping.items():
        refcoco_data['categories'].append({
            'id': cat_id,
            'name': cat_name,
            'supercategory': 'aerial_object'
        })
    
    # Process each XML file
    xml_files = [f for f in os.listdir(annotations_dir) if f.endswith('.xml')]
    image_id = 1
    
    for xml_file in tqdm(xml_files, desc=f"Processing {split}"):
        xml_path = os.path.join(annotations_dir, xml_file)
        
        # Get corresponding image info
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_filename = root.find('filename').text
        image_path = os.path.join(images_dir, image_filename)
        
        if not os.path.exists(image_path):
            continue
        
        # Get image dimensions
        width = int(root.find('size/width').text)
        height = int(root.find('size/height').text)
        
        image_info = {
            'id': image_id,
            'file_name': image_filename,
            'width': width,
            'height': height,
            'split': split
        }
        refcoco_data['images'].append(image_info)
        
        # Process annotations and refs
        annotations, refs = process_xml_file(xml_path, image_info, category_mapping, args)
        
        # Update split information in refs
        for ref in refs:
            ref['split'] = split
        
        refcoco_data['annotations'].extend(annotations)
        refcoco_data['refs'].extend(refs)
        
        image_id += 1
    
    # Save split-specific RefCOCO file
    output_file = os.path.join(output_dir, f'refcoco_{split}.json')
    with open(output_file, 'w') as f:
        json.dump(refcoco_data, f, indent=2)
    
    print(f"Saved {split} split to {output_file}")
    print(f"  Images: {len(refcoco_data['images'])}")
    print(f"  Annotations: {len(refcoco_data['annotations'])}")
    print(f"  Referring expressions: {len(refcoco_data['refs'])}")
    print(f"  Total sentences: {sum(len(ref['sentences']) for ref in refcoco_data['refs'])}")

def main():
    args = setup_args()
    
    # Setup paths
    dataset_dir = os.path.abspath(args.dataset_dir)
    if args.output_dir is None:
        output_dir = os.path.join(dataset_dir, 'refcoco_format')
    else:
        output_dir = os.path.abspath(args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Converting XML annotations to RefCOCO JSON format")
    print(f"Dataset directory: {dataset_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Splits: {args.splits}")
    print(f"Include groups: {args.include_groups}")
    print(f"Include enhanced: {args.include_enhanced}")
    print(f"Include unique: {args.include_unique}")
    
    # Get category mapping
    category_mapping = get_category_mapping()
    
    # Process each split
    for split in args.splits:
        convert_split(dataset_dir, split, output_dir, category_mapping, args)
    
    # Create combined file with all splits
    print("Creating combined RefCOCO file...")
    combined_data = {
        'info': {
            'description': 'AerialSeg Dataset in RefCOCO format',
            'version': '1.0',
            'year': datetime.now().year,
            'contributor': 'AerialSeg Pipeline',
            'date_created': datetime.now().isoformat()
        },
        'images': [],
        'annotations': [],
        'categories': [],
        'refs': []
    }
    
    # Add categories
    for cat_name, cat_id in category_mapping.items():
        combined_data['categories'].append({
            'id': cat_id,
            'name': cat_name,
            'supercategory': 'aerial_object'
        })
    
    # Combine all splits
    for split in args.splits:
        split_file = os.path.join(output_dir, f'refcoco_{split}.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_data = json.load(f)
            
            combined_data['images'].extend(split_data['images'])
            combined_data['annotations'].extend(split_data['annotations'])
            combined_data['refs'].extend(split_data['refs'])
    
    # Save combined file
    combined_file = os.path.join(output_dir, 'refcoco_all.json')
    with open(combined_file, 'w') as f:
        json.dump(combined_data, f, indent=2)
    
    print(f"Saved combined RefCOCO file to {combined_file}")
    print(f"Total images: {len(combined_data['images'])}")
    print(f"Total annotations: {len(combined_data['annotations'])}")
    print(f"Total referring expressions: {len(combined_data['refs'])}")
    print(f"Total sentences: {sum(len(ref['sentences']) for ref in combined_data['refs'])}")
    
    # Save conversion summary
    summary = {
        'conversion_date': datetime.now().isoformat(),
        'source_format': 'XML with RLE masks',
        'target_format': 'RefCOCO JSON',
        'splits_processed': args.splits,
        'include_groups': args.include_groups,
        'include_enhanced': args.include_enhanced,
        'include_unique': args.include_unique,
        'statistics': {
            'total_images': len(combined_data['images']),
            'total_annotations': len(combined_data['annotations']),
            'total_refs': len(combined_data['refs']),
            'total_sentences': sum(len(ref['sentences']) for ref in combined_data['refs'])
        }
    }
    
    summary_file = os.path.join(output_dir, 'conversion_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Conversion completed successfully!")
    print(f"Summary saved to {summary_file}")

if __name__ == '__main__':
    main()