#!/usr/bin/env python3
"""
Convert AERIAL-D XML annotations to JSONL format

This script converts XML annotation files to JSONL format, creating one JSON object
per expression (one per line). Outputs train.jsonl and val.jsonl files.

Usage:
    python xml_to_jsonl.py --dataset_path /path/to/aeriald --output_dir /path/to/output
    python xml_to_jsonl.py --dataset_path /path/to/aeriald --max_files 10  # Test with 10 files
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_domain_from_filename(filename: str) -> tuple[str, int]:
    """Determine domain based on annotation filename prefix"""
    filename = filename.upper()
    if filename.startswith('P'):
        return 'isaid', 0
    elif filename.startswith('L'):
        return 'loveda', 1
    else:
        logger.warning(f"Could not determine domain from filename {filename}, defaulting to iSAID")
        return 'isaid', 0


def parse_rle_segmentation(seg_text: str) -> Optional[Dict[str, Any]]:
    """Parse RLE segmentation from XML text"""
    try:
        # Clean up the text and evaluate as Python dict
        seg_dict = eval(seg_text)
        return {
            'size': seg_dict['size'],
            'counts': seg_dict['counts']
        }
    except Exception as e:
        logger.error(f"Failed to parse segmentation: {e}")
        return None


def parse_expressions(expressions_elem) -> List[Dict[str, Any]]:
    """Parse expressions from XML element"""
    expressions = []
    if expressions_elem is not None:
        for i, exp in enumerate(expressions_elem.findall('expression')):
            exp_id = exp.get('id')
            exp_type_attr = exp.get('type')
            
            # Determine expression type
            if exp_type_attr == 'enhanced':
                expression_type = 'enhanced'
                expression_id = f"enhanced_{i}" if exp_id is None else f"enhanced_{exp_id}"
            elif exp_type_attr == 'unique':
                expression_type = 'unique'
                expression_id = f"unique_{i}" if exp_id is None else f"unique_{exp_id}"
            else:
                # Original expression (has id attribute)
                expression_type = 'original'
                expression_id = exp_id if exp_id is not None else str(i)
            
            exp_data = {
                'id': expression_id,
                'text': exp.text.strip() if exp.text else '',
                'type': expression_type
            }
            expressions.append(exp_data)
    return expressions


def parse_single_xml_file(xml_path: str, image_dir: str, split: str) -> List[Dict[str, Any]]:
    """Parse a single XML file and return list of samples (one per expression)"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse XML file {xml_path}: {e}")
        return []
    
    xml_filename = os.path.basename(xml_path)
    image_filename = root.find('filename').text
    domain_name, domain_id = get_domain_from_filename(image_filename)
    
    # Get image dimensions
    size_elem = root.find('size')
    if size_elem is not None:
        image_width = int(size_elem.find('width').text)
        image_height = int(size_elem.find('height').text)
    else:
        image_width = image_height = 480  # Default size
    
    # Verify image exists (but don't fail if it doesn't)
    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        logger.warning(f"Image file not found: {image_path}")
    
    samples = []
    
    # Process individual objects
    for obj in root.findall('object'):
        obj_id = obj.find('id').text if obj.find('id') is not None else None
        category = obj.find('name').text
        
        # Get bounding box
        bbox_elem = obj.find('bndbox')
        bbox = None
        if bbox_elem is not None:
            bbox = {
                'xmin': int(bbox_elem.find('xmin').text),
                'ymin': int(bbox_elem.find('ymin').text),
                'xmax': int(bbox_elem.find('xmax').text),
                'ymax': int(bbox_elem.find('ymax').text)
            }
        
        # Get segmentation
        seg_elem = obj.find('segmentation')
        if seg_elem is None or not seg_elem.text:
            continue
        
        rle_mask = parse_rle_segmentation(seg_elem.text)
        if rle_mask is None:
            continue
        
        # Get area
        area_elem = obj.find('area')
        area = int(area_elem.text) if area_elem is not None else None
        
        # Get possible colors
        colors_elem = obj.find('possible_colors')
        possible_colors = []
        if colors_elem is not None and colors_elem.text:
            possible_colors = [c.strip() for c in colors_elem.text.split(',')]
        
        # Parse expressions
        expressions = parse_expressions(obj.find('expressions'))
        
        # Create a sample for each expression
        for expression in expressions:
            sample = {
                'image': f"images/{image_filename}",
                'expression_text': expression['text'],
                'expression_id': expression['id'],
                'expression_type': expression['type'],
                'object_type': 'individual',
                'category': category,
                'object_id': obj_id,
                'group_id': None,
                'group_size': 1,
                'bbox': bbox,
                'area': area,
                'possible_colors': possible_colors,
                'rle_mask': rle_mask,
                'centroid': None,
                'grid_position': None,
                'instance_ids': [],
                'split': split,
                'domain_name': domain_name,
                'domain_id': domain_id,
                'image_width': image_width,
                'image_height': image_height
            }
            samples.append(sample)
    
    # Process groups
    groups_elem = root.find('groups')
    if groups_elem is not None:
        for group in groups_elem.findall('group'):
            group_id = group.find('id').text if group.find('id') is not None else None
            category = group.find('category').text if group.find('category') is not None else 'unknown'
            
            # Get group size
            size_elem = group.find('size')
            group_size = int(size_elem.text) if size_elem is not None else 1
            
            # Get centroid
            centroid_elem = group.find('centroid')
            centroid = None
            if centroid_elem is not None:
                x_elem = centroid_elem.find('x')
                y_elem = centroid_elem.find('y')
                if x_elem is not None and y_elem is not None:
                    centroid = {
                        'x': float(x_elem.text),
                        'y': float(y_elem.text)
                    }
            
            # Get grid position
            grid_pos_elem = group.find('grid_position')
            grid_position = grid_pos_elem.text if grid_pos_elem is not None else None
            
            # Get instance IDs
            instance_ids_elem = group.find('instance_ids')
            instance_ids = []
            if instance_ids_elem is not None and instance_ids_elem.text:
                instance_ids = [id.strip() for id in instance_ids_elem.text.split(',')]
            
            # Get segmentation
            seg_elem = group.find('segmentation')
            if seg_elem is None or not seg_elem.text:
                continue
            
            rle_mask = parse_rle_segmentation(seg_elem.text)
            if rle_mask is None:
                continue
            
            # Parse expressions
            expressions = parse_expressions(group.find('expressions'))
            
            # Create a sample for each expression
            for expression in expressions:
                sample = {
                    'image': f"images/{image_filename}",
                    'expression_text': expression['text'],
                    'expression_id': expression['id'],
                    'expression_type': expression['type'],
                    'object_type': 'group',
                    'category': category,
                    'object_id': None,
                    'group_id': group_id,
                    'group_size': group_size,
                    'bbox': None,
                    'area': None,
                    'possible_colors': [],
                    'rle_mask': rle_mask,
                    'centroid': centroid,
                    'grid_position': grid_position,
                    'instance_ids': instance_ids,
                    'split': split,
                    'domain_name': domain_name,
                    'domain_id': domain_id,
                    'image_width': image_width,
                    'image_height': image_height
                }
                samples.append(sample)
    
    return samples


def process_split(annotations_root: str, images_root: str, split: str, max_files: Optional[int] = None) -> List[Dict[str, Any]]:
    """Process all XML files in a split"""
    ann_dir = os.path.join(annotations_root, split, 'annotations')
    image_dir = os.path.join(images_root, split, 'images')
    
    if not os.path.exists(ann_dir):
        logger.error(f"Annotations directory not found: {ann_dir}")
        return []
    
    if not os.path.exists(image_dir):
        logger.error(f"Images directory not found: {image_dir}")
        return []
    
    # Get all XML files and exclude DeepGlobe files (starting with 'D')
    all_xml_files = [f for f in os.listdir(ann_dir) if f.endswith('.xml')]
    xml_files = [f for f in all_xml_files if not f.upper().startswith('D')]
    
    # Limit files if max_files is specified
    if max_files is not None:
        xml_files = xml_files[:max_files]
        logger.info(f"Limiting to {max_files} files for testing")
    
    logger.info(f"Found {len(all_xml_files)} total XML files, excluding {len(all_xml_files) - len(xml_files)} DeepGlobe files")
    logger.info(f"Processing {len(xml_files)} XML files in {split} split")
    
    all_samples = []
    for xml_file in tqdm(xml_files, desc=f"Processing {split} XML files"):
        xml_path = os.path.join(ann_dir, xml_file)
        samples = parse_single_xml_file(xml_path, image_dir, split)
        all_samples.extend(samples)
    
    logger.info(f"Generated {len(all_samples)} samples from {split} split")
    return all_samples


def write_jsonl(samples: List[Dict[str, Any]], output_path: str):
    """Write samples to JSONL file"""
    logger.info(f"Writing {len(samples)} samples to {output_path}")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            json_line = json.dumps(sample, ensure_ascii=False)
            f.write(json_line + '\n')
    logger.info(f"Successfully wrote {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Convert AERIAL-D XML annotations to JSONL format')
    parser.add_argument('--dataset_path', type=str,
                       default='/cfs/home/u035679/datasets/aeriald',
                       help='Path to the aeriald dataset directory')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for JSONL files (default: same as dataset_path)')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of XML files to process per split (for testing)')
    
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir is None:
        args.output_dir = args.dataset_path
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset path does not exist: {args.dataset_path}")
        return
    
    logger.info(f"Processing AERIAL-D dataset")
    logger.info(f"  Dataset path: {args.dataset_path}")
    logger.info(f"  Output directory: {args.output_dir}")
    if args.max_files:
        logger.info(f"  Max files per split: {args.max_files}")
    
    # Process train split
    logger.info("Processing train split...")
    train_samples = process_split(args.dataset_path, args.dataset_path, 'train', args.max_files)
    
    # Process validation split
    logger.info("Processing validation split...")
    val_samples = process_split(args.dataset_path, args.dataset_path, 'val', args.max_files)
    
    if not train_samples and not val_samples:
        logger.error("No samples found in dataset")
        return
    
    # Write JSONL files
    os.makedirs(args.output_dir, exist_ok=True)
    
    if train_samples:
        train_output = os.path.join(args.output_dir, 'train.jsonl')
        write_jsonl(train_samples, train_output)
    
    if val_samples:
        val_output = os.path.join(args.output_dir, 'val.jsonl')
        write_jsonl(val_samples, val_output)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Conversion Summary:")
    logger.info(f"  Train samples: {len(train_samples)}")
    logger.info(f"  Validation samples: {len(val_samples)}")
    logger.info(f"  Total samples: {len(train_samples) + len(val_samples)}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
