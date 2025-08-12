#!/usr/bin/env python3
"""
Upload AERIAL-D dataset to Hugging Face Hub

This script processes the AERIAL-D dataset with XML annotations and uploads it to Hugging Face
using the datasets library. It preserves all the metadata including expression types,
RLE masks, and domain information.

Usage:
    python upload_to_huggingface.py --dataset_path /path/to/aeriald --repo_name aerial-d
"""

import os
import json
import argparse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image
import numpy as np
from pycocotools import mask as mask_utils
from datasets import Dataset, DatasetDict, Features, Value, Sequence, Image as HFImage
from huggingface_hub import HfApi
from tqdm import tqdm
import logging
import gc
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_domain_from_filename(filename: str) -> Tuple[str, int]:
    """Determine domain based on annotation filename prefix"""
    filename = filename.upper()
    if filename.startswith('P'):
        return 'isaid', 0  
    elif filename.startswith('L'):
        return 'loveda', 1
    else:
        logger.warning(f"Could not determine domain from filename {filename}, defaulting to iSAID")
        return 'isaid', 0

def parse_rle_segmentation(seg_text: str) -> Dict[str, Any]:
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

def rle_to_binary_mask(rle: Dict[str, Any]) -> np.ndarray:
    """Convert RLE to binary mask"""
    try:
        return mask_utils.decode(rle)
    except Exception as e:
        logger.error(f"Failed to decode RLE mask: {e}")
        return None

def parse_expressions(expressions_elem) -> List[Dict[str, Any]]:
    """Parse expressions from XML element"""
    expressions = []
    if expressions_elem is not None:
        for i, exp in enumerate(expressions_elem.findall('expression')):
            exp_data = {
                'id': exp.get('id', str(i)),
                'text': exp.text.strip() if exp.text else '',
                'type': exp.get('type', 'original')  # original, enhanced, unique
            }
            expressions.append(exp_data)
    return expressions

def parse_single_xml_file(xml_path: str, image_dir: str, split: str) -> List[Dict[str, Any]]:
    """Parse a single XML file and return list of samples"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        logger.error(f"Failed to parse XML file {xml_path}: {e}")
        return []
    
    xml_filename = os.path.basename(xml_path)
    image_filename = root.find('filename').text
    domain_name, domain_id = get_domain_from_filename(xml_filename)
    
    # Get image dimensions
    size_elem = root.find('size')
    if size_elem is not None:
        image_width = int(size_elem.find('width').text)
        image_height = int(size_elem.find('height').text)
    else:
        image_width = image_height = 480  # Default size
    
    image_path = os.path.join(image_dir, image_filename)
    if not os.path.exists(image_path):
        logger.warning(f"Image file not found: {image_path}")
        return []
    
    samples = []
    
    # Process individual objects
    for obj in root.findall('object'):
        obj_id = obj.find('id').text if obj.find('id') is not None else None
        category = obj.find('name').text
        
        # Get bounding box
        bbox_elem = obj.find('bndbox')
        if bbox_elem is not None:
            bbox = {
                'xmin': int(bbox_elem.find('xmin').text),
                'ymin': int(bbox_elem.find('ymin').text),
                'xmax': int(bbox_elem.find('xmax').text),
                'ymax': int(bbox_elem.find('ymax').text)
            }
        else:
            bbox = None
        
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
        possible_colors = colors_elem.text.split(',') if colors_elem is not None else []
        
        # Parse expressions
        expressions = parse_expressions(obj.find('expressions'))
        
        # Create a sample for each expression
        for expression in expressions:
            sample = {
                'image_path': image_path,
                'image_filename': image_filename,
                'xml_filename': xml_filename,
                'split': split,
                'domain_name': domain_name,
                'domain_id': domain_id,
                'image_width': image_width,
                'image_height': image_height,
                'object_type': 'individual',
                'object_id': obj_id,
                'category': category,
                'bbox': bbox,
                'area': area,
                'possible_colors': possible_colors,
                'rle_mask': rle_mask,
                'expression_id': expression['id'],
                'expression_text': expression['text'],
                'expression_type': expression['type'],
                'group_id': None,
                'group_size': 1,
                'instance_ids': [obj_id] if obj_id else [],
                'centroid': None,
                'grid_position': None
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
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'xml_filename': xml_filename,
                    'split': split,
                    'domain_name': domain_name,
                    'domain_id': domain_id,
                    'image_width': image_width,
                    'image_height': image_height,
                    'object_type': 'group',
                    'object_id': None,
                    'category': category,
                    'bbox': None,
                    'area': None,
                    'possible_colors': [],
                    'rle_mask': rle_mask,
                    'expression_id': expression['id'],
                    'expression_text': expression['text'],
                    'expression_type': expression['type'],
                    'group_id': group_id,
                    'group_size': group_size,
                    'instance_ids': instance_ids,
                    'centroid': centroid,
                    'grid_position': grid_position
                }
                samples.append(sample)
    
    return samples

def process_split(annotations_root: str, images_root: str, split: str) -> List[Dict[str, Any]]:
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
    
    logger.info(f"Found {len(all_xml_files)} total XML files, excluding {len(all_xml_files) - len(xml_files)} DeepGlobe files")
    logger.info(f"Processing {len(xml_files)} XML files in {split} split")
    
    all_samples = []
    for xml_file in tqdm(xml_files, desc=f"Processing {split} XML files"):
        xml_path = os.path.join(ann_dir, xml_file)
        samples = parse_single_xml_file(xml_path, image_dir, split)
        all_samples.extend(samples)
    
    logger.info(f"Generated {len(all_samples)} samples from {split} split")
    return all_samples

def create_sample_generator(samples: List[Dict[str, Any]]):
    """Generator that yields samples one by one to avoid loading all into memory"""
    for sample in samples:
        try:
            # Load and convert image
            image = Image.open(sample['image_path']).convert('RGB')
            
            # Validate RLE mask but don't convert to binary here to save memory
            if sample['rle_mask'] is None:
                logger.warning(f"Skipping sample due to invalid mask: {sample['image_filename']}")
                continue
            
            # Yield sample data
            yield {
                'image': image,
                'image_filename': sample['image_filename'],
                'xml_filename': sample['xml_filename'],
                'split': sample['split'],
                'domain_name': sample['domain_name'],
                'domain_id': sample['domain_id'],
                'image_width': sample['image_width'],
                'image_height': sample['image_height'],
                'object_type': sample['object_type'],
                'object_id': sample['object_id'] or '',
                'category': sample['category'],
                'bbox': sample['bbox'] or {'xmin': 0, 'ymin': 0, 'xmax': 0, 'ymax': 0},
                'area': sample['area'] or 0,
                'possible_colors': sample['possible_colors'],
                'rle_mask': sample['rle_mask'],
                'expression_id': sample['expression_id'],
                'expression_text': sample['expression_text'],
                'expression_type': sample['expression_type'],
                'group_id': sample['group_id'] or '',
                'group_size': sample['group_size'],
                'instance_ids': sample['instance_ids'],
                'centroid': sample['centroid'] or {'x': 0.0, 'y': 0.0},
                'grid_position': sample['grid_position'] or ''
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {sample['image_filename']}: {e}")
            continue

def convert_to_hf_dataset(samples: List[Dict[str, Any]], batch_size: int = 1000) -> Dataset:
    """Convert samples to Hugging Face Dataset using batched processing"""
    
    # Define the dataset features schema (removed binary_mask to save memory)
    features = Features({
        'image': HFImage(),
        'image_filename': Value('string'),
        'xml_filename': Value('string'),
        'split': Value('string'),
        'domain_name': Value('string'),
        'domain_id': Value('int32'),
        'image_width': Value('int32'),
        'image_height': Value('int32'),
        'object_type': Value('string'),  # 'individual' or 'group'
        'object_id': Value('string'),
        'category': Value('string'),
        'bbox': {
            'xmin': Value('int32'),
            'ymin': Value('int32'),
            'xmax': Value('int32'),
            'ymax': Value('int32')
        },
        'area': Value('int32'),
        'possible_colors': Sequence(Value('string')),
        'rle_mask': {
            'size': Sequence(Value('int32')),
            'counts': Value('string')
        },
        'expression_id': Value('string'),
        'expression_text': Value('string'),
        'expression_type': Value('string'),  # 'original', 'enhanced', 'unique'
        'group_id': Value('string'),
        'group_size': Value('int32'),
        'instance_ids': Sequence(Value('string')),
        'centroid': {
            'x': Value('float32'),
            'y': Value('float32')
        },
        'grid_position': Value('string')
    })
    
    logger.info(f"Converting {len(samples)} samples to Hugging Face format using batched processing...")
    
    # Create dataset from generator to avoid loading all data into memory
    dataset = Dataset.from_generator(
        lambda: create_sample_generator(samples),
        features=features
    )
    
    logger.info(f"Successfully created dataset with {len(dataset)} samples")
    return dataset

def create_dataset_card(repo_name: str, dataset_stats: Dict[str, Any]) -> str:
    """Create a dataset card for the repository"""
    
    card_content = f"""---
language:
- en
tags:
- computer-vision
- instance-segmentation
- referring-expression-segmentation
- aerial-imagery
- remote-sensing
task_categories:
- image-segmentation
- zero-shot-image-classification
pretty_name: AERIAL-D
size_categories:
- 10K<n<100K
---

# AERIAL-D: Referring Expression Instance Segmentation in Aerial Imagery

## Dataset Description

AERIAL-D is a comprehensive dataset for Referring Expression Instance Segmentation (RRSIS) in aerial and satellite imagery. The dataset contains high-resolution aerial photos with detailed instance segmentation masks and natural language referring expressions that describe specific objects within the images.

### Key Features

- **Multiple Expression Types**: Each object can have multiple referring expressions of different types:
  - `original`: Base expressions generated using rule-based methods
  - `enhanced`: Improved expressions using multimodal LLM augmentation
  - `unique`: Distinctive expressions that uniquely identify objects
- **Rich Annotations**: Includes bounding boxes, segmentation masks in RLE format, and object categories
- **Multi-Domain**: Contains images from two different sources:
  - iSAID (Patches starting with 'P')
  - LoveDA (Patches starting with 'L')
- **Group Annotations**: Some objects are grouped together with collective referring expressions

### Dataset Statistics

- **Total Images**: {dataset_stats.get('total_images', 'N/A')}
- **Total Samples**: {dataset_stats.get('total_samples', 'N/A')}
- **Train Split**: {dataset_stats.get('train_samples', 'N/A')} samples
- **Validation Split**: {dataset_stats.get('val_samples', 'N/A')} samples
- **Expression Types**:
  - Original: {dataset_stats.get('original_expressions', 'N/A')}
  - Enhanced: {dataset_stats.get('enhanced_expressions', 'N/A')}
  - Unique: {dataset_stats.get('unique_expressions', 'N/A')}

### Categories

The dataset includes various object categories commonly found in aerial imagery:
- Buildings
- Water bodies
- Forests/Vegetation
- Agricultural areas
- Roads
- And more...

## Dataset Structure

Each sample in the dataset contains:

- `image`: The aerial/satellite image (PIL Image)
- `image_filename`: Original filename of the image
- `xml_filename`: Corresponding XML annotation filename  
- `split`: Dataset split ('train' or 'val')
- `domain_name`: Source domain ('isaid' or 'loveda')  
- `domain_id`: Numeric domain identifier (0 or 1)
- `image_width`, `image_height`: Image dimensions
- `object_type`: Type of annotation ('individual' or 'group')
- `category`: Object category/class
- `bbox`: Bounding box coordinates (for individual objects)
- `area`: Object area in pixels
- `rle_mask`: Segmentation mask in RLE format (can be decoded using pycocotools)
- `expression_text`: Natural language referring expression
- `expression_type`: Type of expression ('original', 'enhanced', 'unique')
- `expression_id`: Unique identifier for the expression
- Additional metadata for groups and spatial relationships

## Usage

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("luisml77/{repo_name}")

# Load specific split
train_dataset = load_dataset("luisml77/{repo_name}", split="train")
val_dataset = load_dataset("luisml77/{repo_name}", split="validation")

# Example usage
from pycocotools import mask as mask_utils

sample = dataset['train'][0]
image = sample['image']
expression = sample['expression_text']
rle_mask = sample['rle_mask']
category = sample['category']

# Decode RLE mask to binary mask if needed
binary_mask = mask_utils.decode(rle_mask)
```

## Applications

This dataset is designed for research in:

- **Referring Expression Instance Segmentation (RRSIS)**
- **Open-vocabulary semantic segmentation**
- **Vision-language understanding in remote sensing**
- **Multimodal learning with aerial imagery**
- **Zero-shot object detection and segmentation**

## Citation

If you use this dataset in your research, please cite:

```bibtex
@misc{{aerial-d-2024,
  title={{AERIAL-D: Open-Vocabulary Semantic Segmentation of Aerial Photos}},
  author={{[Your Name]}},
  year={{2024}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/luisml77/{repo_name}}}}}
}}
```

## License

[Specify your license here]

## Acknowledgments

This dataset builds upon and enhances existing RRSIS datasets including RefSegRS, RISBench, and RRSIS-D, with additional LLM-generated descriptions and multi-domain aerial imagery.
"""
    
    return card_content

def calculate_dataset_statistics(all_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate statistics about the dataset"""
    stats = {
        'total_samples': len(all_samples),
        'total_images': len(set(sample['image_filename'] for sample in all_samples)),
        'train_samples': len([s for s in all_samples if s['split'] == 'train']),
        'val_samples': len([s for s in all_samples if s['split'] == 'val']),
        'original_expressions': len([s for s in all_samples if s['expression_type'] == 'original']),
        'enhanced_expressions': len([s for s in all_samples if s['expression_type'] == 'enhanced']),
        'unique_expressions': len([s for s in all_samples if s['expression_type'] == 'unique']),
        'individual_objects': len([s for s in all_samples if s['object_type'] == 'individual']),
        'group_objects': len([s for s in all_samples if s['object_type'] == 'group']),
        'domains': {
            'isaid': len([s for s in all_samples if s['domain_name'] == 'isaid']),
            'loveda': len([s for s in all_samples if s['domain_name'] == 'loveda'])
        },
        'categories': {}
    }
    
    # Count categories
    for sample in all_samples:
        category = sample['category']
        stats['categories'][category] = stats['categories'].get(category, 0) + 1
    
    return stats

def log_memory_usage():
    """Log current memory usage"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_gb = memory_info.rss / 1024 / 1024 / 1024
    logger.info(f"Current memory usage: {memory_gb:.2f} GB")

def main():
    parser = argparse.ArgumentParser(description='Upload AERIAL-D dataset to Hugging Face Hub')
    parser.add_argument('--annotations_path', type=str, 
                       default='/cfs/home/u035679/aerialseg/datagen/dataset/patches_rules_expressions_unique',
                       help='Path to the annotations directory (patches_rules_expressions_unique)')
    parser.add_argument('--images_path', type=str,
                       default='/cfs/home/u035679/aerialseg/datagen/dataset/patches',
                       help='Path to the images directory (patches)')
    parser.add_argument('--repo_name', type=str, default='aerial-d',
                       help='Name for the Hugging Face repository')
    parser.add_argument('--username', type=str, default='luisml77',
                       help='Hugging Face username')
    parser.add_argument('--private', action='store_true',
                       help='Make the repository private')
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Actually push to Hugging Face Hub (otherwise just prepare locally)')
    parser.add_argument('--max_samples_per_split', type=int, default=None,
                       help='Limit number of samples per split for testing')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for processing samples (lower = less memory usage)')
    
    args = parser.parse_args()
    
    # Validate dataset paths
    if not os.path.exists(args.annotations_path):
        logger.error(f"Annotations path does not exist: {args.annotations_path}")
        return
    
    if not os.path.exists(args.images_path):
        logger.error(f"Images path does not exist: {args.images_path}")
        return
    
    logger.info(f"Processing AERIAL-D dataset")
    logger.info(f"  Annotations from: {args.annotations_path}")
    logger.info(f"  Images from: {args.images_path}")
    
    log_memory_usage()
    
    # Process each split
    all_samples = []
    
    # Process train split
    logger.info("Processing train split...")
    train_samples = process_split(args.annotations_path, args.images_path, 'train')
    if args.max_samples_per_split:
        train_samples = train_samples[:args.max_samples_per_split]
    all_samples.extend(train_samples)
    log_memory_usage()
    
    # Process validation split
    logger.info("Processing validation split...")
    val_samples = process_split(args.annotations_path, args.images_path, 'val')
    if args.max_samples_per_split:
        val_samples = val_samples[:args.max_samples_per_split]
    all_samples.extend(val_samples)
    log_memory_usage()
    
    if not all_samples:
        logger.error("No samples found in dataset")
        return
    
    # Calculate statistics
    logger.info("Calculating dataset statistics...")
    stats = calculate_dataset_statistics(all_samples)
    logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
    
    # Convert to Hugging Face datasets
    logger.info("Converting to Hugging Face dataset format...")
    log_memory_usage()
    
    # Split samples by split type
    train_data = [s for s in all_samples if s['split'] == 'train']
    val_data = [s for s in all_samples if s['split'] == 'val']
    
    # Clear the all_samples list to free memory
    del all_samples
    gc.collect()
    log_memory_usage()
    
    # Create datasets one at a time to reduce memory pressure
    datasets = {}
    if train_data:
        logger.info("Creating train dataset...")
        datasets['train'] = convert_to_hf_dataset(train_data, batch_size=args.batch_size)
        log_memory_usage()
        # Clear train_data after creating dataset
        del train_data
        gc.collect()
        
    if val_data:
        logger.info("Creating validation dataset...")
        datasets['validation'] = convert_to_hf_dataset(val_data, batch_size=args.batch_size)
        log_memory_usage()
        # Clear val_data after creating dataset
        del val_data
        gc.collect()
    
    # Create DatasetDict
    logger.info("Creating DatasetDict...")
    dataset_dict = DatasetDict(datasets)
    log_memory_usage()
    
    if args.push_to_hub:
        logger.info(f"Pushing dataset to Hugging Face Hub: {args.username}/{args.repo_name}")
        
        # Create dataset card
        card_content = create_dataset_card(args.repo_name, stats)
        
        # Push to hub
        dataset_dict.push_to_hub(
            f"{args.username}/{args.repo_name}",
            private=args.private,
            card_data={'license': 'apache-2.0'}  # Update as needed
        )
        
        # Upload dataset card
        api = HfApi()
        api.upload_file(
            path_or_fileobj=card_content.encode(),
            path_in_repo="README.md",
            repo_id=f"{args.username}/{args.repo_name}",
            repo_type="dataset"
        )
        
        logger.info(f"Dataset successfully uploaded to: https://huggingface.co/datasets/{args.username}/{args.repo_name}")
    else:
        logger.info("Dataset prepared but not uploaded (use --push_to_hub to upload)")
        logger.info(f"Train samples: {len(train_data) if train_data else 0}")
        logger.info(f"Validation samples: {len(val_data) if val_data else 0}")

if __name__ == '__main__':
    main()
