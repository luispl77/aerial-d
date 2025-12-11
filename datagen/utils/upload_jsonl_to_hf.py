#!/usr/bin/env python3
"""
Upload JSONL dataset to Hugging Face Hub

This script uploads a JSONL-formatted dataset to Hugging Face Hub.
It's designed for datasets that are already in JSONL format (e.g., from organize_jsonl_dataset.py).

Usage:
    python upload_jsonl_to_hf.py --jsonl_path /path/to/dataset.jsonl --repo_name my-dataset
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
    """Load JSONL file and return list of dictionaries"""
    samples = []
    logger.info(f"Loading JSONL file: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping invalid JSON on line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(samples)} samples from JSONL file")
    return samples


def split_samples_by_split(samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Split samples by their 'split' field"""
    splits = {}
    for sample in samples:
        split_name = sample.get('split', 'train')
        if split_name not in splits:
            splits[split_name] = []
        splits[split_name].append(sample)
    
    logger.info(f"Split samples: {', '.join(f'{k}: {len(v)}' for k, v in splits.items())}")
    return splits


def create_dataset_from_jsonl(jsonl_path: str, split_name: Optional[str] = None) -> Dataset:
    """Create Hugging Face Dataset from JSONL file"""
    samples = load_jsonl(jsonl_path)
    
    if split_name:
        # Filter by split if specified
        samples = [s for s in samples if s.get('split') == split_name]
        logger.info(f"Filtered to {len(samples)} samples for split '{split_name}'")
    
    if not samples:
        raise ValueError(f"No samples found in JSONL file (split: {split_name})")
    
    # Create dataset
    logger.info(f"Creating Hugging Face Dataset from {len(samples)} samples...")
    dataset = Dataset.from_list(samples)
    
    return dataset


def create_dataset_dict_from_jsonl(jsonl_path: str) -> DatasetDict:
    """Create DatasetDict with train/val/test splits from JSONL"""
    samples = load_jsonl(jsonl_path)
    splits = split_samples_by_split(samples)
    
    datasets = {}
    for split_name, split_samples in splits.items():
        logger.info(f"Creating dataset for split '{split_name}' with {len(split_samples)} samples...")
        datasets[split_name] = Dataset.from_list(split_samples)
    
    return DatasetDict(datasets)


def create_dataset_card(repo_name: str, stats: Dict[str, Any]) -> str:
    """Create a dataset card for the repository"""
    
    card_content = f"""---
language:
- en
tags:
- computer-vision
- referring-expression-segmentation
- aerial-imagery
task_categories:
- image-segmentation
pretty_name: {repo_name}
---

# {repo_name}

## Dataset Description

This dataset contains referring expression segmentation annotations for aerial imagery.

### Dataset Statistics

- **Total Samples**: {stats.get('total_samples', 'N/A')}
- **Splits**: {', '.join(stats.get('splits', {}).keys())}

## Usage

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("luisml77/{repo_name}")

# Load specific split
train_dataset = load_dataset("luisml77/{repo_name}", split="train")
```

## Citation

If you use this dataset, please cite:

```bibtex
@misc{{{repo_name.lower().replace('-', '')}-2024,
  title={{{repo_name}}},
  author={{[Your Name]}},
  year={{2024}},
  publisher={{Hugging Face}},
  howpublished={{\\url{{https://huggingface.co/datasets/luisml77/{repo_name}}}}}
}}
```

## License

[Specify your license here]
"""
    
    return card_content


def calculate_statistics(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate basic statistics about the dataset"""
    stats = {
        'total_samples': len(samples),
        'splits': {}
    }
    
    # Count by split
    for sample in samples:
        split_name = sample.get('split', 'train')
        stats['splits'][split_name] = stats['splits'].get(split_name, 0) + 1
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Upload JSONL dataset to Hugging Face Hub')
    parser.add_argument('--jsonl_path', type=str, required=True,
                       help='Path to the JSONL file')
    parser.add_argument('--repo_name', type=str, required=True,
                       help='Name for the Hugging Face repository')
    parser.add_argument('--username', type=str, default='luisml77',
                       help='Hugging Face username')
    parser.add_argument('--private', action='store_true',
                       help='Make the repository private')
    parser.add_argument('--push_to_hub', action='store_true',
                       help='Actually push to Hugging Face Hub (otherwise just prepare locally)')
    parser.add_argument('--split', type=str, default=None,
                       help='Process only a specific split (train/val/test). If not specified, creates DatasetDict with all splits.')
    
    args = parser.parse_args()
    
    # Validate JSONL path
    if not os.path.exists(args.jsonl_path):
        logger.error(f"JSONL file does not exist: {args.jsonl_path}")
        return
    
    logger.info(f"Processing JSONL dataset: {args.jsonl_path}")
    logger.info(f"Target repository: {args.username}/{args.repo_name}")
    
    try:
        if args.split:
            # Create single dataset for specific split
            logger.info(f"Creating dataset for split: {args.split}")
            dataset = create_dataset_from_jsonl(args.jsonl_path, split_name=args.split)
            dataset_dict = DatasetDict({args.split: dataset})
        else:
            # Create DatasetDict with all splits
            logger.info("Creating DatasetDict with all splits...")
            dataset_dict = create_dataset_dict_from_jsonl(args.jsonl_path)
        
        # Calculate statistics
        samples = load_jsonl(args.jsonl_path)
        stats = calculate_statistics(samples)
        logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
        
        if args.push_to_hub:
            logger.info(f"Pushing dataset to Hugging Face Hub: {args.username}/{args.repo_name}")
            
            # Create dataset card
            card_content = create_dataset_card(args.repo_name, stats)
            
            # Push to hub
            dataset_dict.push_to_hub(
                f"{args.username}/{args.repo_name}",
                private=args.private
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
            logger.info(f"Dataset splits: {list(dataset_dict.keys())}")
            for split_name, split_dataset in dataset_dict.items():
                logger.info(f"  {split_name}: {len(split_dataset)} samples")
    
    except Exception as e:
        logger.error(f"Error processing dataset: {e}", exc_info=True)
        return


if __name__ == '__main__':
    main()
