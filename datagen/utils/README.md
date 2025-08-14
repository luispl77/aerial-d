---
language: en
tags:
- computer-vision
- instance-segmentation
- referring-expression-segmentation  
- aerial-imagery
task_categories:
- image-segmentation
license: apache-2.0
---

# AERIAL-D: Referring Expression Segmentation in Aerial Imagery

Dataset for referring expression segmentation in aerial and satellite imagery with XML annotations and natural language descriptions.

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("luisml77/aerial-d")

# Example usage
sample = dataset['train'][0]
image = sample['image']
expression = sample['expression_text']
rle_mask = sample['rle_mask']

# Decode mask if needed
from pycocotools import mask as mask_utils
binary_mask = mask_utils.decode(rle_mask)
```

## Features

- Multiple expression types (original, enhanced, unique)
- XML annotations with segmentation masks in RLE format
- Multi-domain data from iSAID and LoveDA datasets
- Bounding boxes and object categories