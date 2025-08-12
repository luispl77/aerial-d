# AERIAL-D Dataset Upload to Hugging Face

This script uploads the AERIAL-D dataset to Hugging Face Hub with proper formatting and metadata preservation.

## Features

- **Excludes DeepGlobe data**: Only processes iSAID (P prefix) and LoveDA (L prefix) files
- **Preserves all metadata**: Expression types (original, enhanced, unique), RLE masks, domain information
- **Proper dataset schema**: Structured for easy use with Hugging Face datasets library
- **Automatic documentation**: Generates comprehensive dataset card with statistics

## Installation

Install the required dependencies:

```bash
pip install -r requirements_upload.txt
```

## Usage

### Basic Usage (with default paths)

```bash
python upload_to_huggingface.py --push_to_hub
```

This will:
- Use default paths:
  - Annotations: `/cfs/home/u035679/aerialseg/datagen/dataset/patches_rules_expressions_unique`
  - Images: `/cfs/home/u035679/aerialseg/datagen/dataset/patches`
- Upload to `luisml77/aerial-d` on Hugging Face
- Make the repository public

### Custom Configuration

```bash
python upload_to_huggingface.py \
    --annotations_path /path/to/annotations \
    --images_path /path/to/images \
    --repo_name my-aerial-dataset \
    --username myusername \
    --private \
    --push_to_hub
```

### Testing (without uploading)

```bash
python upload_to_huggingface.py --max_samples_per_split 100
```

This will process only 100 samples per split and prepare the dataset locally without uploading.

## Arguments

- `--annotations_path`: Path to annotations directory (default: patches_rules_expressions_unique)
- `--images_path`: Path to images directory (default: patches)
- `--repo_name`: Repository name on Hugging Face (default: aerial-d)
- `--username`: Hugging Face username (default: luisml77)
- `--private`: Make repository private
- `--push_to_hub`: Actually upload to Hugging Face (otherwise just prepare locally)
- `--max_samples_per_split`: Limit samples for testing

## Dataset Structure

The script processes:

### Individual Objects
- Each object with its segmentation mask
- Multiple expressions per object (original, enhanced, unique)
- Bounding boxes, categories, and metadata

### Groups
- Collections of objects with group-level expressions
- Group segmentation masks
- Spatial relationships and centroid information

### Output Format

Each sample contains:
- **image**: PIL Image object
- **expression_text**: Natural language description
- **expression_type**: 'original', 'enhanced', or 'unique'
- **binary_mask**: 2D array segmentation mask
- **rle_mask**: RLE format mask
- **category**: Object category
- **domain_info**: Source dataset (iSAID or LoveDA)
- **Additional metadata**: Bounding boxes, areas, spatial info

## Authentication

Make sure you're logged into Hugging Face:

```bash
huggingface-cli login
```

## Example Output

The uploaded dataset will be available at:
`https://huggingface.co/datasets/{username}/{repo_name}`

You can then use it in your code:

```python
from datasets import load_dataset

dataset = load_dataset("luisml77/aerial-d")
sample = dataset['train'][0]

image = sample['image']
expression = sample['expression_text']
mask = sample['binary_mask']
category = sample['category']
```

## Filtering

The script automatically:
- Excludes all DeepGlobe files (starting with 'D')
- Includes only iSAID (P prefix) and LoveDA (L prefix) files
- Preserves all expression types and metadata
- Maintains train/validation splits

## Dataset Statistics

The script generates comprehensive statistics including:
- Total samples and images
- Expression type distribution
- Domain distribution (iSAID vs LoveDA)
- Category distribution
- Individual vs group object counts

These statistics are included in the generated dataset card on Hugging Face.
