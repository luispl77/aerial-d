# AERIAL-D Dataset Upload Guide

## What you need to upload to Hugging Face:

1. **The dataset folder structure**: `/cfs/home/u035679/datasets/aeriald/`
2. **The dataset loading script**: `dataset.py` 
3. **Optional**: The zip file `aeriald.zip` as backup

## Upload Steps:

### Method 1: Using Hugging Face Hub (Recommended)

1. **Create a new dataset repository on Hugging Face**:
   - Go to https://huggingface.co/new-dataset
   - Name it `aerial-d` (or your preferred name)
   - Make it public
   - Create repository

2. **Clone the repository locally**:
   ```bash
   git clone https://huggingface.co/datasets/YOUR_USERNAME/aerial-d
   cd aerial-d
   ```

3. **Copy your files**:
   ```bash
   # Copy the dataset structure
   cp -r /cfs/home/u035679/datasets/aeriald/* .
   
   # Copy the loading script
   cp /cfs/home/u035679/aerialseg/datagen/utils/dataset.py .
   
   # Optional: Copy the zip file as backup
   cp /cfs/home/u035679/datasets/aeriald.zip .
   ```

4. **Create a README.md** (or let the dataset.py generate it automatically):
   ```bash
   echo "# AERIAL-D Dataset" > README.md
   echo "Referring Expression Segmentation in Aerial Imagery" >> README.md
   ```

5. **Upload to Hub**:
   ```bash
   git add .
   git commit -m "Upload AERIAL-D dataset with loading script"
   git push
   ```

### Method 2: Using the Web Interface

1. Create a new dataset repository on Hugging Face
2. Upload files via the web interface:
   - Upload `dataset.py` to the root
   - Upload the entire `aeriald` folder contents
   - Maintain the directory structure: `train/annotations/`, `train/images/`, `val/annotations/`, `val/images/`

## How users will load your dataset:

```python
from datasets import load_dataset

# Load the full dataset
dataset = load_dataset("YOUR_USERNAME/aerial-d")

# Load specific configuration  
enhanced_only = load_dataset("YOUR_USERNAME/aerial-d", "enhanced_only")

# Load specific split
train_data = load_dataset("YOUR_USERNAME/aerial-d", split="train")

# Example usage
sample = dataset['train'][0]
image = sample['image']
expression = sample['expression_text']
rle_mask = sample['rle_mask']

# Decode mask if needed
from pycocotools import mask as mask_utils
binary_mask = mask_utils.decode(rle_mask)
```

## File Structure on Hugging Face:

```
aerial-d/
├── dataset.py              # Loading script (REQUIRED)
├── README.md              # Auto-generated or manual
├── train/
│   ├── annotations/       # XML files
│   │   ├── L0_patch_0.xml
│   │   └── ...
│   └── images/           # PNG files  
│       ├── L0_patch_0.png
│       └── ...
├── val/
│   ├── annotations/      
│   └── images/
└── aeriald.zip          # Optional backup
```

## Key Points:

- **The `dataset.py` file is essential** - this tells Hugging Face how to load your data
- **Maintain the directory structure** - train/val with annotations/images subdirs
- **The script automatically filters out DeepGlobe files** (files starting with 'D')
- **Multiple configurations available**: default, enhanced_only, unique_only
- **Users don't need to manually download** - `load_dataset()` handles everything

## Testing locally (optional):

You can test the dataset loading script locally:

```python
from datasets import load_dataset

# Point to your local dataset.py
dataset = load_dataset("/path/to/your/dataset.py", data_dir="/cfs/home/u035679/datasets/aeriald")
print(dataset)
```