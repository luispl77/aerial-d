import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import argparse

def get_total_files(directories):
    total = 0
    for directory in directories:
        for root, _, files in os.walk(directory):
            total += len(files)
    return total

def create_dataset_zip(base_dir, zip_path):
    # Convert to Path objects
    base_dir = Path(base_dir)
    zip_path = Path(zip_path)
    
    # Define source directories
    train_images_dir = base_dir / "patches" / "train" / "images"
    val_images_dir = base_dir / "patches" / "val" / "images"
    train_ann_dir = base_dir / "patches_rules_expressions_unique" / "train" / "annotations"
    val_ann_dir = base_dir / "patches_rules_expressions_unique" / "val" / "annotations"
    
    # Verify directories exist
    for dir_path in [train_images_dir, val_images_dir, train_ann_dir, val_ann_dir]:
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {dir_path}")
    
    # Calculate total files for progress bar
    directories = [train_images_dir, val_images_dir, train_ann_dir, val_ann_dir]
    total_files = get_total_files(directories)
    
    # Create zip file
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        pbar = tqdm(total=total_files, desc="Zipping dataset")
        
        # Add train images
        for root, _, files in os.walk(train_images_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join("aeriald", "train", "images", 
                                     os.path.relpath(file_path, train_images_dir))
                zipf.write(file_path, arcname)
                pbar.update(1)
        
        # Add val images
        for root, _, files in os.walk(val_images_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join("aeriald", "val", "images", 
                                     os.path.relpath(file_path, val_images_dir))
                zipf.write(file_path, arcname)
                pbar.update(1)
        
        # Add train annotations
        for root, _, files in os.walk(train_ann_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join("aeriald", "train", "annotations", 
                                     os.path.relpath(file_path, train_ann_dir))
                zipf.write(file_path, arcname)
                pbar.update(1)
        
        # Add val annotations
        for root, _, files in os.walk(val_ann_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.join("aeriald", "val", "annotations", 
                                     os.path.relpath(file_path, val_ann_dir))
                zipf.write(file_path, arcname)
                pbar.update(1)
        
        pbar.close()
    
    print(f"Dataset has been zipped to {zip_path} with simplified structure:")
    print("  aeriald/")
    print("  ├── train/")
    print("  │   ├── images/")
    print("  │   └── annotations/")
    print("  └── val/")
    print("      ├── images/")
    print("      └── annotations/")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Zip the dataset with a specific structure')
    parser.add_argument('--base_dir', type=str, default='dataset',
                      help='Base directory containing the dataset (default: dataset)')
    parser.add_argument('--zip_path', type=str, default='aeriald.zip',
                      help='Path where the zip file will be saved (default: aeriald.zip)')
    
    args = parser.parse_args()
    create_dataset_zip(args.base_dir, args.zip_path) 