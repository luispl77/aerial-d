#!/bin/bash

# Download the DeepGlobe Road Extraction dataset from Kaggle
# Usage: ./download_deepglobe.sh [destination_dir]
# 
# Prerequisites:
# 1. Install kaggle API: pip install kaggle
# 2. Setup Kaggle credentials: Place kaggle.json in ~/.kaggle/ or set KAGGLE_USERNAME and KAGGLE_KEY
# 3. Accept the dataset terms on Kaggle website: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset

set -e

BASE_DIR="${1:-$(pwd)/deepglobe}"
DATASET_NAME="balraj98/deepglobe-road-extraction-dataset"

echo "============================================="
echo "Downloading DeepGlobe Road Extraction Dataset"
echo "============================================="

# Check if kaggle is installed
if ! command -v kaggle &> /dev/null; then
    echo "Error: Kaggle API not found. Installing..."
    pip install kaggle
fi

# Check if kaggle credentials are configured
if ! kaggle datasets list &> /dev/null; then
    echo "Error: Kaggle credentials not configured!"
    echo "Please do one of the following:"
    echo "1. Place your kaggle.json file in ~/.kaggle/"
    echo "2. Set environment variables KAGGLE_USERNAME and KAGGLE_KEY"
    echo "3. Make sure you've accepted the dataset terms at: https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset"
    exit 1
fi

# Create base directory
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

echo "Downloading DeepGlobe dataset to: $BASE_DIR"

# Download the dataset
if [ -d "deepglobe-road-extraction-dataset" ] || [ -f "deepglobe-road-extraction-dataset.zip" ]; then
    echo "Dataset files already exist, skipping download."
else
    echo "Downloading dataset from Kaggle..."
    kaggle datasets download -d "$DATASET_NAME"
    
    # Extract the dataset
    echo "Extracting dataset..."
    unzip -o deepglobe-road-extraction-dataset.zip -d .
    
    # Clean up zip file
    rm deepglobe-road-extraction-dataset.zip
fi

# Organize the dataset structure
echo "Organizing dataset structure..."

# Check what was extracted and organize accordingly
if [ -d "deepglobe-road-extraction-dataset" ]; then
    # Move contents up one level if needed
    mv deepglobe-road-extraction-dataset/* .
    rmdir deepglobe-road-extraction-dataset
fi

# Remove test and valid folders (valid has no masks, we'll split train internally)
if [ -d "test" ]; then
    echo "Removing test folder..."
    rm -rf test
fi

if [ -d "valid" ]; then
    echo "Removing valid folder (no masks available, will split train set internally)..."
    rm -rf valid
fi

# Print dataset structure
echo ""
echo "Dataset downloaded successfully!"
echo "Dataset structure (train only - will be split internally):"
find . -maxdepth 2 -type d | sort

echo ""
echo "DeepGlobe Road Extraction dataset downloaded to: $BASE_DIR"
echo "=============================================" 