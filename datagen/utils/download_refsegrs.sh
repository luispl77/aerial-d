#!/bin/bash

# Download RefSegRS dataset from Hugging Face
# https://huggingface.co/datasets/JessicaYuan/RefSegRS

set -e

# Create refsegrs directory if it doesn't exist
mkdir -p refsegrs

# Download the dataset zip file
echo "Downloading RefSegRS.zip..."
wget -O RefSegRS.zip https://huggingface.co/datasets/JessicaYuan/RefSegRS/resolve/main/RefSegRS.zip

# Extract the zip file to refsegrs folder
echo "Extracting RefSegRS.zip to refsegrs folder..."
unzip -o RefSegRS.zip -d refsegrs/

# Clean up the zip file
rm RefSegRS.zip

echo "RefSegRS dataset downloaded and extracted to refsegrs/ folder"