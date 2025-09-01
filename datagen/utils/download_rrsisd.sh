#!/bin/bash

# Create directory structure
mkdir -p rrsisd/images/rrsisd

# Install gdown if not already installed
pip install gdown

# Download refs(unc).p
echo "Downloading refs(unc).p..."
gdown https://drive.google.com/uc?id=1oRILB_o_LTH2jwr1isF6rDOLZP4YgIoV -O rrsisd/refs\(unc\).p

# Download instances.json
echo "Downloading instances.json..."
gdown https://drive.google.com/uc?id=16u61hnDzVqt6pmnIeYsaWdyYpXCdzAq9 -O rrsisd/instances.json

# Download and extract JPEGImages
echo "Downloading JPEGImages.zip..."
gdown https://drive.google.com/uc?id=1j-0tri3oTroD4iY4IugK5XuBxWAOwjgc -O JPEGImages.zip
echo "Extracting JPEGImages.zip..."
unzip -q JPEGImages.zip -d rrsisd/images/rrsisd
rm JPEGImages.zip

# Download and extract annotations
echo "Downloading annotations.zip..."
gdown https://drive.google.com/uc?id=1heyS10o0xa2lkDpSCBNEl-R0fZ7njA6W -O ann_split.zip
echo "Extracting ann_split.zip..."
unzip -q ann_split.zip -d rrsisd/images/rrsisd
rm ann_split.zip

echo "Download and extraction complete. Dataset is organized according to the required structure." 