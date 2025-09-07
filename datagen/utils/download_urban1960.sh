#!/bin/bash

# Download Urban1960SatBench dataset
echo "Downloading Urban1960SatBench dataset..."
wget -O Urban1960SatBench_full.zip "https://dataverse.harvard.edu/api/access/dataset/:persistentId?persistentId=doi:10.7910/DVN/HT2B1S&format=original"

# Check if download was successful
if [ ! -f "Urban1960SatBench_full.zip" ]; then
    echo "Error: Download failed"
    exit 1
fi

# Create Urban1960SatBench directory
mkdir -p Urban1960SatBench

# Extract the main zip file
echo "Extracting main zip file..."
unzip Urban1960SatBench_full.zip -d Urban1960SatBench/

# Find and extract nested zip files
echo "Extracting nested zip files..."
cd Urban1960SatBench/
for zipfile in *.zip; do
    if [ -f "$zipfile" ]; then
        echo "Extracting $zipfile..."
        unzip "$zipfile"
        rm "$zipfile"
    fi
done
cd ..

# Clean up the main zip file
echo "Cleaning up zip files..."
rm Urban1960SatBench_full.zip

echo "Dataset extraction complete. Contents are in Urban1960SatBench/ directory"