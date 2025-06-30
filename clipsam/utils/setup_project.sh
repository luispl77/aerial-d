#!/bin/bash

# Simple setup script
echo "=== Project Setup Script ==="

# Get token from argument
HF_TOKEN=$1

if [ -z "$HF_TOKEN" ]; then
    echo "Usage: ./setup_project.sh <hf_token>"
    exit 1
fi

echo "Setting up project..."

# Install fast transfer for downloads
echo "Installing hf_transfer for faster downloads..."
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1

# Download dataset
echo "Downloading dataset..."
huggingface-cli download luisml77/aeriald aeriald.zip --repo-type=dataset --local-dir ./ --token ${HF_TOKEN}

# Unzip dataset
echo "Extracting dataset..."
unzip aeriald.zip
rm aeriald.zip

# Download models
echo "Downloading models..."
huggingface-cli download luisml77/aerial-seg --repo-type=model --local-dir ./models --token ${HF_TOKEN}

# Install requirements if they exist
if [ -f "requirements.txt" ]; then
    echo "Installing requirements..."
    pip install -r requirements.txt
fi

echo "Setup complete!" 