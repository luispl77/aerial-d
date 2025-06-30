#!/bin/bash

# Download the LoveDA dataset from Zenodo and extract it.
# Usage: ./download_loveda.sh [destination_dir]

set -e

BASE_DIR="${1:-$(pwd)/LoveDA}"
mkdir -p "$BASE_DIR"
cd "$BASE_DIR"

TRAIN_URL="https://zenodo.org/records/5706578/files/Train.zip?download=1"
VAL_URL="https://zenodo.org/records/5706578/files/Val.zip?download=1"

function download_file() {
    local url="$1"
    local output="$2"
    if [ -f "$output" ]; then
        echo "$output already exists, skipping download."
    else
        echo "Downloading $output ..."
        wget -c "$url" -O "$output"
    fi
}

download_file "$TRAIN_URL" "Train.zip"
download_file "$VAL_URL" "Val.zip"
echo "Extracting archives..."
unzip -o Train.zip -d .
unzip -o Val.zip -d .

rm Train.zip Val.zip

echo "LoveDA dataset downloaded to $BASE_DIR"

