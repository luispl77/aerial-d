#!/bin/bash

# Set base directory to current directory
BASE_DIR="$(pwd)/isaid"

# Install gdown if not already installed
pip install gdown

# Create all necessary directories
mkdir -p $BASE_DIR/train/Annotations
mkdir -p $BASE_DIR/train/Instance_masks
mkdir -p $BASE_DIR/train/images
mkdir -p $BASE_DIR/val/Annotations
mkdir -p $BASE_DIR/val/Instance_masks
mkdir -p $BASE_DIR/val/images
mkdir -p $BASE_DIR/test/images

# TRAIN DATA
# Download train annotations
echo "Downloading train annotations..."
gdown 1-PYSXak2JBg3xuZWzAWVXfKQO1TkPCqF -O $BASE_DIR/train/Annotations/instances_train.json

# Download and extract train instance masks
echo "Downloading and extracting train instance masks..."
gdown 12XhSgEt_Xw4FJQxLJZgMutw3awoAq2Ve -O $BASE_DIR/train/Instance_masks/instances_masks.zip
unzip $BASE_DIR/train/Instance_masks/instances_masks.zip -d $BASE_DIR/train/Instance_masks/
rm $BASE_DIR/train/Instance_masks/instances_masks.zip

# Download train images (3 parts)
echo "Downloading train images (part 1/3)..."
gdown 1BlaGYNNEKGmT6OjZjsJ8HoUYrTTmFcO2 -O $BASE_DIR/train/images/train_part1.zip
echo "Extracting train images (part 1/3)..."
unzip $BASE_DIR/train/images/train_part1.zip -d $BASE_DIR/train/images/
rm $BASE_DIR/train/images/train_part1.zip

echo "Downloading train images (part 2/3)..."
gdown 1JBWCHdyZOd9ULX0ng5C9haAt3FMPXa3v -O $BASE_DIR/train/images/train_part2.zip
echo "Extracting train images (part 2/3)..."
unzip $BASE_DIR/train/images/train_part2.zip -d $BASE_DIR/train/images/
rm $BASE_DIR/train/images/train_part2.zip

echo "Downloading train images (part 3/3)..."
gdown 1pEmwJtugIWhiwgBqOtplNUtTG2T454zn -O $BASE_DIR/train/images/train_part3.zip
echo "Extracting train images (part 3/3)..."
unzip $BASE_DIR/train/images/train_part3.zip -d $BASE_DIR/train/images/
rm $BASE_DIR/train/images/train_part3.zip

# VALIDATION DATA
# Download val annotations
echo "Downloading validation annotations..."
gdown 1QDKeAQ8Ka6_wxoN3Ld5t3EP9UHk7Fw98 -O $BASE_DIR/val/Annotations/instances_val.json

# Download and extract val instance masks
echo "Downloading and extracting validation instance masks..."
gdown 1GCExuFqEKOY5Hyp1WSAmdW6I4RAaxBGG -O $BASE_DIR/val/Instance_masks/instances_masks.zip
unzip $BASE_DIR/val/Instance_masks/instances_masks.zip -d $BASE_DIR/val/Instance_masks/
rm $BASE_DIR/val/Instance_masks/instances_masks.zip

# Download val images
echo "Downloading and extracting validation images..."
gdown 1uCCCFhFQOJLfjBpcL5MC0DHJ9lgOaXWP -O $BASE_DIR/val/images/val.zip
unzip $BASE_DIR/val/images/val.zip -d $BASE_DIR/val/images/
rm $BASE_DIR/val/images/val.zip

# TEST DATA
# Download test images
echo "Downloading test images (part 1/2)..."
gdown 1fwiTNqRRen09E-O9VSpcMV2e6_d4GGVK -O $BASE_DIR/test/images/images_part1.zip
echo "Extracting test images (part 1/2)..."
unzip $BASE_DIR/test/images/images_part1.zip -d $BASE_DIR/test/images/
rm $BASE_DIR/test/images/images_part1.zip

echo "Downloading test images (part 2/2)..."
gdown 1wTwmxvPVujh1I6mCMreoKURxCUI8f-qv -O $BASE_DIR/test/images/images_part2.zip
echo "Extracting test images (part 2/2)..."
unzip $BASE_DIR/test/images/images_part2.zip -d $BASE_DIR/test/images/
rm $BASE_DIR/test/images/images_part2.zip

# Download test info
echo "Downloading test info..."
gdown 1nQokIxSy3DEHImJribSCODTRkWlPJLE3 -O $BASE_DIR/test_info.json

echo "iSAID dataset successfully downloaded to $BASE_DIR"
echo "Done!"