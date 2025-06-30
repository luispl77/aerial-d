# Remote Sensing Referring Expression Instance Segmentation (RRSIS) Dataset Generation Pipeline

This repository contains the code for generating a new Referring Expression Instance Segmentation dataset based on the iSAID dataset for remote sensing imagery.

## Overview

The pipeline processes multiple remote sensing datasets through several steps:
1. Crop iSAID images into smaller patches  
2. Crop LoveDA images into smaller patches
3. Crop DeepGlobe road images into smaller patches
4. Annotate instances with position and relationship information
5. Generate rule-based referring expressions
6. Filter expressions for uniqueness
7. Apply historic filtering

## iSAID Data Structure

The pipeline expects the iSAID dataset in the following structure:

```
.
  |-isaid
  |  |-test
  |  |  |-images
  |  |  |  |-images
  |  |-train
  |  |  |-Annotations
  |  |  |-images
  |  |  |  |-images
  |  |  |-Instance_masks
  |  |  |  |-images
  |  |-val
  |  |  |-Annotations
  |  |  |-images
  |  |  |  |-images
  |  |  |-Instance_masks
  |  |  |  |-images
```

## Pipeline Steps

### 1. Generate Patches from iSAID

The first step breaks down the large satellite images from iSAID into smaller, manageable patches. This module extracts patches containing a controlled number of instances, maintaining complete instances without cutoffs at patch boundaries.

```bash
python pipeline/1_isaid_patches.py
```

**Key Features**:
- Creates 480×480 pixel patches using sliding window approach
- Generates 50 patches by default
- Uses 20% window overlap
- Creates debug visualizations
- Outputs to `dataset/patches`

### 2. Generate Patches from LoveDA

This step processes LoveDA dataset images into smaller patches compatible with the pipeline.

```bash
python pipeline/2_loveda_patches.py
```

### 3. Generate Patches from DeepGlobe

This step processes DeepGlobe road extraction dataset images into smaller patches compatible with the pipeline.

```bash
python pipeline/3_deepglobe_patches.py
```

**Key Features**:
- Processes road extraction masks (binary classification)
- Creates 480×480 pixel patches using sliding window approach or full image resize
- Extracts connected components for road instances
- Generates XML annotations compatible with the pipeline format
- Outputs to `dataset/patches` with prefix "D" for DeepGlobe images

### 4. Add Position and Spatial Relationship Information

This module enriches each instance with spatial context by dividing the image into a 3×3 grid and labeling where each object is located. It also calculates relational information between nearby objects based on relative positions.

```bash
python pipeline/4_add_rules.py
```

**Key Features**:
- Uses a 3×3 grid with "no-man's land" around center (20% alpha value)
- Calculates spatial relationships between objects within 200px distance
- Identifies extreme positions (topmost, bottommost, leftmost, rightmost) when objects are sufficiently separated
- Determines size attributes (largest/smallest) within each object category when size differences are significant
- Groups related objects using DBSCAN clustering (e.g., vehicle clusters, building complexes)
- Creates debug visualizations of grid positions, relationships, and groups
- Outputs to `dataset/patches_rules`

### 5. Generate Raw Referring Expressions

Using the position and relationship information, this module creates rule-based referring expressions for each instance combining category names with positional and relational attributes.

```bash
python pipeline/5_generate_all_expressions.py
```

**Key Features**:
- Generates multiple types of expressions by combining different attributes:
  1. Category only (e.g., "the ship")
  2. Category + position (e.g., "the ship in the bottom right")
  3. Category + position + relationship (e.g., "the ship in the bottom right that is to the left of a harbor")
  4. Extreme position + category (e.g., "the topmost ship")
  5. Size attribute + category (e.g., "the largest ship")
  6. Combinations of all above attributes
- Creates debug visualizations of expressions with color-coded instances
- Outputs text files listing all generated expressions per image
- Outputs to `dataset/patches_rules_expressions`

### 6. Filter Expressions for Uniqueness

This module filters the expressions to ensure each one refers uniquely to exactly one instance in the image. It removes any expression that appears for multiple objects.

```bash
python pipeline/6_filter_unique.py
```

**Key Features**:
- Ensures each expression uniquely identifies exactly one object
- Standardizes class names (e.g., "Large_Vehicle" → "large vehicle")
- Creates debug visualizations showing filtered vs. raw expressions
- Outputs to `dataset/patches_rules_expressions_unique`

### 7. Apply Historic Filtering

The final filtering step applies historic processing to the dataset.

```bash
python pipeline/7_historic_filter.py
```

## Final Dataset Structure

```
dataset/patches_rules_expressions_unique_historic/
  ├── annotations/
  │   └── *.xml
  └── debug/
      └── *.json (debug outputs from processing)
```

Each XML annotation includes:
- Basic instance information (bbox, segmentation, category)
- Grid position and spatial relationships
- Filtered unique referring expressions
- Historic processing metadata


