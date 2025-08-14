---
language: en
tags:
- computer-vision
- instance-segmentation
- referring-expression-segmentation  
- aerial-imagery
- remote-sensing
- xml-annotations
task_categories:
- image-segmentation
license: apache-2.0
size_categories:
- 10K<n<100K
---

# AERIAL-D: Referring Expression Segmentation in Aerial Imagery

**AERIAL-D** is a comprehensive dataset for Referring Expression Instance Segmentation (RRSIS) in aerial and satellite imagery. The dataset contains high-resolution aerial photos (480×480 patches) with detailed instance segmentation masks and natural language referring expressions that describe specific objects within the images.

🗂️ **Dataset Structure**: Due to Hugging Face's file limit constraints, the dataset is provided as a zip file. Please download and extract to use.

## 📊 Dataset Statistics

- **37,288 patches** total (27,480 train + 9,808 val)
- **128,715 object instances** with referring expressions
- **130,994 groups** with collective expressions  
- **1,522,523 total expressions** across all types
- **Multiple domains**: iSAID (P prefix) and LoveDA (L prefix)
- **Expression distribution**: 318,591 original + 313,323 enhanced + 257,440 unique

## 🏗️ Dataset Structure

```
aeriald/
├── train/
│   ├── annotations/     # XML annotation files
│   │   ├── L0_patch_0.xml
│   │   ├── P0001_patch_000001.xml
│   │   └── ...
│   └── images/          # PNG image files (480×480)
│       ├── L0_patch_0.png
│       ├── P0001_patch_000001.png
│       └── ...
└── val/
    ├── annotations/
    └── images/
```

## 🏷️ Object Categories

The dataset includes diverse aerial imagery categories:

**iSAID Categories** (P prefix):
- `plane`, `ship`, `storage tank`, `baseball diamond`, `tennis court`
- `swimming pool`, `roundabout`, `harbor`, `bridge`, `large vehicle`, `small vehicle`
- `helicopter`, `roundabout`, `soccer ball field`, `ground track field`

**LoveDA Categories** (L prefix):  
- `building`, `water`, `agriculture`, `forest`, `road`, `barren`

## 📝 XML Annotation Format

Each image has a corresponding XML file with the following structure:

```xml
<?xml version='1.0' encoding='utf-8'?>
<annotation>
  <filename>L0_patch_0.png</filename>
  <size>
    <width>480</width>
    <height>480</height>
  </size>
  
  <!-- Individual Objects -->
  <object>
    <name>building</name>
    <bndbox>
      <xmin>0</xmin>
      <ymin>0</ymin>
      <xmax>43</xmax>
      <ymax>21</ymax>
    </bndbox>
    <id>1</id>
    <segmentation>{'size': [480, 480], 'counts': 'RLE_ENCODED_MASK'}</segmentation>
    <area>494</area>
    <possible_colors>light,dark</possible_colors>
    <expressions>
      <expression id="0">the dark topmost building</expression>
      <expression id="1">the dark topmost building in the top left</expression>
      <expression type="enhanced">the darkest building at the very top</expression>
      <expression type="unique">the highest dark building on the upper left</expression>
    </expressions>
  </object>
  
  <!-- Group Annotations -->
  <groups>
    <group>
      <id>1000000</id>
      <size>3</size>
      <centroid>
        <x>44.0</x>
        <y>240.0</y>
      </centroid>
      <category>building</category>
      <segmentation>{'size': [480, 480], 'counts': 'GROUP_RLE_MASK'}</segmentation>
      <expressions>
        <expression id="0">all buildings in the image</expression>
        <expression type="enhanced">every building shown in the picture</expression>
        <expression type="unique">all structures from red houses to grey buildings</expression>
      </expressions>
    </group>
  </groups>
</annotation>
```

## 🎯 Expression Types

1. **Original** (`id="0"`, `id="1"`): Rule-based generated expressions using spatial and visual rules
   - `"the dark topmost building"`  
   - `"the water in the bottom center"`

2. **Enhanced** (`type="enhanced"`): LLM-enhanced expressions that vary the language of original expressions while maintaining the same meaning
   - **1 enhanced per original expression**
   - `"the darkest building at the very top"` (enhanced from "the dark topmost building")
   - `"every building shown in the picture"` (enhanced from "all buildings in the image")

3. **Unique** (`type="unique"`): LLM-generated expressions that capture new visual details seen by the LLM, providing distinctive identifying information
   - **2 unique expressions per target** (regardless of number of original expressions)
   - `"the highest dark building on the upper left"`
   - `"the pond flanked by trees on the left and a ruined shed on the right"`

## 💻 Usage Example

```python
import xml.etree.ElementTree as ET
from PIL import Image
from pycocotools import mask as mask_utils
import numpy as np

# Load an annotation
tree = ET.parse('aeriald/train/annotations/L0_patch_0.xml')
root = tree.getroot()

# Load corresponding image
image_path = 'aeriald/train/images/L0_patch_0.png'
image = Image.open(image_path)

# Extract objects and expressions
for obj in root.findall('object'):
    category = obj.find('name').text
    
    # Get expressions
    expressions = obj.find('expressions')
    for expr in expressions.findall('expression'):
        expression_text = expr.text
        expression_type = expr.get('type', 'original')
        print(f"{category}: {expression_text} (type: {expression_type})")
    
    # Decode segmentation mask
    seg_text = obj.find('segmentation').text
    rle_mask = eval(seg_text)  # Parse RLE format
    binary_mask = mask_utils.decode(rle_mask)
```

## 🔍 Key Features

- **Multi-scale Referring Expressions**: From simple object names to complex spatial relationships
- **RLE Segmentation Masks**: Efficient storage format compatible with COCO tools  
- **Bounding Boxes**: Standard object detection format
- **Group Annotations**: Collective referring expressions for multiple objects
- **Spatial Relationships**: Positional descriptions (top-left, bottom-right, etc.)
- **Multi-domain**: Combines urban (iSAID) and rural (LoveDA) aerial imagery

## 📚 Applications

- **Referring Expression Segmentation (RES)**
- **Open-vocabulary semantic segmentation**  
- **Vision-language understanding in remote sensing**
- **Multimodal learning with aerial imagery**
- **Zero-shot object detection and segmentation**

## 📁 Download Instructions

1. Download the `aeriald.zip` file from this repository
2. Extract the zip file: `unzip aeriald.zip`
3. The dataset will be available in the `aeriald/` directory with train/val splits

## 🏗️ Technical Details

- **Image Size**: 480×480 pixels
- **Format**: PNG (images), XML (annotations)
- **Coordinate System**: Standard image coordinates (top-left origin)
- **Mask Format**: RLE (Run-Length Encoding) compatible with pycocotools
- **Text Encoding**: UTF-8

## 📜 Citation

```bibtex
@dataset{aerial-d-2024,
  title={AERIAL-D: Referring Expression Instance Segmentation in Aerial Imagery},
  author={[Your Name]},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/luisml77/aerial-d}
}
```

## 🤝 Acknowledgments

This dataset builds upon the iSAID and LoveDA datasets, enhanced with rule-based and LLM-generated referring expressions for comprehensive aerial image understanding.