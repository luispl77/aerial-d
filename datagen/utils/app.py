import os
import io
import random
import xml.etree.ElementTree as ET
import textwrap
from flask import Flask, render_template_string, send_file, request
import matplotlib.pyplot as plt
import cv2
import numpy as np
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='LLM Expression Viewer')
parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                    help='Dataset split to use (default: train)')
parser.add_argument('--port', type=int, default=5001,
                    help='Port to run the server on (default: 5001)')
parser.add_argument('--dataset', type=str, default='dataset',
                    help='Dataset folder name (default: dataset)')
args = parser.parse_args()

app = Flask(__name__)

# Set global default split from command line
DEFAULT_SPLIT = args.split

# Update paths to include split-specific directories
ANNOTATIONS_DIR = os.path.join(args.dataset, "patches_rules_expressions_unique_llm")
IMAGES_DIR = os.path.join(args.dataset, "patches")

# Cache for image listings
IMAGE_LISTINGS = {}

def get_image_numbers(split=None):
    if split is None:
        split = DEFAULT_SPLIT
    """Get list of unique image numbers for a split"""
    if split not in IMAGE_LISTINGS:
        split_annotations_dir = os.path.join(ANNOTATIONS_DIR, split, 'annotations')
        # Check if directory exists
        if not os.path.exists(split_annotations_dir):
            print(f"Warning: No annotations found for split '{split}' at {split_annotations_dir}")
            return []
            
        # Extract unique image numbers from filenames
        image_numbers = set()
        for f in os.listdir(split_annotations_dir):
            if f.endswith('.xml'):
                # Extract image number from filename (e.g., P0000_patch_000059.xml -> 0000)
                try:
                    image_num = f.split('_')[0][1:]  # Remove 'P' prefix
                    image_numbers.add(image_num)
                except:
                    continue
        IMAGE_LISTINGS[split] = sorted(list(image_numbers))
    return IMAGE_LISTINGS[split]

def get_patches_for_image(image_num, split=None):
    if split is None:
        split = DEFAULT_SPLIT
    """Get all patches for a specific image number"""
    split_annotations_dir = os.path.join(ANNOTATIONS_DIR, split, 'annotations')
    patches = []
    
    # Find all XML files for this image number
    for f in os.listdir(split_annotations_dir):
        if f.startswith(f'P{image_num}_patch_') and f.endswith('.xml'):
            patches.append(f)
    
    return sorted(patches)

def read_single_annotation(xml_file, split=None):
    if split is None:
        split = DEFAULT_SPLIT
    """Read a single XML annotation file"""
    split_annotations_dir = os.path.join(ANNOTATIONS_DIR, split, 'annotations')
    split_images_dir = os.path.join(IMAGES_DIR, split, 'images')
    
    tree = ET.parse(os.path.join(split_annotations_dir, xml_file))
    root = tree.getroot()
    image_filename = root.find('filename').text
    image_path = os.path.join(split_images_dir, image_filename)
    
    if not os.path.exists(image_path):
        print(f"Warning: Image not found at {image_path}")
        return None
        
    # Create a dictionary to store object data by ID for group reference
    objects_by_id = {}
    all_items = []
    
    # Process individual objects
    for obj in root.findall('object'):
        obj_id = obj.find('id').text
        category = obj.find('name').text
        bbox_elem = obj.find('bndbox')
        if bbox_elem is None:
            continue
            
        bbox = [
            int(bbox_elem.find('xmin').text),
            int(bbox_elem.find('ymin').text),
            int(bbox_elem.find('xmax').text),
            int(bbox_elem.find('ymax').text)
        ]
        
        # Store object data for group reference
        objects_by_id[obj_id] = {
            'category': category,
            'bbox': bbox
        }
        
        original_expressions = []
        expressions_elem = obj.find('expressions')
        if expressions_elem is not None:
            for expr in expressions_elem.findall('expression'):
                if expr.text and expr.text.strip():
                    original_expressions.append(expr.text)
        
        enhanced_expressions = []
        enhanced_elem = obj.find('enhanced_expressions')
        if enhanced_elem is not None:
            enhanced_by_base = {}
            for expr in enhanced_elem.findall('expression'):
                base_id = expr.get('base_id')
                if base_id not in enhanced_by_base:
                    enhanced_by_base[base_id] = []
                enhanced_by_base[base_id].append(expr.text)
                
            for orig_idx, orig_expr in enumerate(original_expressions):
                base_id = str(orig_idx)
                if base_id in enhanced_by_base and len(enhanced_by_base[base_id]) == 2:  # Now expecting 2 language variations
                    enhanced_expressions.append({
                        'original': orig_expr,
                        'variations': enhanced_by_base[base_id]
                    })
        
        # Get unique expressions
        unique_expressions = []
        unique_elem = obj.find('unique_expressions')
        if unique_elem is not None:
            for expr in unique_elem.findall('expression'):
                if expr.text and expr.text.strip():
                    unique_expressions.append(expr.text)
        
        # Get unique description
        unique_description = None
        unique_desc_elem = obj.find('unique_description')
        if unique_desc_elem is not None and unique_desc_elem.text:
            unique_description = unique_desc_elem.text
        
        raw_llm_description = None
        desc_elem = obj.find('raw_llm_description')
        if desc_elem is not None and desc_elem.text:
            raw_llm_description = desc_elem.text
            
        if original_expressions and (enhanced_expressions or unique_expressions):
            all_items.append({
                'obj_id': obj_id,
                'category': category,
                'bbox': bbox,
                'image_path': image_path,
                'image_filename': image_filename,
                'expressions': enhanced_expressions,
                'unique_expressions': unique_expressions,
                'unique_description': unique_description,
                'raw_description': raw_llm_description,
                'type': 'instance'
            })
    
    # Process groups
    groups_elem = root.find('groups')
    if groups_elem is not None:
        for group in groups_elem.findall('group'):
            group_id = group.find('id').text
            category = group.find('category').text if group.find('category') is not None else "Unknown"
            
            instance_ids_elem = group.find('instance_ids')
            if instance_ids_elem is None or not instance_ids_elem.text:
                continue
            
            instance_ids = instance_ids_elem.text.split(',')
            
            member_bboxes = []
            for obj_id in instance_ids:
                if obj_id in objects_by_id:
                    member_bboxes.append(objects_by_id[obj_id]['bbox'])
            
            if not member_bboxes:
                continue
            
            min_x = min(bbox[0] for bbox in member_bboxes)
            min_y = min(bbox[1] for bbox in member_bboxes)
            max_x = max(bbox[2] for bbox in member_bboxes)
            max_y = max(bbox[3] for bbox in member_bboxes)
            
            group_bbox = [min_x, min_y, max_x, max_y]
            
            original_expressions = []
            expressions_elem = group.find('expressions')
            if expressions_elem is not None:
                for expr in expressions_elem.findall('expression'):
                    if expr.text and expr.text.strip():
                        original_expressions.append(expr.text)
            
            enhanced_expressions = []
            enhanced_elem = group.find('enhanced_expressions')
            if enhanced_elem is not None:
                enhanced_by_base = {}
                for expr in enhanced_elem.findall('expression'):
                    base_id = expr.get('base_id')
                    if base_id not in enhanced_by_base:
                        enhanced_by_base[base_id] = []
                    enhanced_by_base[base_id].append(expr.text)
                
                for orig_idx, orig_expr in enumerate(original_expressions):
                    base_id = str(orig_idx)
                    if base_id in enhanced_by_base and len(enhanced_by_base[base_id]) == 2:  # Now expecting 2 language variations
                        enhanced_expressions.append({
                            'original': orig_expr,
                            'variations': enhanced_by_base[base_id]
                        })
            
            # Get unique expressions for group
            unique_expressions = []
            unique_elem = group.find('unique_expressions')
            if unique_elem is not None:
                for expr in unique_elem.findall('expression'):
                    if expr.text and expr.text.strip():
                        unique_expressions.append(expr.text)
            
            # Get unique description for group
            unique_description = None
            unique_desc_elem = group.find('unique_description')
            if unique_desc_elem is not None and unique_desc_elem.text:
                unique_description = unique_desc_elem.text
            
            raw_llm_description = None
            desc_elem = group.find('raw_llm_description')
            if desc_elem is not None and desc_elem.text:
                raw_llm_description = desc_elem.text
            
            if original_expressions and (enhanced_expressions or unique_expressions):
                all_items.append({
                    'group_id': group_id,
                    'category': category,
                    'bbox': group_bbox,
                    'member_ids': instance_ids,
                    'member_bboxes': member_bboxes,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'expressions': enhanced_expressions,
                    'unique_expressions': unique_expressions,
                    'unique_description': unique_description,
                    'raw_description': raw_llm_description,
                    'type': 'group'
                })
    
    return all_items

@app.route('/')
def index():
    split = request.args.get('split', DEFAULT_SPLIT)
    image_num = request.args.get('image_num')
    patch_idx = int(request.args.get('patch_idx', 0))
    item_idx = int(request.args.get('item_idx', 0))
    item_type = request.args.get('type', 'all')
    
    # Get list of image numbers
    image_numbers = get_image_numbers(split)
    
    if not image_numbers:
        return "No images found in the selected split", 404
    
    # If no image number selected, show image selection page
    if not image_num:
        return render_template_string("""
        <html>
        <head>
            <title>Select Image</title>
            <style>
                body { font-family: sans-serif; text-align: center; margin: 40px; }
                .image-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
                    gap: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .image-item {
                    padding: 10px;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    cursor: pointer;
                }
                .image-item:hover {
                    background-color: #f0f0f0;
                }
                .split-selector {
                    margin-bottom: 20px;
                }
                .split-button {
                    padding: 8px 16px;
                    margin: 0 5px;
                    cursor: pointer;
                    background-color: #f0f0f0;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                .split-button.active {
                    background-color: #2196F3;
                    color: white;
                }
            </style>
        </head>
        <body>
            <h1>Select Image</h1>
            
            <div class="split-selector">
                <a href="/?split=train"><button class="split-button {{ 'active' if split == 'train' else '' }}">Train Split</button></a>
                <a href="/?split=val"><button class="split-button {{ 'active' if split == 'val' else '' }}">Validation Split</button></a>
            </div>
            
            <div class="image-grid">
                {% for num in image_numbers %}
                <a href="/?split={{ split }}&image_num={{ num }}" style="text-decoration: none; color: inherit;">
                    <div class="image-item">
                        <h3>Image {{ num }}</h3>
                    </div>
                </a>
                {% endfor %}
            </div>
        </body>
        </html>
        """, split=split, image_numbers=image_numbers)
    
    # Get patches for selected image
    patches = get_patches_for_image(image_num, split)
    if not patches:
        return "No patches found for selected image", 404
    
    # Get items from current patch
    current_patch = patches[patch_idx % len(patches)]
    items = read_single_annotation(current_patch, split)
    if not items:
        return "No items found in selected patch", 404
    
    # Filter items based on type
    if item_type == 'instance':
        items = [item for item in items if item.get('type') == 'instance']
    elif item_type == 'group':
        items = [item for item in items if item.get('type') == 'group']
    
    if not items:
        return "No items found with selected filter", 404
    
    # Get current item
    item_idx = item_idx % len(items)  # Ensure item_idx is within bounds
    item = items[item_idx]
    
    # Calculate total patches and items
    total_patches = len(patches)
    total_items = len(items)
    
    return render_template_string("""
    <html>
    <head>
        <title>LLM-Enhanced Expression Viewer</title>
        <style>
            body { font-family: sans-serif; text-align: center; margin: 40px; }
            table { margin: 20px auto; border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ccc; padding: 10px; vertical-align: top; }
            th { background-color: #f0f0f0; font-weight: bold; }
            .desc-box { background: #f9f9f9; padding: 15px; margin: 20px auto; width: 100%; border: 1px solid #ddd; }
            img { max-width: 100%; border: 1px solid #ccc; }
            .nav { margin-top: 20px; }
            .filter-buttons { margin-bottom: 20px; }
            .filter-button { 
                padding: 8px 16px; 
                margin: 0 5px; 
                cursor: pointer;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            .filter-button.active {
                background-color: #4CAF50;
                color: white;
            }
            .item-type {
                display: inline-block;
                padding: 3px 8px;
                margin-left: 10px;
                border-radius: 4px;
                font-size: 0.8em;
            }
            .item-type.instance {
                background-color: #5bc0de;
                color: white;
            }
            .item-type.group {
                background-color: #f0ad4e;
                color: white;
            }
            .stats {
                margin: 15px auto;
                font-size: 0.9em;
                color: #666;
            }
            .split-selector {
                margin-bottom: 20px;
            }
            .split-button {
                padding: 8px 16px;
                margin: 0 5px;
                cursor: pointer;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            .split-button.active {
                background-color: #2196F3;
                color: white;
            }
            .back-link {
                margin-bottom: 20px;
            }
            .item-nav {
                margin: 10px 0;
            }
            .content-container {
                display: flex;
                gap: 20px;
                max-width: 1800px;
                margin: 0 auto;
                align-items: flex-start;
            }
            .image-container {
                flex: 1;
                min-width: 0;
            }
            .expressions-container {
                flex: 1;
                min-width: 0;
                text-align: left;
            }
            .unique-expressions {
                margin-top: 20px;
            }
            .unique-expressions table {
                width: 100%;
            }
            details {
                margin: 10px 0;
                border: 1px solid #ddd;
                border-radius: 4px;
                background: #f9f9f9;
            }
            details summary {
                padding: 10px 15px;
                cursor: pointer;
                font-weight: bold;
                background: #f0f0f0;
                border-radius: 4px 4px 0 0;
                user-select: none;
            }
            details summary:hover {
                background: #e0e0e0;
            }
            details[open] summary {
                border-bottom: 1px solid #ddd;
            }
            details > div {
                padding: 15px;
            }
            .expressions-table {
                margin-top: 0;
            }
        </style>
    </head>
    <body>
        <h1>LLM-Enhanced Unique Expressions Viewer</h1>
        
        <div class="back-link">
            <a href="/?split={{ split }}">← Back to Image Selection</a>
        </div>
        
        <div class="split-selector">
            <a href="/?split=train&image_num={{ image_num }}&patch_idx=0&item_idx=0&type={{ item_type }}"><button class="split-button {{ 'active' if split == 'train' else '' }}">Train Split</button></a>
            <a href="/?split=val&image_num={{ image_num }}&patch_idx=0&item_idx=0&type={{ item_type }}"><button class="split-button {{ 'active' if split == 'val' else '' }}">Validation Split</button></a>
        </div>
        
        <div class="filter-buttons">
            <a href="/?split={{ split }}&image_num={{ image_num }}&patch_idx={{ patch_idx }}&item_idx=0&type=all"><button class="filter-button {{ 'active' if item_type == 'all' else '' }}">All</button></a>
            <a href="/?split={{ split }}&image_num={{ image_num }}&patch_idx={{ patch_idx }}&item_idx=0&type=instance"><button class="filter-button {{ 'active' if item_type == 'instance' else '' }}">Instances Only</button></a>
            <a href="/?split={{ split }}&image_num={{ image_num }}&patch_idx={{ patch_idx }}&item_idx=0&type=group"><button class="filter-button {{ 'active' if item_type == 'group' else '' }}">Groups Only</button></a>
        </div>
        
        <h2>
            {% if item['type'] == 'instance' %}
                {{ item['category'] }} (ID: {{ item['obj_id'] }})
                <span class="item-type instance">Instance</span>
            {% else %}
                Group of {{ item['category'] }} (ID: {{ item['group_id'] }})
                <span class="item-type group">Group</span>
            {% endif %}
        </h2>

        <div class="content-container">
            <div class="image-container">
                <img src="/image/{{ patch_idx }}?type={{ item_type }}&split={{ split }}&image_num={{ image_num }}&item_idx={{ item_idx }}"><br>

                <div class="stats">
                    <p>Split: {{ split|upper }}</p>
                    <p>Image Number: {{ image_num }}</p>
                    <p>Patch: {{ current_patch }}</p>
                    <p>Item {{ item_idx + 1 }} of {{ total_items }} in current patch</p>
                    <p>Number of Expressions: {{ item['expressions']|length }}</p>
                </div>
            </div>

            <div class="expressions-container">
                {% if item['raw_description'] %}
                <details>
                    <summary>LLM Description</summary>
                    <div class="desc-box">
                        {{ item['raw_description'] }}
                    </div>
                </details>
                {% endif %}

                <details open>
                    <summary>Language Variations ({{ item['expressions']|length }})</summary>
                    <div>
                        <table class="expressions-table">
                            <tr>
                                <th>Original</th>
                                <th>Language Variation 1</th>
                                <th>Language Variation 2</th>
                            </tr>
                            {% for expr in item['expressions'] %}
                            <tr>
                                <td>{{ expr['original'] }}</td>
                                <td>{{ expr['variations'][0] }}</td>
                                <td>{{ expr['variations'][1] }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                </details>

                {% if item['unique_expressions'] %}
                <details>
                    <summary>Unique Expressions ({{ item['unique_expressions']|length }})</summary>
                    <div>
                        {% if item['unique_description'] %}
                        <div class="desc-box">
                            <strong>Spatial Context Analysis:</strong><br>
                            {{ item['unique_description'] }}
                        </div>
                        {% endif %}
                        <table class="expressions-table">
                            <tr>
                                <th>Expression</th>
                            </tr>
                            {% for expr in item['unique_expressions'] %}
                            <tr>
                                <td>{{ expr }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                </details>
                {% endif %}
            </div>
        </div>

        <div class="nav">
            <div class="item-nav">
                <a href="/?split={{ split }}&image_num={{ image_num }}&patch_idx={{ patch_idx }}&item_idx={{ (item_idx - 1) % total_items }}&type={{ item_type }}">Previous Item</a> |
                <a href="/?split={{ split }}&image_num={{ image_num }}&patch_idx={{ patch_idx }}&item_idx={{ (item_idx + 1) % total_items }}&type={{ item_type }}">Next Item</a>
                <p>Item {{ item_idx + 1 }} of {{ total_items }}</p>
            </div>
            
            <div class="patch-nav">
                <a href="/?split={{ split }}&image_num={{ image_num }}&patch_idx={{ (patch_idx - 1) % total_patches }}&item_idx=0&type={{ item_type }}">Previous Patch</a> |
                <a href="/?split={{ split }}&image_num={{ image_num }}&patch_idx={{ (patch_idx + 1) % total_patches }}&item_idx=0&type={{ item_type }}">Next Patch</a>
                <p>Patch {{ patch_idx + 1 }} of {{ total_patches }}</p>
                <form method="get" action="/" style="margin-top:10px;">
                    <label for="patch_idx">Go to patch:</label>
                    <input type="number" id="patch_idx" name="patch_idx" min="0" max="{{ total_patches - 1 }}" value="{{ patch_idx }}">
                    <input type="hidden" name="type" value="{{ item_type }}">
                    <input type="hidden" name="split" value="{{ split }}">
                    <input type="hidden" name="image_num" value="{{ image_num }}">
                    <input type="hidden" name="item_idx" value="0">
                    <input type="submit" value="Go">
                </form>
            </div>
        </div>
    </body>
    </html>
    """, item=item, patch_idx=patch_idx, item_idx=item_idx, total_patches=total_patches, 
         total_items=total_items, item_type=item_type, split=split, image_num=image_num, 
         current_patch=current_patch)

@app.route('/image/<int:patch_idx>')
def image(patch_idx):
    split = request.args.get('split', DEFAULT_SPLIT)
    item_type = request.args.get('type', 'all')
    image_num = request.args.get('image_num')
    item_idx = int(request.args.get('item_idx', 0))
    
    if not image_num:
        return "Image number not specified", 404
    
    # Get patches for selected image
    patches = get_patches_for_image(image_num, split)
    if not patches:
        return "No patches found for selected image", 404
    
    # Get items from current patch
    current_patch = patches[patch_idx % len(patches)]
    items = read_single_annotation(current_patch, split)
    if not items:
        return "No items found in selected patch", 404
    
    # Filter items based on type
    if item_type == 'instance':
        items = [item for item in items if item.get('type') == 'instance']
    elif item_type == 'group':
        items = [item for item in items if item.get('type') == 'group']
    
    if not items:
        return "No items found with selected filter", 404
    
    # Get current item
    item_idx = item_idx % len(items)  # Ensure item_idx is within bounds
    item = items[item_idx]
    
    img = cv2.imread(item['image_path'])
    if img is None:
        return "Image not found", 404

    # Draw bounding box(es)
    if item['type'] == 'instance':
        x1, y1, x2, y2 = item['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
        for member_bbox in item.get('member_bboxes', []):
            x1, y1, x2, y2 = member_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        x1, y1, x2, y2 = item['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    _, buffer = cv2.imencode('.png', img)
    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype='image/png'
    )

if __name__ == '__main__':
    # Check if the specified split has any images
    image_numbers = get_image_numbers(DEFAULT_SPLIT)
    if not image_numbers:
        print(f"Error: No images found for split '{DEFAULT_SPLIT}'")
        print("Available splits:")
        for split in ['train', 'val']:
            if os.path.exists(os.path.join(ANNOTATIONS_DIR, split, 'annotations')):
                print(f"- {split}")
        exit(1)
        
    print(f"Starting server for split '{DEFAULT_SPLIT}' on port {args.port}")
    print(f"Found {len(image_numbers)} images")
    app.run(debug=True, port=args.port) 