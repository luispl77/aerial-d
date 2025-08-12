import os
import io
import json
from flask import Flask, render_template_string, send_file, request, redirect
import cv2
from pathlib import Path
from collections import defaultdict
import argparse
import random
import numpy as np
import pycocotools.mask as mask_util
import base64

app = Flask(__name__)

# Default paths
DEFAULT_DATASET_DIR = "dataset"
REFCOCO_DIR = None
IMAGES_DIR = None

# Cache for data
REFCOCO_DATA = None
IMAGE_CACHE = {}

def load_refcoco_data(split='train'):
    """Load RefCOCO JSON data for a specific split"""
    global REFCOCO_DATA
    
    if REFCOCO_DATA is None or REFCOCO_DATA.get('current_split') != split:
        refcoco_file = os.path.join(REFCOCO_DIR, f'refcoco_{split}.json')
        if not os.path.exists(refcoco_file):
            print(f"RefCOCO file not found: {refcoco_file}")
            return None
            
        with open(refcoco_file, 'r') as f:
            data = json.load(f)
        
        # Create lookup dictionaries for faster access
        data['images_by_id'] = {img['id']: img for img in data['images']}
        data['annotations_by_id'] = {ann['id']: ann for ann in data['annotations']}
        data['annotations_by_image'] = defaultdict(list)
        data['refs_by_image'] = defaultdict(list)
        data['categories_by_id'] = {cat['id']: cat for cat in data['categories']}
        
        for ann in data['annotations']:
            data['annotations_by_image'][ann['image_id']].append(ann)
        
        for ref in data['refs']:
            data['refs_by_image'][ref['image_id']].append(ref)
        
        data['current_split'] = split
        REFCOCO_DATA = data
    
    return REFCOCO_DATA

def get_image_numbers(split='train'):
    """Get list of unique image identifiers for a split"""
    data = load_refcoco_data(split)
    if data is None:
        return []
    
    # Extract unique image identifiers, preserving prefixes
    image_identifiers = set()
    for image in data['images']:
        filename = image['file_name']
        # Extract the identifier part (e.g., P0000, L1234, D100034) from filename
        if '_patch_' in filename:
            image_id = filename.split('_patch_')[0]
            image_identifiers.add(image_id)
    
    return sorted(list(image_identifiers))

def get_patches_for_image(image_id, split='train'):
    """Get all patches for a specific image identifier"""
    data = load_refcoco_data(split)
    if data is None:
        return []
    
    patches = []
    for image in data['images']:
        filename = image['file_name']
        if filename.startswith(f'{image_id}_patch_'):
            patches.append({
                'filename': filename,
                'image_id': image['id'],
                'width': image['width'],
                'height': image['height']
            })
    
    return sorted(patches, key=lambda x: x['filename'])

def decode_rle_mask(rle_data, height, width):
    """Decode RLE mask from RefCOCO format"""
    try:
        if isinstance(rle_data, dict) and 'counts' in rle_data:
            # COCO RLE format
            if isinstance(rle_data['counts'], str):
                # String format, encode to bytes
                rle_data['counts'] = rle_data['counts'].encode('utf-8')
            return mask_util.decode(rle_data)
        else:
            # Fallback: assume it's base64 encoded
            rle_bytes = base64.b64decode(rle_data)
            rle = {'size': [height, width], 'counts': rle_bytes}
            return mask_util.decode(rle)
    except Exception as e:
        print(f"Error decoding RLE mask: {e}")
        return None

def get_annotation_data(patch_info, split='train'):
    """Get annotation data for a specific patch"""
    data = load_refcoco_data(split)
    if data is None:
        return []
    
    image_id = patch_info['image_id']
    
    # Get annotations for this image
    annotations = data['annotations_by_image'].get(image_id, [])
    refs = data['refs_by_image'].get(image_id, [])
    
    # Create a mapping from annotation ID to referring expressions
    refs_by_ann_id = defaultdict(list)
    for ref in refs:
        refs_by_ann_id[ref['ann_id']].append(ref)
    
    all_items = []
    
    for ann in annotations:
        category_name = data['categories_by_id'].get(ann['category_id'], {}).get('name', 'unknown')
        
        # Get all referring expressions for this annotation
        all_expressions = []
        for ref in refs_by_ann_id[ann['id']]:
            for sentence in ref['sentences']:
                expr_type = sentence.get('type', 'original')
                all_expressions.append({
                    'text': sentence['sent'],
                    'type': expr_type,
                    'ref_id': ref['ref_id'],
                    'sent_id': sentence['sent_id']
                })
        
        # Group expressions by type
        expressions_by_type = defaultdict(list)
        for expr in all_expressions:
            expressions_by_type[expr['type']].append(expr['text'])
        
        # Determine item type
        item_type = 'group' if 'member_ids' in ann else 'instance'
        
        item = {
            'ann_id': ann['id'],
            'category': category_name,
            'bbox': ann['bbox'],  # [x, y, width, height] format
            'area': ann['area'],
            'image_path': os.path.join(IMAGES_DIR, split, 'images', patch_info['filename']),
            'image_filename': patch_info['filename'],
            'expressions': expressions_by_type.get('original', []),
            'enhanced_expressions': expressions_by_type.get('enhanced', []),
            'unique_expressions': expressions_by_type.get('unique', []),
            'all_expressions': all_expressions,
            'type': item_type,
            'segmentation': ann['segmentation'],
            'width': patch_info['width'],
            'height': patch_info['height']
        }
        
        # Add group-specific fields
        if item_type == 'group':
            item['group_id'] = ann.get('original_group_id', f"group_{ann['id']}")
            item['member_ids'] = ann.get('member_ids', [])
        else:
            item['obj_id'] = ann.get('original_obj_id', f"obj_{ann['id']}")
        
        all_items.append(item)
    
    return all_items

@app.route('/')
def index():
    split = request.args.get('split', 'train')
    image_id = request.args.get('image_id')
    patch_idx = int(request.args.get('patch_idx', 0))
    item_idx = int(request.args.get('item_idx', 0))
    item_type = request.args.get('type', 'all')
    
    # Get list of image identifiers
    image_identifiers = get_image_numbers(split)
    
    if not image_identifiers:
        return "No images found in the selected split", 404
    
    # If no image identifier selected, show image selection page
    if not image_id:
        return render_template_string("""
        <html>
        <head>
            <title>Select Image - RefCOCO JSON Viewer</title>
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
                .dataset-type {
                    display: inline-block;
                    padding: 3px 8px;
                    margin-left: 10px;
                    border-radius: 4px;
                    font-size: 0.8em;
                }
                .dataset-type.isaid {
                    background-color: #5bc0de;
                    color: white;
                }
                .dataset-type.loveda {
                    background-color: #5cb85c;
                    color: white;
                }
                .dataset-type.deepglobe {
                    background-color: #f0ad4e;
                    color: white;
                }
            </style>
        </head>
        <body>
            <h1>Select Image - RefCOCO JSON Viewer</h1>
            
            <div class="split-selector">
                <a href="/?split=train"><button class="split-button {{ 'active' if split == 'train' else '' }}">Train Split</button></a>
                <a href="/?split=val"><button class="split-button {{ 'active' if split == 'val' else '' }}">Validation Split</button></a>
            </div>
            
            <div class="image-grid">
                {% for identifier in image_identifiers %}
                <a href="/?split={{ split }}&image_id={{ identifier }}" style="text-decoration: none; color: inherit;">
                    <div class="image-item">
                        <h3>{{ identifier }}
                            {% if identifier.startswith('P') %}
                                <span class="dataset-type isaid">iSAID</span>
                            {% elif identifier.startswith('L') %}
                                <span class="dataset-type loveda">LoveDA</span>
                            {% elif identifier.startswith('D') %}
                                <span class="dataset-type deepglobe">DeepGlobe</span>
                            {% endif %}
                        </h3>
                    </div>
                </a>
                {% endfor %}
            </div>
        </body>
        </html>
        """, split=split, image_identifiers=image_identifiers)
    
    # Get patches for selected image
    patches = get_patches_for_image(image_id, split)
    if not patches:
        return "No patches found for selected image", 404
    
    # Get items from current patch
    current_patch = patches[patch_idx % len(patches)]
    items = get_annotation_data(current_patch, split)
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
    item_idx = item_idx % len(items)
    item = items[item_idx]
    
    # Calculate totals
    total_patches = len(patches)
    total_items = len(items)
    
    return render_template_string("""
    <html>
    <head>
        <title>RefCOCO JSON Expression Viewer</title>
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
            .expr-type {
                display: inline-block;
                padding: 2px 6px;
                margin-left: 8px;
                border-radius: 3px;
                font-size: 0.7em;
                font-weight: bold;
            }
            .expr-type.original {
                background-color: #337ab7;
                color: white;
            }
            .expr-type.enhanced {
                background-color: #5cb85c;
                color: white;
            }
            .expr-type.unique {
                background-color: #f0ad4e;
                color: white;
            }
        </style>
    </head>
    <body>
        <h1>RefCOCO JSON Expression Viewer</h1>
        
        <div class="back-link">
            <a href="/?split={{ split }}">← Back to Image Selection</a>
        </div>
        
        <div class="split-selector">
            <a href="/?split=train&image_id={{ image_id }}&patch_idx=0&item_idx=0&type={{ item_type }}"><button class="split-button {{ 'active' if split == 'train' else '' }}">Train Split</button></a>
            <a href="/?split=val&image_id={{ image_id }}&patch_idx=0&item_idx=0&type={{ item_type }}"><button class="split-button {{ 'active' if split == 'val' else '' }}">Validation Split</button></a>
        </div>
        
        <div class="filter-buttons">
            <a href="/?split={{ split }}&image_id={{ image_id }}&patch_idx={{ patch_idx }}&item_idx=0&type=all"><button class="filter-button {{ 'active' if item_type == 'all' else '' }}">All</button></a>
            <a href="/?split={{ split }}&image_id={{ image_id }}&patch_idx={{ patch_idx }}&item_idx=0&type=instance"><button class="filter-button {{ 'active' if item_type == 'instance' else '' }}">Instances Only</button></a>
            <a href="/?split={{ split }}&image_id={{ image_id }}&patch_idx={{ patch_idx }}&item_idx=0&type=group"><button class="filter-button {{ 'active' if item_type == 'group' else '' }}">Groups Only</button></a>
        </div>
        
        <h2>
            {% if item['type'] == 'instance' %}
                {{ item['category'] }} (ID: {{ item.get('obj_id', item['ann_id']) }})
                <span class="item-type instance">Instance</span>
            {% else %}
                Group of {{ item['category'] }} (ID: {{ item.get('group_id', item['ann_id']) }})
                <span class="item-type group">Group</span>
            {% endif %}
        </h2>

        <div class="content-container">
            <div class="image-container">
                <img src="/image/{{ patch_idx }}?type={{ item_type }}&split={{ split }}&image_id={{ image_id }}&item_idx={{ item_idx }}"><br>

                <div class="stats">
                    <p>Split: {{ split|upper }}</p>
                    <p>Image ID: {{ image_id }}</p>
                    <p>Patch: {{ current_patch['filename'] }}</p>
                    <p>Item {{ item_idx + 1 }} of {{ total_items }} in current patch</p>
                    <p>Total Expressions: {{ item['all_expressions']|length }}</p>
                    <p>Area: {{ "%.1f"|format(item['area']) }} pixels</p>
                </div>
            </div>

            <div class="expressions-container">
                <details open>
                    <summary>All Expressions ({{ item['all_expressions']|length }})</summary>
                    <div>
                        <table class="expressions-table">
                            <tr>
                                <th>Expression</th>
                                <th>Type</th>
                                <th>IDs</th>
                            </tr>
                            {% for expr in item['all_expressions'] %}
                            <tr>
                                <td>{{ expr['text'] }}</td>
                                <td><span class="expr-type {{ expr['type'] }}">{{ expr['type']|upper }}</span></td>
                                <td>Ref: {{ expr['ref_id'] }}, Sent: {{ expr['sent_id'] }}</td>
                            </tr>
                            {% endfor %}
                        </table>
                    </div>
                </details>

                {% if item['expressions'] %}
                <details>
                    <summary>Original Expressions ({{ item['expressions']|length }})</summary>
                    <div>
                        <table class="expressions-table">
                            <tr><th>Expression</th></tr>
                            {% for expr in item['expressions'] %}
                            <tr><td>{{ expr }}</td></tr>
                            {% endfor %}
                        </table>
                    </div>
                </details>
                {% endif %}

                {% if item['enhanced_expressions'] %}
                <details>
                    <summary>Enhanced Expressions ({{ item['enhanced_expressions']|length }})</summary>
                    <div>
                        <table class="expressions-table">
                            <tr><th>Expression</th></tr>
                            {% for expr in item['enhanced_expressions'] %}
                            <tr><td>{{ expr }}</td></tr>
                            {% endfor %}
                        </table>
                    </div>
                </details>
                {% endif %}

                {% if item['unique_expressions'] %}
                <details>
                    <summary>Unique Expressions ({{ item['unique_expressions']|length }})</summary>
                    <div>
                        <table class="expressions-table">
                            <tr><th>Expression</th></tr>
                            {% for expr in item['unique_expressions'] %}
                            <tr><td>{{ expr }}</td></tr>
                            {% endfor %}
                        </table>
                    </div>
                </details>
                {% endif %}
            </div>
        </div>

        <div class="nav">
            <div class="item-nav">
                <a href="/?split={{ split }}&image_id={{ image_id }}&patch_idx={{ patch_idx }}&item_idx={{ (item_idx - 1) % total_items }}&type={{ item_type }}">Previous Item</a> |
                <a href="/?split={{ split }}&image_id={{ image_id }}&patch_idx={{ patch_idx }}&item_idx={{ (item_idx + 1) % total_items }}&type={{ item_type }}">Next Item</a>
                <p>Item {{ item_idx + 1 }} of {{ total_items }}</p>
            </div>
            
            <div class="patch-nav">
                <a href="/?split={{ split }}&image_id={{ image_id }}&patch_idx={{ (patch_idx - 1) % total_patches }}&item_idx=0&type={{ item_type }}">Previous Patch</a> |
                <a href="/?split={{ split }}&image_id={{ image_id }}&patch_idx={{ (patch_idx + 1) % total_patches }}&item_idx=0&type={{ item_type }}">Next Patch</a>
                <p>Patch {{ patch_idx + 1 }} of {{ total_patches }}</p>
                <form method="get" action="/" style="margin-top:10px;">
                    <label for="patch_idx">Go to patch:</label>
                    <input type="number" id="patch_idx" name="patch_idx" min="0" max="{{ total_patches - 1 }}" value="{{ patch_idx }}">
                    <input type="hidden" name="type" value="{{ item_type }}">
                    <input type="hidden" name="split" value="{{ split }}">
                    <input type="hidden" name="image_id" value="{{ image_id }}">
                    <input type="hidden" name="item_idx" value="0">
                    <input type="submit" value="Go">
                </form>
            </div>
        </div>
    </body>
    </html>
    """, item=item, patch_idx=patch_idx, item_idx=item_idx, total_patches=total_patches, 
         total_items=total_items, item_type=item_type, split=split, image_id=image_id, 
         current_patch=current_patch)

@app.route('/image/<int:patch_idx>')
def image(patch_idx):
    split = request.args.get('split', 'train')
    item_type = request.args.get('type', 'all')
    image_id = request.args.get('image_id')
    item_idx = int(request.args.get('item_idx', 0))
    
    if not image_id:
        return "Image ID not specified", 404
    
    # Get patches for selected image
    patches = get_patches_for_image(image_id, split)
    if not patches:
        return "No patches found for selected image", 404
    
    # Get items from current patch
    current_patch = patches[patch_idx % len(patches)]
    items = get_annotation_data(current_patch, split)
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
    item_idx = item_idx % len(items)
    item = items[item_idx]
    
    img = cv2.imread(item['image_path'])
    if img is None:
        return "Image not found", 404

    # Draw segmentation mask if available
    if item['segmentation']:
        try:
            mask = decode_rle_mask(item['segmentation'], item['height'], item['width'])
            if mask is not None:
                # Create colored overlay
                mask_colored = np.zeros_like(img, dtype=np.uint8)
                if item['type'] == 'group':
                    mask_colored[mask > 0] = [0, 255, 255]  # Yellow for groups
                else:
                    mask_colored[mask > 0] = [0, 0, 255]    # Red for instances
                
                # Blend with original image
                alpha = 0.3
                img = cv2.addWeighted(img, 1-alpha, mask_colored, alpha, 0)
        except Exception as e:
            print(f"Error drawing segmentation mask: {e}")
    
    # Draw bounding box (convert from COCO format [x, y, width, height] to [x1, y1, x2, y2])
    x, y, w, h = item['bbox']
    x1, y1, x2, y2 = x, y, x + w, y + h
    
    if item['type'] == 'group':
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Yellow for groups
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)    # Red for instances

    _, buffer = cv2.imencode('.png', img)
    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype='image/png'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RefCOCO JSON Rule Viewer')
    parser.add_argument('--split', type=str, default='train', choices=['train', 'val'],
                        help='Dataset split to use (default: train)')
    parser.add_argument('--port', type=int, default=5003,
                        help='Port to run the server on (default: 5003)')
    parser.add_argument('--dataset', type=str, default='dataset',
                        help='Dataset folder name (default: dataset)')
    args = parser.parse_args()
    
    # Set global paths
    dataset_dir = os.path.abspath(args.dataset)
    REFCOCO_DIR = os.path.join(dataset_dir, 'refcoco_format')
    IMAGES_DIR = os.path.join(dataset_dir, 'patches')
    
    print(f"Starting RefCOCO JSON viewer for split '{args.split}' on port {args.port}")
    print(f"Dataset directory: {dataset_dir}")
    print(f"RefCOCO directory: {REFCOCO_DIR}")
    print(f"Images directory: {IMAGES_DIR}")
    
    # Check if RefCOCO files exist
    refcoco_file = os.path.join(REFCOCO_DIR, f'refcoco_{args.split}.json')
    if not os.path.exists(refcoco_file):
        print(f"Error: RefCOCO file not found: {refcoco_file}")
        print("Please run step 9 (9_convert_to_refcoco.py) first to generate the JSON files.")
        exit(1)
    
    # Test loading data
    data = load_refcoco_data(args.split)
    if data is None:
        print("Error loading RefCOCO data")
        exit(1)
    
    print(f"Loaded RefCOCO data:")
    print(f"  Images: {len(data['images'])}")
    print(f"  Annotations: {len(data['annotations'])}")
    print(f"  Referring expressions: {len(data['refs'])}")
    print(f"  Total sentences: {sum(len(ref['sentences']) for ref in data['refs'])}")
    
    app.run(debug=True, port=args.port)