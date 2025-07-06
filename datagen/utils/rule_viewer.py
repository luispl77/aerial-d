import os
import io
import xml.etree.ElementTree as ET
from flask import Flask, render_template_string, send_file, request, redirect
import cv2
from pathlib import Path
from collections import defaultdict
import argparse
import random  # Add random import
import numpy as np
import pycocotools.mask as mask_util  # Add for RLE decoding

app = Flask(__name__)

# Default paths
DEFAULT_DATASET_DIR = "dataset"
ANNOTATIONS_DIR = None
IMAGES_DIR = None

# Cache for image listings
IMAGE_LISTINGS = {}

# Dataset types
DATASET_TYPES = ['isaid', 'loveda', 'deepglobe']

def get_image_numbers(split='train'):
    """Get list of unique image identifiers for a split, keeping full prefix+number"""
    if split not in IMAGE_LISTINGS:
        split_annotations_dir = os.path.join(ANNOTATIONS_DIR, split, 'annotations')
        # Extract unique image identifiers from filenames, keeping P, L, and D prefixes
        image_identifiers = set()
        for f in os.listdir(split_annotations_dir):
            if f.endswith('.xml'):
                try:
                    # Support P-prefixed (iSAID), L-prefixed (LoveDA), and D-prefixed (DeepGlobe) files
                    if f.startswith('P') or f.startswith('L') or f.startswith('D'):
                        # Keep the full prefix+number (e.g., P0000, L1234, D100034)
                        image_id = f.split('_patch_')[0]  # Get everything before _patch_
                        image_identifiers.add(image_id)
                except:
                    continue
        IMAGE_LISTINGS[split] = sorted(list(image_identifiers))
    return IMAGE_LISTINGS[split]

def get_patches_for_image(image_id, split='train'):
    """Get all patches for a specific image identifier (e.g., P0000, L1234, or D100034)"""
    split_annotations_dir = os.path.join(ANNOTATIONS_DIR, split, 'annotations')
    patches = []
    
    # Find all XML files for this image identifier
    for f in os.listdir(split_annotations_dir):
        if f.startswith(f'{image_id}_patch_') and f.endswith('.xml'):
            patches.append(f)
    
    return sorted(patches)

def read_single_annotation(xml_file, split='train'):
    """Read a single XML annotation file"""
    split_annotations_dir = os.path.join(ANNOTATIONS_DIR, split, 'annotations')
    split_images_dir = os.path.join(IMAGES_DIR, split, 'images')
    
    tree = ET.parse(os.path.join(split_annotations_dir, xml_file))
    root = tree.getroot()
    image_filename = root.find('filename').text
    image_path = os.path.join(split_images_dir, image_filename)
    
    if not os.path.exists(image_path):
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
        
        # Parse expressions by type
        expressions = []
        enhanced_expressions = []
        unique_expressions = []
        expressions_elem = obj.find('expressions')
        if expressions_elem is not None:
            for expr in expressions_elem.findall('expression'):
                if expr.text and expr.text.strip():
                    expr_type = expr.get('type', 'original')  # Default to 'original' if no type attribute
                    if expr_type == 'enhanced':
                        enhanced_expressions.append(expr.text)
                    elif expr_type == 'unique':
                        unique_expressions.append(expr.text)
                    else:
                        expressions.append(expr.text)
        
        if expressions or enhanced_expressions or unique_expressions:
            all_items.append({
                'obj_id': obj_id,
                'category': category,
                'bbox': bbox,
                'image_path': image_path,
                'image_filename': image_filename,
                'expressions': expressions,
                'enhanced_expressions': enhanced_expressions,
                'unique_expressions': unique_expressions,
                'type': 'instance'
            })
    
    # Process groups
    groups_elem = root.find('groups')
    if groups_elem is not None:
        for group in groups_elem.findall('group'):
            group_id = group.find('id').text
            category = group.find('category').text if group.find('category') is not None else "Unknown"
            
            # Try to get instance IDs (for backward compatibility)
            instance_ids = []
            member_bboxes = []
            instance_ids_elem = group.find('instance_ids')
            if instance_ids_elem is not None and instance_ids_elem.text:
                instance_ids = instance_ids_elem.text.split(',')
                for obj_id in instance_ids:
                    if obj_id in objects_by_id:
                        member_bboxes.append(objects_by_id[obj_id]['bbox'])
            
            # Calculate group bounding box - try from members first, then from segmentation
            group_bbox = None
            if member_bboxes:
                min_x = min(bbox[0] for bbox in member_bboxes)
                min_y = min(bbox[1] for bbox in member_bboxes)
                max_x = max(bbox[2] for bbox in member_bboxes)
                max_y = max(bbox[3] for bbox in member_bboxes)
                group_bbox = [min_x, min_y, max_x, max_y]
            else:
                # Calculate bbox from segmentation mask if available
                group_seg_elem = group.find('segmentation')
                if group_seg_elem is not None and group_seg_elem.text:
                    try:
                        group_segmentation = eval(group_seg_elem.text)
                        if isinstance(group_segmentation, dict) and 'size' in group_segmentation:
                            rle = {'size': group_segmentation['size'], 'counts': group_segmentation['counts'].encode('utf-8')}
                            mask = mask_util.decode(rle)
                            
                            # Find bounding box from mask
                            rows, cols = np.where(mask > 0)
                            if len(rows) > 0:
                                min_y, max_y = rows.min(), rows.max()
                                min_x, max_x = cols.min(), cols.max()
                                group_bbox = [min_x, min_y, max_x + 1, max_y + 1]
                    except Exception as e:
                        print(f"Error calculating bbox from group segmentation for group {group_id}: {e}")
            
            # Skip groups without valid bbox
            if group_bbox is None:
                continue
            
            # Parse expressions by type for groups
            expressions = []
            enhanced_expressions = []
            unique_expressions = []
            expressions_elem = group.find('expressions')
            if expressions_elem is not None:
                for expr in expressions_elem.findall('expression'):
                    if expr.text and expr.text.strip():
                        expr_type = expr.get('type', 'original')  # Default to 'original' if no type attribute
                        if expr_type == 'enhanced':
                            enhanced_expressions.append(expr.text)
                        elif expr_type == 'unique':
                            unique_expressions.append(expr.text)
                        else:
                            expressions.append(expr.text)
            
            # Get group segmentation RLE if available
            group_segmentation = None
            group_seg_elem = group.find('segmentation')
            if group_seg_elem is not None and group_seg_elem.text:
                try:
                    group_segmentation = eval(group_seg_elem.text)
                except Exception as e:
                    print(f"Error parsing group segmentation for group {group_id}: {e}")
            
            if expressions or enhanced_expressions or unique_expressions:
                all_items.append({
                    'group_id': group_id,
                    'category': category,
                    'bbox': group_bbox,
                    'member_ids': instance_ids,
                    'member_bboxes': member_bboxes,
                    'group_segmentation': group_segmentation,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'expressions': expressions,
                    'enhanced_expressions': enhanced_expressions,
                    'unique_expressions': unique_expressions,
                    'type': 'group'
                })
    
    return all_items

@app.route('/random')
def random_view():
    split = request.args.get('split', 'train')
    item_type = request.args.get('type', 'all')
    
    # Get list of image identifiers
    image_identifiers = get_image_numbers(split)
    if not image_identifiers:
        return "No images found in the selected split", 404
    
    # Randomly select an image
    image_id = random.choice(image_identifiers)
    
    # Get patches for selected image
    patches = get_patches_for_image(image_id, split)
    if not patches:
        return "No patches found for selected image", 404
    
    # Randomly select a patch
    patch_idx = random.randrange(len(patches))
    current_patch = patches[patch_idx]
    
    # Get items from current patch
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
    
    # Randomly select an item
    item_idx = random.randrange(len(items))
    
    # Redirect to the main view with the random selections
    return redirect(f'/?split={split}&image_id={image_id}&patch_idx={patch_idx}&item_idx={item_idx}&type={item_type}')

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
    
    # If no image number selected, show image selection page
    if not image_id:
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
                .selector {
                    margin-bottom: 20px;
                }
                .selector-button {
                    padding: 8px 16px;
                    margin: 0 5px;
                    cursor: pointer;
                    background-color: #f0f0f0;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                .selector-button.active {
                    background-color: #2196F3;
                    color: white;
                }
                .dataset-selector {
                    margin-bottom: 20px;
                }
                .dataset-button {
                    padding: 8px 16px;
                    margin: 0 5px;
                    cursor: pointer;
                    background-color: #f0f0f0;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                }
                .dataset-button.active {
                    background-color: #9C27B0;
                    color: white;
                }
                .random-button {
                    padding: 12px 24px;
                    margin: 20px 0;
                    cursor: pointer;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    font-size: 1.1em;
                }
                .random-button:hover {
                    background-color: #45a049;
                }
            </style>
        </head>
        <body>
            <h1>Select Image</h1>
            
            <div class="selector">
                <strong>Split:</strong>
                <a href="/?split=train"><button class="selector-button {{ 'active' if split == 'train' else '' }}">Train Split</button></a>
                <a href="/?split=val"><button class="selector-button {{ 'active' if split == 'val' else '' }}">Validation Split</button></a>
            </div>
            
            <a href="/random?split={{ split }}&type=all"><button class="random-button">View Random Object</button></a>
            
            <div class="image-grid">
                {% for id in image_numbers %}
                <a href="/?split={{ split }}&image_id={{ id }}" style="text-decoration: none; color: inherit;">
                    <div class="image-item">
                        <h3>Image {{ id }}</h3>
                    </div>
                </a>
                {% endfor %}
            </div>
        </body>
        </html>
        """, split=split, image_numbers=image_identifiers)
    
    # Get patches for selected image
    patches = get_patches_for_image(image_id, split)
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
        <title>Rule-Based Expression Viewer</title>
        <style>
            body { font-family: sans-serif; text-align: center; margin: 40px; }
            table { margin: 20px auto; border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ccc; padding: 10px; vertical-align: top; }
            th { background-color: #f0f0f0; font-weight: bold; }
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
            .expressions-list {
                text-align: left;
                background-color: #f9f9f9;
                padding: 15px;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin-top: 20px;
            }
            .expressions-list li {
                margin-bottom: 8px;
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
            .mask-toggle-container {
                margin: 10px 0;
            }
            .mask-toggle {
                padding: 8px 16px;
                cursor: pointer;
                background-color: #f0f0f0;
                border: 1px solid #ccc;
                border-radius: 4px;
            }
            .mask-toggle.active {
                background-color: #4CAF50;
                color: white;
            }
        </style>
    </head>
    <body>
        <div class="back-link">
            <a href="/?split={{ split }}">← Back to Image Selection</a>
        </div>

        <div class="random-nav" style="margin: 20px 0;">
            <a href="/random?split={{ split }}&type={{ item_type }}" class="random-button" style="text-decoration: none; display: inline-block;">
                Next Random Object
            </a>
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
                <img src="/image/{{ patch_idx }}?type={{ item_type }}&split={{ split }}&image_id={{ image_id }}&item_idx={{ item_idx }}&show_masks={{ request.args.get('show_masks', 'true') }}"><br>

                <div class="mask-toggle-container">
                    <button class="mask-toggle {{ 'active' if request.args.get('show_masks', 'true') == 'true' else '' }}" 
                            onclick="toggleMasks()">Toggle Masks</button>
                </div>

                <div class="stats">
                    <p>Split: {{ split|upper }}</p>
                    <p>Image ID: {{ image_id }}</p>
                    <p>Patch: {{ current_patch }}</p>
                    <p>Item {{ item_idx + 1 }} of {{ total_items }} in current patch</p>
                    <p>Number of Expressions: {{ (item['expressions']|length) + (item['enhanced_expressions']|length) + (item['unique_expressions']|length) }}</p>
                </div>
            </div>

            <div class="expressions-container">
                {% if item['expressions'] %}
                <div class="expressions-list">
                    <h3>Original Rule-Based Expressions:</h3>
                    <ol>
                        {% for expr in item['expressions'] %}
                        <li>{{ expr }}</li>
                        {% endfor %}
                    </ol>
                </div>
                {% endif %}
                
                {% if item['enhanced_expressions'] %}
                <div class="expressions-list" style="background-color: #e8f5e8; border-color: #4CAF50;">
                    <h3>Enhanced Expressions:</h3>
                    <ol>
                        {% for expr in item['enhanced_expressions'] %}
                        <li>{{ expr }}</li>
                        {% endfor %}
                    </ol>
                </div>
                {% endif %}
                
                {% if item['unique_expressions'] %}
                <div class="expressions-list" style="background-color: #fff3cd; border-color: #ffc107;">
                    <h3>Unique Expressions:</h3>
                    <ol>
                        {% for expr in item['unique_expressions'] %}
                        <li>{{ expr }}</li>
                        {% endfor %}
                    </ol>
                </div>
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

        <div class="split-selector" style="margin-top: 30px;">
            <a href="/?split=train&image_id={{ image_id }}&patch_idx=0&item_idx=0&type={{ item_type }}"><button class="split-button {{ 'active' if split == 'train' else '' }}">Train Split</button></a>
            <a href="/?split=val&image_id={{ image_id }}&patch_idx=0&item_idx=0&type={{ item_type }}"><button class="split-button {{ 'active' if split == 'val' else '' }}">Validation Split</button></a>
        </div>
        
        <div class="filter-buttons" style="margin: 20px 0;">
            <a href="/?split={{ split }}&image_id={{ image_id }}&patch_idx={{ patch_idx }}&item_idx=0&type=all"><button class="filter-button {{ 'active' if item_type == 'all' else '' }}">All</button></a>
            <a href="/?split={{ split }}&image_id={{ image_id }}&patch_idx={{ patch_idx }}&item_idx=0&type=instance"><button class="filter-button {{ 'active' if item_type == 'instance' else '' }}">Instances Only</button></a>
            <a href="/?split={{ split }}&image_id={{ image_id }}&patch_idx={{ patch_idx }}&item_idx=0&type=group"><button class="filter-button {{ 'active' if item_type == 'group' else '' }}">Groups Only</button></a>
        </div>

        <script>
            function toggleMasks() {
                const currentShowMasks = new URLSearchParams(window.location.search).get('show_masks') || 'true';
                const newShowMasks = currentShowMasks === 'true' ? 'false' : 'true';
                const url = new URL(window.location.href);
                url.searchParams.set('show_masks', newShowMasks);
                window.location.href = url.toString();
            }
        </script>
    </body>
    </html>
    """, item=item, patch_idx=patch_idx, item_idx=item_idx, total_patches=total_patches, 
         total_items=total_items, item_type=item_type, split=split, image_id=image_id, 
         current_patch=current_patch)

@app.route('/image/<int:patch_idx>')
def image(patch_idx):
    item_type = request.args.get('type', 'all')
    split = request.args.get('split', 'train')
    image_id = request.args.get('image_id')
    item_idx = int(request.args.get('item_idx', 0))
    show_masks = request.args.get('show_masks', 'true').lower() == 'true'
    
    if not image_id:
        return "Image ID not specified", 404
    
    # Get patches for selected image
    patches = get_patches_for_image(image_id, split)
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

    # Parse XML file once for all mask operations
    tree = ET.parse(os.path.join(ANNOTATIONS_DIR, split, 'annotations', current_patch))
    root = tree.getroot()
    
    # Create a dictionary of object IDs to their segmentation data
    seg_data_dict = {}
    for obj in root.findall('object'):
        obj_id = obj.find('id').text
        seg_elem = obj.find('segmentation')
        if seg_elem is not None:
            try:
                seg_data = eval(seg_elem.text)
                if isinstance(seg_data, dict) and 'size' in seg_data and 'counts' in seg_data:
                    seg_data_dict[obj_id] = seg_data
            except Exception as e:
                print(f"Error processing mask for object {obj_id}: {e}")

    def draw_mask(img, seg_data, color):
        try:
            rle = {'size': seg_data['size'], 'counts': seg_data['counts'].encode('utf-8')}
            mask = mask_util.decode(rle)
            
            # Create a colored overlay for the mask
            overlay = img.copy()
            overlay[mask == 1] = color
            
            # Blend the overlay with the original image
            alpha = 0.3  # Transparency factor
            cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        except Exception as e:
            print(f"Error drawing mask: {e}")

    if item['type'] == 'instance':
        x1, y1, x2, y2 = item['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # Draw mask for instance if enabled
        if show_masks and item['obj_id'] in seg_data_dict:
            draw_mask(img, seg_data_dict[item['obj_id']], [0, 255, 0])  # Green for instance
    else:
        # For groups, first check if we have a combined group mask
        if show_masks and item.get('group_segmentation'):
            # Use the combined group mask
            try:
                draw_mask(img, item['group_segmentation'], [255, 0, 255])  # Magenta for combined group mask
            except Exception as e:
                print(f"Error drawing group mask: {e}")
                # Fallback to individual member masks
                for member_id in item.get('member_ids', []):
                    if member_id in seg_data_dict:
                        draw_mask(img, seg_data_dict[member_id], [0, 165, 255])  # Orange for group members
        
        # Draw member bounding boxes
        for member_bbox in item.get('member_bboxes', []):
            x1, y1, x2, y2 = member_bbox
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 1)
        
        # Draw group bounding box
        x1, y1, x2, y2 = item['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    _, buffer = cv2.imencode('.png', img)
    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype='image/png'
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rule-Based Expression Viewer')
    parser.add_argument('--dataset_dir', type=str, default=DEFAULT_DATASET_DIR,
                      help=f'Path to dataset directory (default: {DEFAULT_DATASET_DIR})')
    args = parser.parse_args()
    
    # Update global paths
    ANNOTATIONS_DIR = os.path.join(args.dataset_dir, "patches_rules_expressions_unique")
    IMAGES_DIR = os.path.join(args.dataset_dir, "patches")
    
    print(f"Using dataset directory: {args.dataset_dir}")
    print(f"Annotations directory: {ANNOTATIONS_DIR}")
    print(f"Images directory: {IMAGES_DIR}")
    
    app.run(debug=True, port=5004) 