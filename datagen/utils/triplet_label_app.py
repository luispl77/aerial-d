import os
import io
import json
import random
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from flask import Flask, render_template_string, send_file, request, redirect, url_for
import cv2
import numpy as np
import copy
from google.cloud import storage

app = Flask(__name__)

# Paths for input data
ANNOTATIONS_DIR = "dataset/patches_rules_expressions_unique_llm/annotations"
IMAGES_DIR = "dataset/patches/images"

# Paths for output data (will be created if they don't exist)
OUTPUT_DIR = "gemini_triplet_finetuning_data"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "training_data.jsonl")
METADATA_JSON_PATH = os.path.join(OUTPUT_DIR, "metadata.json")

# Google Cloud Storage settings
GCS_BUCKET_NAME = "aerial-bucket"  # Replace with your actual bucket name
GCS_CREDENTIALS_PATH = "gen-lang-client-0356477555-51def127f63b.json"  # Replace with path to your credentials file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = GCS_CREDENTIALS_PATH

# Ensure output directories exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

# --- Data Loading (Similar to previous version) ---
def read_annotations():
    all_items = []
    
    for xml_file in os.listdir(ANNOTATIONS_DIR):
        if not xml_file.endswith('.xml'):
            continue
        
        try:
            tree = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file))
        except ET.ParseError:
            print(f"Warning: Skipping corrupted XML file: {xml_file}")
            continue

        root = tree.getroot()
        image_filename_elem = root.find('filename')
        if image_filename_elem is None or not image_filename_elem.text:
            print(f"Warning: Skipping XML file without filename: {xml_file}")
            continue
        image_filename = image_filename_elem.text
        image_path = os.path.join(IMAGES_DIR, image_filename)
        
        if not os.path.exists(image_path):
            continue
            
        objects_by_id = {}
        
        # Process individual objects
        for obj in root.findall('object'):
            obj_id_elem = obj.find('id')
            name_elem = obj.find('name')
            bbox_elem = obj.find('bndbox')

            if obj_id_elem is None or name_elem is None or bbox_elem is None:
                 # Skip incomplete objects silently
                 continue

            obj_id = obj_id_elem.text
            category = name_elem.text

            try:
                bbox = [
                    int(bbox_elem.find('xmin').text),
                    int(bbox_elem.find('ymin').text),
                    int(bbox_elem.find('xmax').text),
                    int(bbox_elem.find('ymax').text)
                ]
            except (ValueError, AttributeError):
                # Skip if bbox is invalid
                continue
            
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
                    if expr.text and expr.text.strip(): # Ensure enhanced text exists
                        enhanced_by_base[base_id].append(expr.text)

                for orig_idx, orig_expr in enumerate(original_expressions):
                    base_id = str(orig_idx) # Assuming base_id corresponds to index if missing
                    # Find the correct base_id from the original expressions list
                    for expr_elem in expressions_elem.findall('expression'):
                        if expr_elem.text == orig_expr:
                           base_id = expr_elem.get('id', str(orig_idx))
                           break

                    if base_id in enhanced_by_base and len(enhanced_by_base[base_id]) == 3:
                        enhanced_expressions.append({
                            'original': orig_expr,
                            'enhanced': enhanced_by_base[base_id]
                        })
                    elif base_id in enhanced_by_base and len(enhanced_by_base[base_id]) > 0:
                         # Handle cases where not exactly 3 were generated, pad if necessary
                         padded_enhanced = enhanced_by_base[base_id] + [""] * (3 - len(enhanced_by_base[base_id]))
                         enhanced_expressions.append({
                             'original': orig_expr,
                             'enhanced': padded_enhanced[:3] # Ensure only 3
                         })


            raw_llm_description = None
            desc_elem = obj.find('raw_llm_description')
            if desc_elem is not None and desc_elem.text:
                raw_llm_description = desc_elem.text.strip()
                
            # Only add if we have expressions to correct
            if enhanced_expressions:
                all_items.append({
                    'obj_id': obj_id,
                    'category': category,
                    'bbox': bbox,
                    'image_path': image_path,
                    'image_filename': image_filename,
                    'expressions': enhanced_expressions, # List of {'original': str, 'enhanced': [str, str, str]}
                    'raw_description': raw_llm_description if raw_llm_description else "", # Default to empty string
                    'type': 'instance'
                })
        
        # Process groups
        groups_elem = root.find('groups')
        if groups_elem is not None:
            for group in groups_elem.findall('group'):
                group_id_elem = group.find('id')
                instance_ids_elem = group.find('instance_ids')
                category_elem = group.find('category')

                if group_id_elem is None or instance_ids_elem is None or not instance_ids_elem.text:
                    continue # Skip incomplete groups silently

                group_id = group_id_elem.text
                category = category_elem.text if category_elem is not None else "Unknown"
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
                        if expr.text and expr.text.strip(): # Ensure enhanced text exists
                            enhanced_by_base[base_id].append(expr.text)
                            
                    for orig_idx, orig_expr in enumerate(original_expressions):
                        base_id = str(orig_idx) # Default assumption
                         # Find the correct base_id from the original expressions list
                        for expr_elem in expressions_elem.findall('expression'):
                            if expr_elem.text == orig_expr:
                               base_id = expr_elem.get('id', str(orig_idx))
                               break

                        if base_id in enhanced_by_base and len(enhanced_by_base[base_id]) == 3:
                            enhanced_expressions.append({
                                'original': orig_expr,
                                'enhanced': enhanced_by_base[base_id]
                            })
                        elif base_id in enhanced_by_base and len(enhanced_by_base[base_id]) > 0:
                             # Handle cases where not exactly 3 were generated, pad if necessary
                             padded_enhanced = enhanced_by_base[base_id] + [""] * (3 - len(enhanced_by_base[base_id]))
                             enhanced_expressions.append({
                                 'original': orig_expr,
                                 'enhanced': padded_enhanced[:3] # Ensure only 3
                             })
                
                raw_llm_description = None
                desc_elem = group.find('raw_llm_description')
                if desc_elem is not None and desc_elem.text:
                    raw_llm_description = desc_elem.text.strip()
                
                if enhanced_expressions:
                    all_items.append({
                        'group_id': group_id,
                        'category': category,
                        'bbox': group_bbox,
                        'member_ids': instance_ids,
                        'member_bboxes': member_bboxes,
                        'image_path': image_path,
                        'image_filename': image_filename,
                        'expressions': enhanced_expressions, # List of {'original': str, 'enhanced': [str, str, str]}
                        'raw_description': raw_llm_description if raw_llm_description else "", # Default to empty string
                        'type': 'group'
                    })
    
    return all_items

# --- Finetuning Data Handling ---
def load_training_data():
    data = {"examples": {}, "item_map": {}} # Ensure item_map exists
    if os.path.exists(METADATA_JSON_PATH):
        try:
            with open(METADATA_JSON_PATH, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}. Starting fresh.")
            data = {"examples": {}, "item_map": {}} # Reset on error
    if "examples" not in data: data["examples"] = {}
    if "item_map" not in data: data["item_map"] = {}
    return data

def save_training_data(data):
    if os.path.exists(METADATA_JSON_PATH):
        backup_path = f"{METADATA_JSON_PATH}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            os.rename(METADATA_JSON_PATH, backup_path)
        except OSError as e:
            print(f"Warning: Could not create backup file: {e}")
            
    with open(METADATA_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)
        
    with open(OUTPUT_JSON_PATH, 'w') as f:
        for example_data in data["examples"].values():
            # Ensure we only write the necessary fields for Gemini
            jsonl_entry = {
                "systemInstruction": example_data["systemInstruction"],
                "contents": [ # Should contain user and model turns
                    example_data["contents"][0], # User turn
                    example_data["contents"][1]  # Model turn
                ]
            }
            f.write(json.dumps(jsonl_entry) + '\n')

def generate_image_filename(item):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category = item['category'].replace(' ', '_')
    item_type = item['type']
    item_id = item.get('group_id', item.get('obj_id', 'unknown'))
    return f"{item_type}_{category}_{item_id}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"

# --- Load Data ---
ITEMS = read_annotations()
TRAINING_DATA = load_training_data()

# --- Flask Routes ---
@app.route('/')
def index():
    idx = int(request.args.get('idx', 0))
    item_type = request.args.get('type', 'all')
    
    filtered_items = ITEMS
    if item_type == 'instance':
        filtered_items = [item for item in ITEMS if item.get('type') == 'instance']
    elif item_type == 'group':
        filtered_items = [item for item in ITEMS if item.get('type') == 'group']
    
    total = len(filtered_items)
    if total == 0:
        return "No items found with the selected filter", 404
        
    # Ensure index is valid
    if idx < 0 or idx >= total:
       idx = 0 # Reset index if out of bounds

    item = filtered_items[idx % total]
    
    # Create unique item identifier
    if item['type'] == 'instance':
        item_id = f"instance_{item['image_filename']}_{item['obj_id']}"
    else:
        item_id = f"group_{item['image_filename']}_{item['group_id']}"
        
    # --- Retrieve corrected data or use originals ---
    corrected_description = item['raw_description'] # Default to original LLM description
    corrected_triplets_list = [] # List to hold triplets for each original expression
    existing_image_filename = ""
    is_editing = False # Flag to show if we are editing a saved correction

    # Initialize corrected_triplets_list with original LLM enhanced versions
    for expr_set in item['expressions']:
         # Ensure enhanced is a list of 3 strings, padding if necessary
        enhanced = expr_set.get('enhanced', ["", "", ""])
        if not isinstance(enhanced, list): enhanced = ["", "", ""]
        while len(enhanced) < 3: enhanced.append("")
        corrected_triplets_list.append(list(enhanced[:3])) # Use a copy


    # Check if we have saved corrections for this item
    if item_id in TRAINING_DATA.get('item_map', {}):
        metadata_entry = TRAINING_DATA['item_map'][item_id]
        existing_image_filename = metadata_entry.get('image_filename', '')
        example_id = metadata_entry.get('example_id', '')
        
        if example_id in TRAINING_DATA.get('examples', {}):
            example = TRAINING_DATA['examples'][example_id]
            # Find the 'model' part to extract the corrected data
            model_content = next((c for c in example.get('contents', []) if c.get('role') == 'model'), None)

            if model_content:
                model_part_text = model_content.get('parts', [{}])[0].get('text', '{}')
                try:
                    # Parse the JSON string stored in the model part
                    corrected_data = json.loads(model_part_text)
                    
                    # Get corrected description
                    corrected_description = corrected_data.get('description', item['raw_description'])
                    
                    # Get corrected triplets, ensuring structure matches item['expressions']
                    corrected_enhanced_expressions = corrected_data.get('enhanced_expressions', [])
                    
                    # Overwrite corrected_triplets_list with saved data
                    temp_corrected_triplets = []
                    original_expr_texts = [e['original'] for e in item['expressions']]

                    for i, orig_expr_text in enumerate(original_expr_texts):
                        found_match = False
                        for corrected_expr_set in corrected_enhanced_expressions:
                            if corrected_expr_set.get('original_expression') == orig_expr_text:
                                versions = corrected_expr_set.get('enhanced_versions', ["", "", ""])
                                while len(versions) < 3: versions.append("")
                                temp_corrected_triplets.append(list(versions[:3])) # Use a copy
                                found_match = True
                                break
                        if not found_match:
                            # If no match found in saved data, use the original LLM version
                            # Make sure index i is valid for item['expressions']
                            if i < len(item['expressions']):
                                original_enhanced = item['expressions'][i].get('enhanced', ["", "", ""])
                                if not isinstance(original_enhanced, list): original_enhanced = ["", "", ""]
                                while len(original_enhanced) < 3: original_enhanced.append("")
                                temp_corrected_triplets.append(list(original_enhanced[:3]))
                            else:
                                # Fallback if index is somehow out of bounds
                                temp_corrected_triplets.append(["", "", ""])


                    # Only update if the number of expressions matches
                    if len(temp_corrected_triplets) == len(item['expressions']):
                         corrected_triplets_list = temp_corrected_triplets
                         is_editing = True # Mark as editing existing data
                    else:
                         print(f"Warning: Mismatch in number of expressions for {item_id}. Using original LLM data.")


                except json.JSONDecodeError:
                    print(f"Warning: Could not parse corrected data for {item_id}. Using original LLM data.")
                    is_editing = False # Not editing if data is corrupted
                except Exception as e:
                     print(f"Error loading corrected data for {item_id}: {e}. Using original LLM data.")
                     is_editing = False # Reset flag on error
            else:
                 is_editing = False # No model content found
        else:
            is_editing = False # Example ID not found
    else:
        is_editing = False # Item not in map


    corrected_count = len(TRAINING_DATA.get("examples", {}))

    # Prepare data for template
    template_data = {
        "item": item,
        "idx": idx,
        "total": total,
        "item_type": item_type,
        "corrected_count": corrected_count,
        "corrected_description": corrected_description,
        "corrected_triplets_list": corrected_triplets_list, # This is now a list of lists
        "existing_image_filename": existing_image_filename,
        "is_editing": is_editing,
        "status": request.args.get('status', '')
    }

    # Add enumerate to the context passed to the template
    template_context = {**template_data, "enumerate": enumerate, "len": len} # Added len as well, just in case

    return render_template_string("""
    <html>
    <head>
        <title>Triplet Expression Correction (Multi-Expression)</title>
        <style>
            body { font-family: sans-serif; margin: 40px; }
            .container { max-width: 1200px; margin: 0 auto; }
            img { max-width: 100%; max-height: 500px; border: 1px solid #ccc; display: block; margin: 20px auto; }
            .nav, .filter-buttons, .progress, .direct-nav { margin-bottom: 20px; text-align: center; }
            .direct-nav form { display: inline-block; }
            .nav-input { width: 60px; margin: 0 5px; }
            textarea { width: 98%; height: 80px; margin: 5px 0; padding: 8px; font-size: 14px; border: 1px solid #ccc; border-radius: 4px; }
            .filter-button { padding: 8px 16px; margin: 0 5px; cursor: pointer; background-color: #f0f0f0; border: 1px solid #ccc; border-radius: 4px; }
            .filter-button.active { background-color: #4CAF50; color: white; }
            .item-type { display: inline-block; padding: 3px 8px; margin-left: 10px; border-radius: 4px; font-size: 0.8em; color: white; }
            .item-type.instance { background-color: #5bc0de; }
            .item-type.group { background-color: #f0ad4e; }
            .save-btn, .delete-btn { padding: 10px 20px; border: none; border-radius: 4px; cursor: pointer; font-size: 16px; margin-top: 10px; }
            .save-btn { background-color: #4CAF50; color: white; }
            .save-btn:hover { background-color: #45a049; }
            .delete-btn { background-color: #e74c3c; color: white; margin-left: 10px; }
            .delete-btn:hover { background-color: #c0392b; }
            .status { margin-top: 10px; color: #4CAF50; font-weight: bold; text-align: center; }
            .edit-mode { background-color: #fff7e0; border: 1px solid #ffe0a0; padding: 5px 10px; display: inline-block; margin-bottom: 10px; border-radius: 4px; }
            .expression-block { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 20px; background-color: #fafafa; }
            .expression-block h4 { margin-top: 0; border-bottom: 1px solid #eee; padding-bottom: 5px; }
            .original-text { font-style: italic; color: #555; margin-bottom: 10px; background-color: #eee; padding: 8px; border-radius: 4px; }
            .triplet-section { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 15px; }
            .triplet-section label { display: block; font-weight: bold; margin-bottom: 5px; }
            .description-section { margin-bottom: 20px; }
            .description-section textarea { height: 120px; }
            .original-llm-box { background-color: #f0f8ff; border: 1px solid #cce4ff; padding: 8px; margin-top: 5px; border-radius: 4px; font-size: 0.9em; color: #333; }
            .original-llm-box strong { color: #0056b3; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Triplet Expression Correction (Multi-Expression)</h1>
            
            <div class="progress">
                Progress: {{ corrected_count }} corrected items | Item {{ idx + 1 }} of {{ total }}
                {% if is_editing %}
                    <span class="edit-mode">Editing existing correction</span>
                {% endif %}
            </div>

            <div class="direct-nav">
                <form method="get" action="/">
                    <label>Go to item:</label>
                    <input type="number" name="idx" class="nav-input" min="0" max="{{ total - 1 }}" value="{{ idx }}">
                    <input type="hidden" name="type" value="{{ item_type }}">
                    <input type="submit" value="Go">
                </form>
            </div>
            
            <div class="filter-buttons">
                <a href="/?type=all&idx=0"><button class="filter-button {{ 'active' if item_type == 'all' else '' }}">All</button></a>
                <a href="/?type=instance&idx=0"><button class="filter-button {{ 'active' if item_type == 'instance' else '' }}">Instances Only</button></a>
                <a href="/?type=group&idx=0"><button class="filter-button {{ 'active' if item_type == 'group' else '' }}">Groups Only</button></a>
            </div>
            
            <h2>
                {% if item['type'] == 'instance' %}
                    {{ item['category'] }} (ID: {{ item['obj_id'] }}) <span class="item-type instance">Instance</span>
                {% else %}
                    Group of {{ item['category'] }} (ID: {{ item['group_id'] }}) <span class="item-type group">Group</span>
                {% endif %}
                 - {{ item['image_filename'] }}
            </h2>
            
            <img src="/image/{{ idx }}?type={{ item_type }}">
            
            <form method="post" action="/save">
                <!-- Description Section -->
                <div class="description-section expression-block">
                    <h4>LLM Description Correction</h4>
                    <label for="description">Corrected Description:</label>
                    <textarea id="description" name="description" placeholder="Enter corrected factual description (150-200 words)">{{ corrected_description }}</textarea>
                    <div class="original-llm-box">
                        <strong>Original LLM Description:</strong><br>
                        {{ item['raw_description'] if item['raw_description'] else '(No original description provided)' }}
                    </div>
                </div>

                <!-- Enhanced Expressions Section -->
                <h4>Enhanced Expressions Correction</h4>
                {% for i, expr_set in enumerate(item['expressions']) %} {# Enumerate is now available #}
                <div class="expression-block">
                    <h4>Correction for Original Expression #{{ i+1 }}</h4>
                    <div class="original-text"><strong>Rule-Based Original:</strong> {{ expr_set['original'] }}</div>
                    
                    <div class="triplet-section">
                        <div>
                            <label for="triplet_{{ i }}_0">Corrected Short (5-20 words):</label>
                            <textarea id="triplet_{{ i }}_0" name="triplet_{{ i }}_0" placeholder="Corrected short version">{{ corrected_triplets_list[i][0] }}</textarea>
                            <div class="original-llm-box">
                                <strong>Original LLM Short:</strong><br>
                                {{ expr_set['enhanced'][0] if expr_set['enhanced'] and expr_set['enhanced'][0] else '(empty)' }}
                            </div>
                        </div>
                        <div>
                            <label for="triplet_{{ i }}_1">Corrected Medium (20-30 words):</label>
                            <textarea id="triplet_{{ i }}_1" name="triplet_{{ i }}_1" placeholder="Corrected medium version">{{ corrected_triplets_list[i][1] }}</textarea>
                             <div class="original-llm-box">
                                <strong>Original LLM Medium:</strong><br>
                                {{ expr_set['enhanced'][1] if expr_set['enhanced'] and len(expr_set['enhanced']) > 1 and expr_set['enhanced'][1] else '(empty)' }}
                            </div>
                        </div>
                        <div>
                            <label for="triplet_{{ i }}_2">Corrected Long (30-50 words):</label>
                            <textarea id="triplet_{{ i }}_2" name="triplet_{{ i }}_2" placeholder="Corrected long version">{{ corrected_triplets_list[i][2] }}</textarea>
                             <div class="original-llm-box">
                                <strong>Original LLM Long:</strong><br>
                                {{ expr_set['enhanced'][2] if expr_set['enhanced'] and len(expr_set['enhanced']) > 2 and expr_set['enhanced'][2] else '(empty)' }}
                            </div>
                        </div>
                    </div>
                     <!-- Hidden field to pass the original expression text -->
                    <input type="hidden" name="original_expression_{{ i }}" value="{{ expr_set['original'] }}">
                </div>
                {% endfor %}
                
                <input type="hidden" name="idx" value="{{ idx }}">
                <input type="hidden" name="type" value="{{ item_type }}">
                <input type="hidden" name="num_expressions" value="{{ len(item['expressions']) }}"> {# len is now available #}
                <input type="hidden" name="existing_image_filename" value="{{ existing_image_filename }}">
                
                <div style="text-align: center; margin-top: 20px;">
                    <button type="submit" class="save-btn">Save Correction & Next</button>
                    {% if is_editing %}
                    <button type="button" class="delete-btn" onclick="if(confirm('Are you sure you want to delete this correction? This cannot be undone.')) window.location.href='/delete?idx={{ idx }}&type={{ item_type }}'">Delete Correction</button>
                    {% endif %}
                </div>
                
                {% if status %}
                <div class="status">{{ status }}</div>
                {% endif %}
            </form>
            
            <div class="nav">
                <a href="/?idx={{ (idx - 1) % total }}&type={{ item_type }}">Previous</a> |
                <a href="/?idx={{ (idx + 1) % total }}&type={{ item_type }}">Next</a>
            </div>
        </div>
    </body>
    </html>
    """, **template_context) # Pass the extended context here

@app.route('/image/<int:idx>')
def image(idx):
    item_type = request.args.get('type', 'all')
    
    filtered_items = ITEMS
    if item_type == 'instance':
        filtered_items = [item for item in ITEMS if item.get('type') == 'instance']
    elif item_type == 'group':
        filtered_items = [item for item in ITEMS if item.get('type') == 'group']
    
    if not filtered_items:
        return "No items found", 404
        
    # Ensure index is valid
    if idx < 0 or idx >= len(filtered_items):
       idx = 0 # Reset index if out of bounds

    item = filtered_items[idx % len(filtered_items)]
    img = cv2.imread(item['image_path'])
    if img is None:
        return "Image not found", 404
        
    img_copy = img.copy() # Work on a copy to avoid modifying the original image data
    
    # Draw bounding box(es)
    if item['type'] == 'instance':
        x1, y1, x2, y2 = item['bbox']
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2) # Red box
    else: # Group
        # Member boxes in blue (thin)
        for member_bbox in item.get('member_bboxes', []):
            x1, y1, x2, y2 = member_bbox
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 1) 
        # Group box in red (thick)
        x1, y1, x2, y2 = item['bbox']
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2) 
        
    _, buffer = cv2.imencode('.png', img_copy)
    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype='image/png'
    )

@app.route('/save', methods=['POST'])
def save():
    # --- Get form data ---
    idx = int(request.form.get('idx', 0))
    item_type = request.form.get('type', 'all')
    corrected_description = request.form.get('description', '').strip()
    num_expressions = int(request.form.get('num_expressions', 0))
    existing_image_filename = request.form.get('existing_image_filename', '')
    
    # --- Retrieve all corrected triplets ---
    corrected_enhanced_expressions_list = []
    all_triplets_valid = True
    for i in range(num_expressions):
        original_expr = request.form.get(f'original_expression_{i}', '')
        triplet = [
            request.form.get(f'triplet_{i}_0', '').strip(),
            request.form.get(f'triplet_{i}_1', '').strip(),
            request.form.get(f'triplet_{i}_2', '').strip(),
        ]
        # Basic validation: ensure all parts of the triplet are non-empty
        if not all(triplet):
            all_triplets_valid = False
            # Break early if any triplet is incomplete
            break 
            
        corrected_enhanced_expressions_list.append({
            "original_expression": original_expr,
            "enhanced_versions": triplet
        })

    # --- Validate input ---
    if not corrected_description or not all_triplets_valid:
        return redirect(url_for('index', idx=idx, type=item_type, 
                               status="Error: Description and all three parts of every triplet are required"))
    if len(corrected_enhanced_expressions_list) != num_expressions:
         return redirect(url_for('index', idx=idx, type=item_type, 
                               status="Error: Mismatch in the number of expressions processed."))

    # --- Get current item ---
    filtered_items = ITEMS
    if item_type == 'instance':
        filtered_items = [item for item in ITEMS if item.get('type') == 'instance']
    elif item_type == 'group':
        filtered_items = [item for item in ITEMS if item.get('type') == 'group']
        
    if not filtered_items or idx >= len(filtered_items):
         return redirect(url_for('index', idx=0, type=item_type, # Go back to first item
                               status="Error: Item index out of bounds"))

    item = filtered_items[idx % len(filtered_items)] # Use modulo just in case

    # --- Create unique item identifier ---
    if item['type'] == 'instance':
        item_id = f"instance_{item['image_filename']}_{item['obj_id']}"
    else:
        item_id = f"group_{item['image_filename']}_{item['group_id']}"

    # --- Check if updating or creating ---
    is_update = item_id in TRAINING_DATA.get('item_map', {})

    # --- Handle image saving ---
    img = cv2.imread(item['image_path'])
    if img is None:
        return redirect(url_for('index', idx=idx, type=item_type, 
                              status=f"Error: Could not read image {item['image_path']}"))
                              
    img_copy = img.copy() # Work on a copy
    
    # Draw bounding box(es)
    if item['type'] == 'instance':
        x1, y1, x2, y2 = item['bbox']
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2) 
    else: # Group
        for member_bbox in item.get('member_bboxes', []):
            x1, y1, x2, y2 = member_bbox
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (255, 0, 0), 1) 
        x1, y1, x2, y2 = item['bbox']
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2) 
        
    # Determine image filename and example ID
    if is_update:
        metadata_entry = TRAINING_DATA['item_map'][item_id]
        image_filename = metadata_entry.get('image_filename', '')
        example_id = metadata_entry.get('example_id', '')
        if not image_filename: image_filename = generate_image_filename(item) # Regenerate if missing
        if not example_id: example_id = str(uuid.uuid4()) # Regenerate if missing
    else:
        image_filename = generate_image_filename(item)
        example_id = str(uuid.uuid4())

    # Save the image locally
    image_path = os.path.join(OUTPUT_IMAGES_DIR, image_filename)
    success = cv2.imwrite(image_path, img_copy)
    if not success:
        print(f"Warning: Failed to save image to {image_path}")
        return redirect(url_for('index', idx=idx, type=item_type, status=f"Error: Failed to save image"))

    # Upload the image to Google Cloud Storage
    gcs_success = upload_to_gcs(image_path, image_filename)
    if not gcs_success:
        print(f"Warning: Failed to upload image to GCS: {image_filename}")
        # You might want to continue anyway, as the local file was saved

    # --- Create Gemini Finetuning Data Structure ---
    # Use the exact system prompt from 5_llm_enhancement.py
    system_prompt = (
        "You have two tasks:\n\n"
        "TASK 1: Describe the object highlighted with a red bounding box factually and concisely. Focus on:\n"
        "1. Observable visual features (shape, color, orientation) without speculation\n"
        "2. Definite spatial relationships to nearby objects that are clearly visible\n"
        "3. Distinctive identifying characteristics that would help locate this specific object\n"
        "4. Actual contextual elements (roads, buildings, terrain) that are directly observable\n"
        "Limit your description to 150-200 words. Only describe what you can see with high confidence.\n\n"
        
        f"TASK 2: For each original expression listed below, create EXACTLY 3 enhanced versions that:\n"
        "1. MUST PRESERVE ALL SPATIAL INFORMATION from the original expression:\n"
        "   - Absolute positions (e.g., \"in the top right\", \"near the center\")\n"
        "   - Relative positions (e.g., \"to the right of\", \"below\")\n"
        "2. Add relevant visual details from what you observe in the image\n"
        "3. Ensure each expression uniquely identifies this object to avoid ambiguity\n"
        "4. Vary the length of the enhanced expressions as follows:\n"
        "   - First version: 5–20 words\n"
        "   - Second version: 20–30 words\n"
        "   - Third version: 30–50 words\n\n"
        
        "You must return your output in the following JSON format:\n"
        "{\n"
        "  \"description\": \"<factual description of the object>\",\n"
        "  \"enhanced_expressions\": [\n"
        "    {\n"
        "      \"original_expression\": \"<original expression>\",\n"
        "      \"enhanced_versions\": [\n"
        "        \"<short version>\",\n"
        "        \"<medium version>\",\n"
        "        \"<long version>\"\n"
        "      ]\n"
        "    },\n"
        "    ...\n"
        "  ]\n"
        "}\n"
    )

    # Format original expressions for the user prompt
    original_expressions_for_prompt = "\n".join([f"- {expr['original']}" for expr in item['expressions']])

    # Use the exact user prompt structure from 5_llm_enhancement.py
    user_prompt_text = (
         f"Provide a factual description of this {item['category']} (highlighted with a red bounding box), "
         f"and enhance the provided expressions to make them more descriptive while preserving all spatial information.\n\n"
         f"ORIGINAL EXPRESSIONS TO ENHANCE:\n{original_expressions_for_prompt}"
    )

    # Construct the 'model' output JSON string with corrected data
    model_output_structure = {
        "description": corrected_description,
        "enhanced_expressions": corrected_enhanced_expressions_list
    }
    # Use compact JSON encoding for the model output string
    model_output_json_string = json.dumps(model_output_structure, separators=(',', ':'))


    # Create the full example structure for JSONL
    example = {
        "systemInstruction": {
            "role": "system",
            "parts": [{"text": system_prompt}]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"fileData": {
                        "mimeType": "image/jpeg",
                        # Use the actual GCS bucket name from the settings
                        "fileUri": f"gs://{GCS_BUCKET_NAME}/{image_filename}" 
                    }},
                    {"text": user_prompt_text}
                ]
            },
            {
                "role": "model",
                "parts": [{"text": model_output_json_string}] # Store the structured output as a JSON string
            }
        ],
        # --- Metadata (Not included in JSONL, only in metadata.json) ---
        "metadata": { 
            "item_id": item_id,
            "timestamp": datetime.now().isoformat(),
            "image_filename": image_filename,
             # Add original item details for reference
            "original_image_filename": item['image_filename'],
            "category": item['category'],
            "type": item['type'],
            "internal_id": item.get('obj_id', item.get('group_id')),
            "bbox": item['bbox'],
            "gcs_uri": f"gs://{GCS_BUCKET_NAME}/{image_filename}"
        }
    }

    # --- Update and Save Data ---
    if 'item_map' not in TRAINING_DATA: TRAINING_DATA['item_map'] = {} # Ensure map exists
    if 'examples' not in TRAINING_DATA: TRAINING_DATA['examples'] = {} # Ensure examples exists
    
    TRAINING_DATA['item_map'][item_id] = {
        'image_filename': image_filename,
        'example_id': example_id
        # Add any other quick lookup info if needed
    }
    TRAINING_DATA['examples'][example_id] = example

    try:
        save_training_data(TRAINING_DATA)
        status_message = f"Correction {'Updated' if is_update else 'Saved'}! ({len(TRAINING_DATA['examples'])} examples total)"
    except Exception as e:
        status_message = f"Error saving data: {str(e)}"
        print(status_message) # Log error server-side

    # --- Redirect to next item ---
    next_idx = (idx + 1) # Calculate next index before modulo
    if next_idx >= len(filtered_items): # Check if we wrap around
       next_idx = 0
    
    return redirect(url_for('index', idx=next_idx, type=item_type, status=status_message))

@app.route('/delete')
def delete_item():
    # Get parameters
    idx = int(request.args.get('idx', 0))
    item_type = request.args.get('type', 'all')
    
    # Get current item
    filtered_items = ITEMS
    if item_type == 'instance':
        filtered_items = [item for item in ITEMS if item.get('type') == 'instance']
    elif item_type == 'group':
        filtered_items = [item for item in ITEMS if item.get('type') == 'group']
        
    if not filtered_items or idx >= len(filtered_items):
         return redirect(url_for('index', idx=0, type=item_type,
                               status="Error: Item index out of bounds during delete"))

    item = filtered_items[idx % len(filtered_items)]

    # Create unique item identifier
    if item['type'] == 'instance':
        item_id = f"instance_{item['image_filename']}_{item['obj_id']}"
    else:
        item_id = f"group_{item['image_filename']}_{item['group_id']}"
        
    status = "Correction not found in dataset" # Default status

    # Check if this item exists in our data
    if item_id in TRAINING_DATA.get('item_map', {}):
        metadata_entry = TRAINING_DATA['item_map'][item_id]
        example_id = metadata_entry.get('example_id', '')
        image_filename = metadata_entry.get('image_filename', '')
        
        # Delete from item map
        del TRAINING_DATA['item_map'][item_id]
        
        # Delete from examples if example_id exists
        if example_id and example_id in TRAINING_DATA.get('examples', {}):
            del TRAINING_DATA['examples'][example_id]
        else:
             print(f"Warning: Example ID {example_id} not found in examples dict for item {item_id}")

        # Delete the image file locally if it exists and filename is known
        if image_filename:
            image_path = os.path.join(OUTPUT_IMAGES_DIR, image_filename)
            if os.path.exists(image_path):
                try:
                    os.remove(image_path)
                    print(f"Deleted local image file: {image_path}")
                except Exception as e:
                    print(f"Error deleting local image file {image_path}: {e}")
            else:
                print(f"Warning: Local image file not found for deletion: {image_path}")
                
            # Also delete from GCS if possible
            try:
                storage_client = storage.Client()
                bucket = storage_client.bucket(GCS_BUCKET_NAME)
                blob = bucket.blob(image_filename)
                blob.delete()
                print(f"Deleted GCS image: gs://{GCS_BUCKET_NAME}/{image_filename}")
            except Exception as e:
                print(f"Error deleting GCS image {image_filename}: {e}")
        else:
             print(f"Warning: No image filename found in metadata for item {item_id}, cannot delete image.")

        # Save the updated data
        try:
            save_training_data(TRAINING_DATA)
            status = f"Correction deleted! ({len(TRAINING_DATA.get('examples', {}))} examples remaining)"
        except Exception as e:
            status = f"Error saving data after delete: {str(e)}"
            print(status)
    
    # Redirect back to the same index (or previous if it was the last one)
    # We stay at the same index to allow correcting the original data again if desired
    current_idx = idx 
    if current_idx >= len(filtered_items): # Adjust if deleting the last item caused index issue
        current_idx = max(0, len(filtered_items) -1)

    return redirect(url_for('index', idx=current_idx, type=item_type, status=status))

# Function to upload a file to Google Cloud Storage
def upload_to_gcs(local_file_path, destination_blob_name):
    """Uploads a file to the specified GCS bucket."""
    try:
        # Initialize storage client
        storage_client = storage.Client()
        bucket = storage_client.bucket(GCS_BUCKET_NAME)
        blob = bucket.blob(destination_blob_name)
        
        # Upload the file
        blob.upload_from_filename(local_file_path)
        
        print(f"File {local_file_path} uploaded to gs://{GCS_BUCKET_NAME}/{destination_blob_name}")
        return True
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        return False

if __name__ == '__main__':
    print("Starting server - using Google Cloud Storage bucket:", GCS_BUCKET_NAME)
    # Make sure ITEMS are loaded before running
    if not ITEMS:
        print("Error: No items loaded from annotations. Please check ANNOTATIONS_DIR and image paths.")
    else:
        app.run(debug=True, port=5002) # Use port 5002 