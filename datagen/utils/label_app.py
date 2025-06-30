import os
import io
import json
import random
import uuid
import base64
import xml.etree.ElementTree as ET
from datetime import datetime
from flask import Flask, render_template_string, send_file, request, redirect, url_for
import cv2
import numpy as np

app = Flask(__name__)

# Paths for input data
ANNOTATIONS_DIR = "dataset/patches_rules/annotations"  # Using the rules version without expressions
IMAGES_DIR = "dataset/patches/images"

# Paths for output data (will be created if they don't exist)
OUTPUT_DIR = "gemini_finetuning_data"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "training_data.jsonl")
METADATA_JSON_PATH = os.path.join(OUTPUT_DIR, "metadata.json")  # New metadata file

# Ensure output directories exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)

# Load instances for labeling
def load_instances():
    all_instances = []
    
    for xml_file in os.listdir(ANNOTATIONS_DIR):
        if not xml_file.endswith('.xml'):
            continue
        
        tree = ET.parse(os.path.join(ANNOTATIONS_DIR, xml_file))
        root = tree.getroot()
        image_filename = root.find('filename').text
        image_path = os.path.join(IMAGES_DIR, image_filename)
        
        if not os.path.exists(image_path):
            continue
        
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
            
            # Get position info if available
            position = None
            position_elem = obj.find('grid_position')
            if position_elem is not None and position_elem.text:
                position = position_elem.text
                
            # Add instance to list
            all_instances.append({
                'obj_id': obj_id,
                'category': category,
                'bbox': bbox,
                'position': position,
                'image_path': image_path,
                'image_filename': image_filename
            })
    
    return all_instances

# Load existing training data if available
def load_training_data():
    data = {"examples": {}}
    
    # Load the metadata file for tracking which instances have been labeled
    if os.path.exists(METADATA_JSON_PATH):
        try:
            with open(METADATA_JSON_PATH, 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading metadata: {e}")
    
    # Initialize examples dict if it doesn't exist
    if "examples" not in data:
        data["examples"] = {}
    
    return data

# Save training data
def save_training_data(data):
    # Create a backup of the old metadata file if it exists
    if os.path.exists(METADATA_JSON_PATH):
        backup_path = f"{METADATA_JSON_PATH}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(METADATA_JSON_PATH, backup_path)
    
    # Save metadata (for editing/tracking purposes)
    with open(METADATA_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Write all examples to the JSONL file (one JSON object per line)
    with open(OUTPUT_JSON_PATH, 'w') as f:
        for example_data in data["examples"].values():
            # Create the JSONL format (exclude any metadata we might have added)
            jsonl_entry = {
                "systemInstruction": example_data["systemInstruction"],
                "contents": example_data["contents"]
            }
            f.write(json.dumps(jsonl_entry) + '\n')

# Generate a unique filename for the saved image
def generate_image_filename(instance):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category = instance['category'].replace(' ', '_')
    return f"{category}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"

# Extract a square crop around the instance with some margin
def get_square_crop(img, bbox, margin=0.2):
    x1, y1, x2, y2 = bbox
    
    # Calculate center and current size
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    current_size = max(x2 - x1, y2 - y1)
    
    # Apply margin
    new_size = int(current_size * (1 + margin))
    
    # Calculate new bounds ensuring they stay within image bounds
    h, w = img.shape[:2]
    
    new_x1 = max(0, center_x - new_size // 2)
    new_y1 = max(0, center_y - new_size // 2)
    new_x2 = min(w, center_x + new_size // 2)
    new_y2 = min(h, center_y + new_size // 2)
    
    # If we hit image bounds, adjust to maintain square aspect ratio
    width = new_x2 - new_x1
    height = new_y2 - new_y1
    if width != height:
        if width > height:
            # Shift y to make square if possible
            diff = width - height
            if new_y1 - diff//2 >= 0:
                new_y1 -= diff//2
            else:
                new_y1 = 0
            
            if new_y2 + (diff - diff//2) <= h:
                new_y2 += (diff - diff//2)
            else:
                new_y2 = h
        else:
            # Shift x to make square if possible
            diff = height - width
            if new_x1 - diff//2 >= 0:
                new_x1 -= diff//2
            else:
                new_x1 = 0
            
            if new_x2 + (diff - diff//2) <= w:
                new_x2 += (diff - diff//2)
            else:
                new_x2 = w
    
    # Extract the crop
    crop = img[new_y1:new_y2, new_x1:new_x2]
    
    # Calculate normalized bbox coordinates within the crop
    norm_x1 = x1 - new_x1
    norm_y1 = y1 - new_y1
    norm_x2 = x2 - new_x1
    norm_y2 = y2 - new_y1
    
    return crop, [norm_x1, norm_y1, norm_x2, norm_y2]

# Load all instances once
INSTANCES = load_instances()
TRAINING_DATA = load_training_data()

# Group instances by image path to make navigation easier
def group_instances_by_image():
    image_instances = {}
    for instance in INSTANCES:
        image_path = instance['image_path']
        if image_path not in image_instances:
            image_instances[image_path] = []
        image_instances[image_path].append(instance)
    return image_instances

IMAGE_INSTANCES = group_instances_by_image()
IMAGE_PATHS = list(IMAGE_INSTANCES.keys())

@app.route('/')
def index():
    # Get current image and instance indexes
    img_idx = int(request.args.get('img_idx', 0)) % len(IMAGE_PATHS)
    current_image_path = IMAGE_PATHS[img_idx]
    
    # Get instances for this image
    image_instances = IMAGE_INSTANCES[current_image_path]
    instance_idx = int(request.args.get('instance_idx', 0)) % len(image_instances)
    current_instance = image_instances[instance_idx]
    
    # Create a unique instance identifier
    instance_id = f"{current_instance['image_filename']}_{current_instance['category']}_{current_instance['obj_id']}"
    
    # Check if this instance already has an expression
    existing_expression = ""
    image_filename = ""
    
    # Check if we have this instance in our metadata
    if instance_id in TRAINING_DATA.get('instance_map', {}):
        metadata_entry = TRAINING_DATA['instance_map'][instance_id]
        image_filename = metadata_entry.get('image_filename', '')
        example_id = metadata_entry.get('example_id', '')
        
        # Get the expression if it exists
        if example_id in TRAINING_DATA.get('examples', {}):
            example = TRAINING_DATA['examples'][example_id]
            for content in example.get('contents', []):
                if content.get('role') == 'model':
                    for part in content.get('parts', []):
                        if 'text' in part:
                            existing_expression = part['text']
                            break
                    if existing_expression:
                        break
    
    # Count labeled examples for progress display
    labeled_count = len(TRAINING_DATA.get("examples", {}))
    
    return render_template_string("""
    <html>
    <head>
        <title>Gemini Finetuning Data Labeler</title>
        <style>
            body { font-family: sans-serif; text-align: center; margin: 40px; }
            .container { max-width: 1000px; margin: 0 auto; }
            .image-container { position: relative; display: inline-block; margin: 20px auto; }
            img { max-width: 90%; max-height: 600px; border: 1px solid #ccc; }
            .box { position: absolute; border: 2px solid red; }
            .nav { margin-top: 20px; }
            .progress { margin: 20px 0; text-align: left; }
            textarea { width: 100%; height: 120px; margin: 10px 0; padding: 8px; font-size: 14px; }
            .info-box { background-color: #f9f9f9; border: 1px solid #ddd; 
                      padding: 15px; margin: 10px 0; text-align: left; }
            .save-btn { 
                background-color: #4CAF50; 
                color: white; 
                padding: 10px 20px; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer; 
                font-size: 16px; 
                margin-top: 10px; 
            }
            .save-btn:hover { background-color: #45a049; }
            .delete-btn {
                background-color: #e74c3c;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                margin-top: 10px;
                margin-left: 10px;
            }
            .delete-btn:hover { background-color: #c0392b; }
            .button-group {
                display: flex;
                margin-top: 10px;
            }
            .prompt-container { margin-top: 20px; text-align: left; }
            .status { margin-top: 10px; color: #4CAF50; font-weight: bold; }
            .edit-mode {
                background-color: #fff7e0;
                border: 1px solid #ffe0a0;
                padding: 5px 10px;
                display: inline-block;
                margin-bottom: 10px;
                border-radius: 4px;
            }
            .nav-input {
                padding: 8px;
                width: 80px;
                margin: 0 5px;
                border-radius: 3px;
                border: 1px solid #ccc;
            }
            .direct-nav {
                background-color: #f0f0f0; 
                padding: 10px;
                margin-bottom: 15px;
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
            }
            .direct-nav form {
                display: flex;
                align-items: center;
            }
            .direct-nav button {
                background-color: #4a89dc;
                color: white;
                border: none;
                padding: 8px 15px;
                border-radius: 3px;
                cursor: pointer;
            }
            .direct-nav label {
                margin-right: 5px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Gemini Finetuning Data Labeler</h1>
            
            <div class="progress">
                <p>Progress: {{ labeled_count }} labeled examples</p>
                <p>Current image: {{ img_idx + 1 }}/{{ image_count }} - Instance: {{ instance_idx + 1 }}/{{ len(image_instances) }}</p>
                {% if existing_expression %}
                    <div class="edit-mode">Editing existing expression</div>
                {% endif %}
            </div>
            
            <div class="direct-nav">
                <form method="get" action="/">
                    <label for="goto_img">Go to image:</label>
                    <input type="number" id="goto_img" name="img_idx" class="nav-input" min="0" max="{{ image_count - 1 }}" value="{{ img_idx }}">
                    
                    <label for="goto_instance">Go to instance:</label>
                    <input type="number" id="goto_instance" name="instance_idx" class="nav-input" min="0" max="{{ len(image_instances) - 1 }}" value="{{ instance_idx }}">
                    
                    <button type="submit">Go</button>
                </form>
            </div>
            
            <div class="info-box">
                <p><strong>Category:</strong> {{ current_instance['category'] }}</p>
                <p><strong>Position:</strong> {{ current_instance['position'] if current_instance['position'] else 'Not specified' }}</p>
                <p><strong>Image filename:</strong> {{ current_instance['image_filename'] }}</p>
            </div>
            
            <div class="image-container">
                <img src="/image/{{ img_idx }}/{{ instance_idx }}">
            </div>
            
            <div class="prompt-container">
                <form method="post" action="/save">
                    <textarea name="expression" placeholder="Enter a detailed, precise referring expression that uniquely identifies this object highlighted with a red bounding box.">{{ existing_expression }}</textarea>
                    <input type="hidden" name="img_idx" value="{{ img_idx }}">
                    <input type="hidden" name="instance_idx" value="{{ instance_idx }}">
                    <input type="hidden" name="image_filename" value="{{ image_filename }}">
                    <div class="button-group">
                        <button type="submit" class="save-btn">Save & Next</button>
                        {% if existing_expression %}
                        <button type="button" class="delete-btn" onclick="if(confirm('Are you sure you want to delete this instance from the dataset?')) window.location.href='/delete?img_idx={{ img_idx }}&instance_idx={{ instance_idx }}'">Delete Instance</button>
                        {% endif %}
                    </div>
                </form>
                
                {% if status %}
                <div class="status">{{ status }}</div>
                {% endif %}
            </div>
            
            <div class="nav">
                <a href="/?img_idx={{ img_idx }}&instance_idx={{ (instance_idx - 1) % len(image_instances) }}">Previous Instance</a> |
                <a href="/?img_idx={{ img_idx }}&instance_idx={{ (instance_idx + 1) % len(image_instances) }}">Next Instance</a>
                <br><br>
                <a href="/?img_idx={{ (img_idx - 1) % image_count }}&instance_idx=0">Previous Image</a> |
                <a href="/?img_idx={{ (img_idx + 1) % image_count }}&instance_idx=0">Next Image</a>
            </div>
        </div>
    </body>
    </html>
    """, img_idx=img_idx, instance_idx=instance_idx, 
        current_instance=current_instance, 
        image_instances=image_instances,
        image_count=len(IMAGE_PATHS),
        len=len,
        labeled_count=labeled_count,
        existing_expression=existing_expression,
        image_filename=image_filename,
        status=request.args.get('status', ''))

@app.route('/image/<int:img_idx>/<int:instance_idx>')
def image(img_idx, instance_idx):
    # Get image and instance
    current_image_path = IMAGE_PATHS[img_idx % len(IMAGE_PATHS)]
    image_instances = IMAGE_INSTANCES[current_image_path]
    current_instance = image_instances[instance_idx % len(image_instances)]
    
    # Read the image
    img = cv2.imread(current_instance['image_path'])
    if img is None:
        return "Image not found", 404
    
    # Draw bounding box
    x1, y1, x2, y2 = current_instance['bbox']
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
    
    # Encode and return
    _, buffer = cv2.imencode('.png', img)
    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype='image/png'
    )

@app.route('/save', methods=['POST'])
def save():
    # Get form data
    img_idx = int(request.form.get('img_idx', 0))
    instance_idx = int(request.form.get('instance_idx', 0))
    expression = request.form.get('expression', '').strip()
    existing_image_filename = request.form.get('image_filename', '')
    
    # Validate input
    if not expression:
        return redirect(url_for('index', img_idx=img_idx, instance_idx=instance_idx, 
                               status="Error: Expression cannot be empty"))
    
    # Get current image and instance
    current_image_path = IMAGE_PATHS[img_idx % len(IMAGE_PATHS)]
    image_instances = IMAGE_INSTANCES[current_image_path]
    current_instance = image_instances[instance_idx % len(image_instances)]
    
    # Create a unique instance identifier
    instance_id = f"{current_instance['image_filename']}_{current_instance['category']}_{current_instance['obj_id']}"
    
    # Make sure we have the required structure in our data
    if 'instance_map' not in TRAINING_DATA:
        TRAINING_DATA['instance_map'] = {}
    
    # Check if we're updating an existing entry or creating a new one
    is_update = instance_id in TRAINING_DATA.get('instance_map', {})
    
    # Read the image for either update or new entry
    img = cv2.imread(current_instance['image_path'])
    if img is None:
        return redirect(url_for('index', img_idx=img_idx, instance_idx=instance_idx, 
                               status=f"Error: Could not read image {current_instance['image_path']}"))
    
    # Draw bounding box on the full image
    x1, y1, x2, y2 = current_instance['bbox']
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
    
    if is_update:
        # We're updating an existing entry, use the same filename and example_id
        metadata_entry = TRAINING_DATA['instance_map'][instance_id]
        image_filename = metadata_entry.get('image_filename', '')
        example_id = metadata_entry.get('example_id', '')
        
        # If for some reason we lost the image filename, generate a new one
        if not image_filename:
            image_filename = generate_image_filename(current_instance)
            metadata_entry['image_filename'] = image_filename
        
        # If for some reason we lost the example_id, generate a new one
        if not example_id:
            example_id = str(uuid.uuid4())
            metadata_entry['example_id'] = example_id
    else:
        # Generate a new image filename and example ID
        image_filename = generate_image_filename(current_instance)
        example_id = str(uuid.uuid4())
        
        # Create an entry in our instance map
        TRAINING_DATA['instance_map'][instance_id] = {
            'image_filename': image_filename,
            'example_id': example_id,
            'original_image': current_instance['image_filename'],
            'category': current_instance['category'],
            'obj_id': current_instance['obj_id'],
            'bbox': current_instance['bbox']
        }
    
    # Always save the image (for both new and update cases)
    image_path = os.path.join(OUTPUT_IMAGES_DIR, image_filename)
    success = cv2.imwrite(image_path, img)
    if not success:
        print(f"Warning: Failed to save image to {image_path}")
    else:
        print(f"Saved image to {image_path}")
    
    # Create detailed prompt for referring expression generation
    system_prompt = (
        "You are an AI assistant that creates detailed referring expressions for objects in aerial imagery.\n\n"
        "Given an aerial image with a highlighted object (in a red bounding box), create a referring expression that:\n"
        "1. Identifies the specific object with high precision\n"
        "2. Describes observable visual features (shape, color, size, orientation)\n"
        "3. Includes definite spatial relationships to nearby visible objects\n"
        "4. Notes distinctive identifying characteristics to help locate this object\n"
        "5. Mentions relevant contextual elements like roads, buildings, or terrain features\n\n"
        "Focus only on factual, observable details without speculation. Ensure your description uniquely identifies "
        "this specific instance among similar objects."
    )
    
    user_prompt = f"Write a detailed, precise referring expression that uniquely identifies this {current_instance['category']} highlighted with a red bounding box."
    
    # Create or update the example in the JSONL format
    example = {
        "systemInstruction": {
            "role": "system",
            "parts": [
                {
                    "text": system_prompt
                }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "fileData": {
                            "mimeType": "image/jpeg",
                            "fileUri": f"gs://cloud-samples-data/gemini/{image_filename}"
                        }
                    },
                    {
                        "text": user_prompt
                    }
                ]
            },
            {
                "role": "model",
                "parts": [
                    {
                        "text": expression
                    }
                ]
            }
        ],
        # Add metadata fields (these won't be included in the JSONL output)
        "metadata": {
            "instance_id": instance_id,
            "timestamp": datetime.now().isoformat(),
            "image_filename": image_filename
        }
    }
    
    # Add or update the example in our training data
    TRAINING_DATA["examples"][example_id] = example
    
    # Save the updated training data
    try:
        save_training_data(TRAINING_DATA)
        status_message = f"{'Updated' if is_update else 'Saved'}! ({len(TRAINING_DATA['examples'])} examples in dataset)"
    except Exception as e:
        status_message = f"Error saving data: {str(e)}"
        print(status_message)
    
    # Calculate next instance
    next_instance_idx = (instance_idx + 1) % len(image_instances)
    if next_instance_idx == 0:  # If we've gone through all instances in this image, move to next image
        next_img_idx = (img_idx + 1) % len(IMAGE_PATHS)
    else:
        next_img_idx = img_idx
    
    # Redirect to next instance with success message
    return redirect(url_for('index', img_idx=next_img_idx, instance_idx=next_instance_idx,
                           status=status_message))

# Add a new route to handle deletion
@app.route('/delete')
def delete_instance():
    # Get parameters
    img_idx = int(request.args.get('img_idx', 0))
    instance_idx = int(request.args.get('instance_idx', 0))
    
    # Get current image and instance
    current_image_path = IMAGE_PATHS[img_idx % len(IMAGE_PATHS)]
    image_instances = IMAGE_INSTANCES[current_image_path]
    current_instance = image_instances[instance_idx % len(image_instances)]
    
    # Create a unique instance identifier
    instance_id = f"{current_instance['image_filename']}_{current_instance['category']}_{current_instance['obj_id']}"
    
    # Check if this instance exists in our data
    if instance_id in TRAINING_DATA.get('instance_map', {}):
        # Get the example ID to delete the example
        metadata_entry = TRAINING_DATA['instance_map'][instance_id]
        example_id = metadata_entry.get('example_id', '')
        image_filename = metadata_entry.get('image_filename', '')
        
        # Delete from instance map
        del TRAINING_DATA['instance_map'][instance_id]
        
        # Delete from examples if example_id exists
        if example_id and example_id in TRAINING_DATA['examples']:
            del TRAINING_DATA['examples'][example_id]
        
        # Delete the image file if it exists
        image_path = os.path.join(OUTPUT_IMAGES_DIR, image_filename)
        if os.path.exists(image_path):
            try:
                os.remove(image_path)
                print(f"Deleted image file: {image_path}")
            except Exception as e:
                print(f"Error deleting image file {image_path}: {e}")
        
        # Save the updated data
        try:
            save_training_data(TRAINING_DATA)
            status = f"Instance deleted! ({len(TRAINING_DATA['examples'])} examples remaining)"
        except Exception as e:
            status = f"Error saving data after delete: {str(e)}"
            print(status)
    else:
        status = "Instance not found in dataset"
    
    # Calculate next instance
    next_instance_idx = instance_idx % len(image_instances)
    
    # Redirect back to the page
    return redirect(url_for('index', img_idx=img_idx, instance_idx=next_instance_idx, status=status))

if __name__ == '__main__':
    app.run(debug=True, port=5001)  # Use a different port than the viewing app 