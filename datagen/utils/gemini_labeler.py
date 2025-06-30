#!/usr/bin/env python3
import os
import io
import json
import random
import uuid
import base64
import xml.etree.ElementTree as ET
from datetime import datetime
from flask import Flask, render_template_string, send_file, request, redirect, url_for, jsonify
import cv2
import numpy as np
from google import genai
from pydantic import BaseModel
from typing import List, Optional, Dict
from google.genai import types
from google.cloud import storage

app = Flask(__name__)

# Gemini API configuration
PROJECT_ID = "564504826453"  # Your project ID
LOCATION = "us-central1"
MODEL_ID = "gemini-2.0-flash-001"

# Google Cloud Storage configuration
BUCKET_NAME = "aerial-bucket"
GCS_PATH_PREFIX = "gemini_finetune/images"

# Paths for input data
ANNOTATIONS_DIR = "dataset/patches_rules/annotations"  # Using the rules version without expressions
IMAGES_DIR = "dataset/patches/images"

# Paths for output data
OUTPUT_DIR = "gemini_finetuning_data"
OUTPUT_IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
OUTPUT_JSON_PATH = os.path.join(OUTPUT_DIR, "training_data.jsonl")
METADATA_JSON_PATH = os.path.join(OUTPUT_DIR, "metadata.json")
CONVERSATION_DIR = os.path.join(OUTPUT_DIR, "conversations")

# Ensure output directories exist
os.makedirs(OUTPUT_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CONVERSATION_DIR, exist_ok=True)

# Session storage for conversations
ACTIVE_CONVERSATIONS = {}

class GeneratedResponse(BaseModel):
    description: str
    expression: str

def get_client():
    """Initialize Google generative AI client"""
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )

def get_storage_client():
    """Initialize Google Cloud Storage client"""
    return storage.Client(project=PROJECT_ID)

def upload_image_to_gcs(local_path, blob_name):
    """Upload an image to Google Cloud Storage"""
    try:
        storage_client = get_storage_client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        return f"gs://{BUCKET_NAME}/{blob_name}"
    except Exception as e:
        print(f"Error uploading {local_path}: {e}")
        return None

def load_instances():
    """Load all instances from the dataset"""
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
                'image_filename': image_filename,
                'xml_file': xml_file
            })
    
    return all_instances

def load_training_data():
    """Load existing training data if available"""
    data = {"examples": {}}
    
    # Load the metadata file
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

def save_training_data(data):
    """Save training data to metadata and JSONL files"""
    # Create a backup of the old metadata file if it exists
    if os.path.exists(METADATA_JSON_PATH):
        backup_path = f"{METADATA_JSON_PATH}.bak.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.rename(METADATA_JSON_PATH, backup_path)
    
    # Save metadata
    with open(METADATA_JSON_PATH, 'w') as f:
        json.dump(data, f, indent=2)
    
    # Write all examples to the JSONL file
    with open(OUTPUT_JSON_PATH, 'w') as f:
        for example_data in data["examples"].values():
            # Create the JSONL format
            jsonl_entry = {
                "systemInstruction": example_data["systemInstruction"],
                "contents": example_data["contents"]
            }
            f.write(json.dumps(jsonl_entry) + '\n')

def generate_image_filename(instance):
    """Generate a unique filename for the saved image"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    category = instance['category'].replace(' ', '_')
    return f"{category}_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"

def process_image_with_bbox(image_path, bbox):
    """Process image by adding a red bounding box"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    # Draw a red rectangle around the target instance
    x1, y1, x2, y2 = bbox
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box
    
    return image

def generate_referring_expression(client, image, category, conversation_history=None, feedback=None):
    """Generate a referring expression using Gemini AI with structured output"""
    try:
        # Convert image to bytes for API
        _, img_encoded = cv2.imencode('.jpg', image)
        image_bytes = img_encoded.tobytes()
        
        # Create the prompt for generating a referring expression
        instructions = (
            "You are an AI assistant that creates detailed referring expressions for objects in aerial imagery.\n\n"
            "Given an aerial image with a highlighted object (in a red bounding box), you will provide:\n"
            "1. A detailed reasoning about what you see in the image and how you'll create a unique referring expression\n"
            "2. A final referring expression that uniquely identifies the highlighted object\n\n"
            "Your referring expression should:\n"
            "- Identify the specific object with high precision\n"
            "- Describe observable visual features (shape, color, size, orientation)\n"
            "- Include nearby visible objects\n"
            "- Note distinctive identifying characteristics\n"
            "- Mention relevant contextual elements like roads, buildings, or terrain features\n\n"
            "Focus only on factual, observable details without speculation. Ensure your description uniquely identifies "
            "this specific instance among similar objects in the scene. "
            "The object must be identified without resorting to relative positioning within the frame. Instead, the focus should be on the object's position relative to other objects in the scene."
        )
        
        # Combine instructions with previous feedback if available
        prompt = instructions + "\n\n"
        
        if feedback:
            prompt += f"Based on previous feedback: {feedback}\n\n"
        
        # Add conversation context if available
        if conversation_history:
            for turn in conversation_history:
                if turn['role'] == 'user' and turn != conversation_history[-1]:  # Skip the last user turn as we'll add it separately
                    prompt += f"Previous feedback: {turn['content']}\n"
                elif turn['role'] == 'model':
                    try:
                        content = json.loads(turn['content'])
                        prompt += f"Previous expression: {content.get('expression', '')}\n"
                    except:
                        pass
            prompt += "\n"
        
        user_prompt = f"Write a detailed, precise referring expression that uniquely identifies this {category} highlighted with a red bounding box."
        prompt += user_prompt
        
        # Define the expected JSON output schema
        response_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                'description': types.Schema(
                    type=types.Type.STRING,
                    description="Detailed reasoning about what is observed and how the expression is formulated"
                ),
                'expression': types.Schema(
                    type=types.Type.STRING,
                    description="The final referring expression that uniquely identifies the object"
                )
            },
            required=['description', 'expression']
        )
        
        # Create the content with only user role
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    )
                ]
            )
        ]
        
        # Create generation config
        generation_config_object = types.GenerateContentConfig(
            temperature=0.2,
            top_p=0.95,
            max_output_tokens=1024,
            response_mime_type="application/json",
            response_schema=response_schema
        )
        
        # Call the Gemini API
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=contents,
            config=generation_config_object
        )
        
        # Parse the JSON response
        try:
            response_json = json.loads(response.text)
            return GeneratedResponse(**response_json)
        except json.JSONDecodeError:
            print(f"Error parsing JSON response: {response.text}")
            # Fallback if the response is not valid JSON
            return GeneratedResponse(
                description="Error: Could not parse structured output",
                expression=response.text
            )
        
    except Exception as e:
        print(f"Error generating referring expression: {e}")
        return GeneratedResponse(
            description=f"Error generating expression: {str(e)}",
            expression="An error occurred. Please try regenerating."
        )

def get_conversation(instance_id):
    """Get or create a conversation for an instance"""
    if instance_id not in ACTIVE_CONVERSATIONS:
        ACTIVE_CONVERSATIONS[instance_id] = []
    return ACTIVE_CONVERSATIONS[instance_id]

def parse_conversation(conversation):
    """Parse the JSON content in conversation messages"""
    parsed_conversation = []
    for turn in conversation:
        parsed_turn = turn.copy()
        if turn['role'] == 'model':
            try:
                # Try to parse the JSON content
                content_json = json.loads(turn['content'])
                parsed_turn['parsed_content'] = content_json
                parsed_turn['is_json'] = True
            except json.JSONDecodeError:
                # If it's not valid JSON, keep the original content
                parsed_turn['is_json'] = False
        else:
            parsed_turn['is_json'] = False
        parsed_conversation.append(parsed_turn)
    return parsed_conversation

def save_conversation(instance_id, conversation):
    """Save conversation to file"""
    conversation_path = os.path.join(CONVERSATION_DIR, f"{instance_id}.json")
    with open(conversation_path, 'w') as f:
        json.dump(conversation, f, indent=2)

# Load instances and training data
INSTANCES = load_instances()
TRAINING_DATA = load_training_data()

@app.route('/')
def index():
    # Get a random instance or use the one from the query parameters
    instance_idx = request.args.get('instance_idx')
    
    if instance_idx is None:
        # Pick a random instance
        current_instance = random.choice(INSTANCES)
        # Find its index
        instance_idx = INSTANCES.index(current_instance)
    else:
        # Use the specified instance
        instance_idx = int(instance_idx) % len(INSTANCES)
        current_instance = INSTANCES[instance_idx]
    
    # Create a unique instance identifier
    instance_id = f"{current_instance['image_filename']}_{current_instance['category']}_{current_instance['obj_id']}"
    
    # Get or initialize conversation history
    conversation = get_conversation(instance_id)
    
    # Parse the conversation for the template
    parsed_conversation = parse_conversation(conversation)
    
    # Check if we already have a saved expression
    saved_expression = ""
    image_filename = ""
    is_completed = False
    
    # Check if this instance is in our metadata
    if instance_id in TRAINING_DATA.get('instance_map', {}):
        metadata_entry = TRAINING_DATA['instance_map'][instance_id]
        image_filename = metadata_entry.get('image_filename', '')
        example_id = metadata_entry.get('example_id', '')
        is_completed = True
        
        # Get the expression if it exists
        if example_id in TRAINING_DATA.get('examples', {}):
            example = TRAINING_DATA['examples'][example_id]
            for content in example.get('contents', []):
                if content.get('role') == 'model':
                    for part in content.get('parts', []):
                        if 'text' in part:
                            saved_expression = part['text']
                            break
                    if saved_expression:
                        break
    
    # Generate initial expression if conversation is empty and not completed
    latest_description = ""
    latest_expression = ""
    
    if not conversation and not is_completed:
        # Process the image with bounding box
        bbox_image = process_image_with_bbox(current_instance['image_path'], current_instance['bbox'])
        
        # Generate a referring expression using Gemini
        client = get_client()
        generated_response = generate_referring_expression(client, bbox_image, current_instance['category'])
        
        # Add to conversation
        conversation.append({
            'role': 'model',
            'content': json.dumps({
                'description': generated_response.description,
                'expression': generated_response.expression
            }),
            'timestamp': datetime.now().isoformat()
        })
        
        # Save conversation
        ACTIVE_CONVERSATIONS[instance_id] = conversation
        save_conversation(instance_id, conversation)
        
        # Re-parse for the template
        parsed_conversation = parse_conversation(conversation)
        
        latest_description = generated_response.description
        latest_expression = generated_response.expression
    elif conversation:
        # Get the latest model response
        for turn in reversed(conversation):
            if turn['role'] == 'model':
                try:
                    content = json.loads(turn['content'])
                    latest_description = content.get('description', '')
                    latest_expression = content.get('expression', '')
                    break
                except:
                    pass
    
    # Count labeled examples for progress display
    labeled_count = len(TRAINING_DATA.get("examples", {}))
    
    # Render template
    template = """
    <html>
    <head>
        <title>Gemini Conversational Labeling Tool</title>
        <style>
            body { font-family: sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 1200px; margin: 0 auto; background-color: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            .image-container { position: relative; display: inline-block; margin: 20px auto; text-align: center; width: 100%; }
            img { max-width: 90%; max-height: 500px; border: 1px solid #ccc; border-radius: 4px; }
            .nav { margin-top: 20px; }
            .progress { margin: 20px 0; text-align: left; }
            textarea { width: 100%; height: 120px; margin: 10px 0; padding: 8px; font-size: 14px; border-radius: 4px; border: 1px solid #ccc; }
            .info-box { background-color: #f9f9f9; border: 1px solid #ddd; 
                      padding: 15px; margin: 10px 0; text-align: left; border-radius: 4px; }
            button, .btn {
                padding: 10px 20px; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer; 
                font-size: 16px; 
                margin-top: 10px;
                text-decoration: none;
                display: inline-block;
            }
            .save-btn { 
                background-color: #4CAF50; 
                color: white; 
            }
            .save-btn:hover { background-color: #45a049; }
            .feedback-btn {
                background-color: #4a89dc;
                color: white;
            }
            .feedback-btn:hover { background-color: #3a70bc; }
            .skip-btn {
                background-color: #7f8c8d;
                color: white;
                margin-left: 10px;
            }
            .skip-btn:hover { background-color: #6c7879; }
            .button-group {
                display: flex;
                margin-top: 10px;
                justify-content: center;
            }
            .prompt-container { margin-top: 20px; text-align: left; }
            .status { margin-top: 10px; color: #4CAF50; font-weight: bold; }
            .section {
                background-color: #fff;
                border: 1px solid #ddd;
                padding: 15px;
                margin-bottom: 15px;
                border-radius: 4px;
            }
            .reasoning-section {
                background-color: #f0f7ff;
                border: 1px solid #c0d9ff;
            }
            .expression-section {
                background-color: #f0fff0;
                border: 1px solid #c0ffc0;
            }
            .section-title {
                font-weight: bold;
                margin-bottom: 10px;
                color: #2c3e50;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .conversation-history {
                max-height: 300px;
                overflow-y: auto;
                margin-top: 20px;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 10px;
            }
            .conversation-turn {
                margin-bottom: 15px;
                padding-bottom: 15px;
                border-bottom: 1px solid #eee;
            }
            .conversation-turn:last-child {
                border-bottom: none;
            }
            .user-turn {
                background-color: #f5f5f5;
                padding: 10px;
                border-radius: 4px;
                margin-bottom: 10px;
            }
            .model-turn {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            .message-header {
                font-weight: bold;
                margin-bottom: 5px;
            }
            .timestamp {
                font-size: 12px;
                color: #999;
                margin-left: 10px;
            }
            .header {
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .collapse-btn {
                background: none;
                border: none;
                color: #4a89dc;
                cursor: pointer;
                font-size: 14px;
                padding: 0;
                margin: 0;
            }
            h1 {
                color: #2c3e50;
                margin-top: 0;
            }
            .tabs {
                display: flex;
                margin-bottom: 20px;
                border-bottom: 1px solid #ddd;
            }
            .tab {
                padding: 10px 20px;
                cursor: pointer;
                border: 1px solid transparent;
                border-bottom: none;
                margin-right: 5px;
                border-radius: 4px 4px 0 0;
            }
            .tab.active {
                background-color: white;
                border-color: #ddd;
                border-bottom: 1px solid white;
                margin-bottom: -1px;
                font-weight: bold;
            }
            .tab-content {
                display: none;
            }
            .tab-content.active {
                display: block;
            }
            .completed-badge {
                background-color: #4CAF50;
                color: white;
                padding: 3px 8px;
                border-radius: 12px;
                font-size: 12px;
                margin-left: 10px;
            }
        </style>
        <script>
            function toggleSection(id) {
                const content = document.getElementById(id);
                if (content.style.display === 'none') {
                    content.style.display = 'block';
                    document.getElementById(id + '-toggle').innerHTML = 'Hide';
                } else {
                    content.style.display = 'none';
                    document.getElementById(id + '-toggle').innerHTML = 'Show';
                }
            }
            
            function switchTab(tabId) {
                // Hide all tab contents
                const tabContents = document.querySelectorAll('.tab-content');
                tabContents.forEach(content => {
                    content.classList.remove('active');
                });
                
                // Deactivate all tabs
                const tabs = document.querySelectorAll('.tab');
                tabs.forEach(tab => {
                    tab.classList.remove('active');
                });
                
                // Activate selected tab and content
                document.getElementById('tab-' + tabId).classList.add('active');
                document.getElementById(tabId).classList.add('active');
            }
        </script>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Gemini Conversational Labeling Tool</h1>
                <div class="progress">
                    Progress: {{ labeled_count }} labeled examples
                </div>
            </div>
            
            <div class="info-box">
                <p><strong>Instance:</strong> {{ instance_idx + 1 }}/{{ instances_count }} 
                {% if is_completed %}<span class="completed-badge">Completed</span>{% endif %}</p>
                <p><strong>Category:</strong> {{ current_instance['category'] }}</p>
                <p><strong>Position:</strong> {{ current_instance['position'] if current_instance['position'] else 'Not specified' }}</p>
                <p><strong>Image filename:</strong> {{ current_instance['image_filename'] }}</p>
            </div>
            
            <div class="tabs">
                <div id="tab-current" class="tab active" onclick="switchTab('current')">Current State</div>
                <div id="tab-history" class="tab" onclick="switchTab('history')">Conversation History</div>
            </div>
            
            <div id="current" class="tab-content active">
                <div class="image-container">
                    <img src="/image/{{ instance_idx }}">
                </div>
                
                {% if latest_description or latest_expression %}
                <div class="section reasoning-section">
                    <div class="section-title">
                        Model's Reasoning
                        <button id="reasoning-toggle" class="collapse-btn" onclick="toggleSection('reasoning-content')">Hide</button>
                    </div>
                    <div id="reasoning-content">
                        {{ latest_description }}
                    </div>
                </div>
                
                <div class="section expression-section">
                    <div class="section-title">Generated Expression</div>
                    <div>{{ latest_expression }}</div>
                </div>
                {% endif %}
                
                {% if saved_expression %}
                <div class="section">
                    <div class="section-title">Saved Expression</div>
                    <div>{{ saved_expression }}</div>
                </div>
                {% endif %}
                
                <div class="prompt-container">
                    {% if not is_completed %}
                    <form method="post" action="/feedback">
                        <textarea name="feedback" placeholder="Provide feedback to improve the generated expression. Be specific about what aspects need improvement."></textarea>
                        <input type="hidden" name="instance_idx" value="{{ instance_idx }}">
                        <div class="button-group">
                            <button type="submit" class="btn feedback-btn">Send Feedback & Regenerate</button>
                            <a href="/save?instance_idx={{ instance_idx }}" class="btn save-btn">Accept & Save</a>
                            <a href="/?instance_idx={{ random_instance_idx }}" class="btn skip-btn">Skip</a>
                        </div>
                    </form>
                    {% else %}
                    <div class="button-group">
                        <a href="/?instance_idx={{ random_instance_idx }}" class="btn save-btn">Next Instance</a>
                    </div>
                    {% endif %}
                    
                    {% if status %}
                    <div class="status">{{ status }}</div>
                    {% endif %}
                </div>
            </div>
            
            <div id="history" class="tab-content">
                <div class="conversation-history">
                    {% if conversation %}
                        {% for turn in conversation %}
                            <div class="conversation-turn">
                                {% if turn['role'] == 'user' %}
                                    <div class="user-turn">
                                        <div class="message-header">Your Feedback<span class="timestamp">{{ turn['timestamp'].split('T')[1].split('.')[0] }}</span></div>
                                        <div>{{ turn['content'] }}</div>
                                    </div>
                                {% else %}
                                    <div class="model-turn">
                                        <div class="message-header">Model Response<span class="timestamp">{{ turn['timestamp'].split('T')[1].split('.')[0] }}</span></div>
                                        {% if turn['is_json'] %}
                                            <div class="section reasoning-section">
                                                <div class="section-title">Reasoning</div>
                                                <div>{{ turn['parsed_content']['description'] }}</div>
                                            </div>
                                            <div class="section expression-section">
                                                <div class="section-title">Expression</div>
                                                <div>{{ turn['parsed_content']['expression'] }}</div>
                                            </div>
                                        {% else %}
                                            <div>{{ turn['content'] }}</div>
                                        {% endif %}
                                    </div>
                                {% endif %}
                            </div>
                        {% endfor %}
                    {% else %}
                        <p>No conversation history yet.</p>
                    {% endif %}
                </div>
                
                <div class="button-group" style="margin-top: 20px;">
                    <a href="#" class="btn feedback-btn" onclick="switchTab('current')">Back to Current</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """
    
    return render_template_string(template, 
    instance_idx=instance_idx,
    current_instance=current_instance,
    instances_count=len(INSTANCES),
    latest_description=latest_description,
    latest_expression=latest_expression,
    saved_expression=saved_expression,
    image_filename=image_filename,
    labeled_count=labeled_count,
    conversation=parsed_conversation,
    random_instance_idx=random.randint(0, len(INSTANCES)-1),
    status=request.args.get('status', ''),
    is_completed=is_completed)

@app.route('/image/<int:instance_idx>')
def image(instance_idx):
    """Serve image with bounding box drawn on it"""
    current_instance = INSTANCES[instance_idx % len(INSTANCES)]
    
    # Process the image with bounding box
    bbox_image = process_image_with_bbox(current_instance['image_path'], current_instance['bbox'])
    if bbox_image is None:
        return "Image not found", 404
    
    # Encode and return
    _, buffer = cv2.imencode('.png', bbox_image)
    return send_file(
        io.BytesIO(buffer.tobytes()),
        mimetype='image/png'
    )

@app.route('/feedback', methods=['POST'])
def feedback():
    """Process user feedback and regenerate expression"""
    # Get form data
    instance_idx = int(request.form.get('instance_idx', 0))
    feedback = request.form.get('feedback', '').strip()
    
    # Validate input
    if not feedback:
        return redirect(url_for('index', instance_idx=instance_idx, 
                               status="Error: Feedback cannot be empty"))
    
    # Get current instance
    current_instance = INSTANCES[instance_idx % len(INSTANCES)]
    
    # Create a unique instance identifier
    instance_id = f"{current_instance['image_filename']}_{current_instance['category']}_{current_instance['obj_id']}"
    
    # Get conversation history
    conversation = get_conversation(instance_id)
    
    # Add user feedback to conversation
    conversation.append({
        'role': 'user',
        'content': feedback,
        'timestamp': datetime.now().isoformat()
    })
    
    # Process the image with bounding box
    bbox_image = process_image_with_bbox(current_instance['image_path'], current_instance['bbox'])
    if bbox_image is None:
        return redirect(url_for('index', instance_idx=instance_idx, 
                               status=f"Error: Could not process image {current_instance['image_path']}"))
    
    # Generate a new referring expression with feedback
    client = get_client()
    # Use the parsed conversation for context to the API
    parsed_conversation = parse_conversation(conversation)
    generated_response = generate_referring_expression(
        client, 
        bbox_image, 
        current_instance['category'],
        conversation_history=conversation,
        feedback=feedback
    )
    
    # Add model response to conversation
    conversation.append({
        'role': 'model',
        'content': json.dumps({
            'description': generated_response.description,
            'expression': generated_response.expression
        }),
        'timestamp': datetime.now().isoformat()
    })
    
    # Save updated conversation
    ACTIVE_CONVERSATIONS[instance_id] = conversation
    save_conversation(instance_id, conversation)
    
    # Redirect back to the instance
    return redirect(url_for('index', instance_idx=instance_idx,
                           status="Generated new expression based on your feedback"))

@app.route('/save')
def save():
    """Save the current expression to the training data"""
    # Get instance index
    instance_idx = int(request.args.get('instance_idx', 0))
    
    # Get current instance
    current_instance = INSTANCES[instance_idx % len(INSTANCES)]
    
    # Create a unique instance identifier
    instance_id = f"{current_instance['image_filename']}_{current_instance['category']}_{current_instance['obj_id']}"
    
    # Get conversation history
    conversation = get_conversation(instance_id)
    
    # Extract the latest expression
    latest_expression = None
    for turn in reversed(conversation):
        if turn['role'] == 'model':
            try:
                content = json.loads(turn['content'])
                latest_expression = content.get('expression', '')
                break
            except:
                pass
    
    if not latest_expression:
        return redirect(url_for('index', instance_idx=instance_idx, 
                               status="Error: No expression to save"))
    
    # Make sure we have the required structure in our data
    if 'instance_map' not in TRAINING_DATA:
        TRAINING_DATA['instance_map'] = {}
    
    # Check if we're updating an existing entry or creating a new one
    is_update = instance_id in TRAINING_DATA.get('instance_map', {})
    
    # Process the image with bounding box
    bbox_image = process_image_with_bbox(current_instance['image_path'], current_instance['bbox'])
    if bbox_image is None:
        return redirect(url_for('index', instance_idx=instance_idx, 
                               status=f"Error: Could not process image {current_instance['image_path']}"))
    
    if is_update:
        # We're updating an existing entry, use the same filename and example_id
        metadata_entry = TRAINING_DATA['instance_map'][instance_id]
        image_filename = metadata_entry.get('image_filename', '')
        example_id = metadata_entry.get('example_id', '')
        gcs_uri = metadata_entry.get('gcs_uri', '')
        
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
        gcs_uri = None
    
    # Save the image with bounding box locally first
    local_image_path = os.path.join(OUTPUT_IMAGES_DIR, image_filename)
    success = cv2.imwrite(local_image_path, bbox_image)
    if not success:
        print(f"Warning: Failed to save image to {local_image_path}")
        return redirect(url_for('index', instance_idx=instance_idx, 
                               status=f"Error: Failed to save image to {local_image_path}"))
    
    # Upload the image to Google Cloud Storage
    blob_name = f"{GCS_PATH_PREFIX}/{image_filename}"
    gcs_uri = upload_image_to_gcs(local_image_path, blob_name)
    
    if not gcs_uri:
        return redirect(url_for('index', instance_idx=instance_idx, 
                               status="Error: Failed to upload image to GCS"))
    
    # Create an entry in our instance map if it's a new one
    if not is_update:
        TRAINING_DATA['instance_map'][instance_id] = {
            'image_filename': image_filename,
            'example_id': example_id,
            'gcs_uri': gcs_uri,
            'original_image': current_instance['image_filename'],
            'category': current_instance['category'],
            'obj_id': current_instance['obj_id'],
            'bbox': current_instance['bbox'],
            'conversation_id': instance_id
        }
    else:
        # Update the GCS URI
        TRAINING_DATA['instance_map'][instance_id]['gcs_uri'] = gcs_uri
    
    # Create prompt for referring expression
    system_prompt = (
        "You are an AI assistant that creates detailed referring expressions for objects in aerial imagery.\n\n"
        "Given an aerial image with a highlighted object (in a red bounding box), create a referring expression that:\n"
        "1. Identifies the specific object with high precision\n"
        "2. Describes observable visual features (shape, color, size, orientation)\n"
        "3. Include nearby visible objects\n"
        "4. Notes distinctive identifying characteristics to help locate this object\n"
        "5. Mentions relevant contextual elements like roads, buildings, or terrain features\n\n"
        "Focus only on factual, observable details without speculation. Ensure your description uniquely identifies "
        "this specific instance among similar objects in the scene. "
        "The object must be identified without resorting to relative positioning within the frame. Instead, the focus should be on the object's position relative to other objects in the scene.\n\n"
        "You must respond with a JSON object using the following format:\n"
        "{\n"
        "  \"description\": \"Your detailed reasoning about what you observe in the image and how you formulated the expression\",\n"
        "  \"expression\": \"The final referring expression that uniquely identifies the object\"\n"
        "}\n\n"
    )
    
    user_prompt = f"Write a detailed, precise referring expression that uniquely identifies this {current_instance['category']} highlighted with a red bounding box."
    
    # Create the example in the JSONL format
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
                            "fileUri": gcs_uri
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
                        "text": json.dumps({
                            "description": "Reasoning not saved for final example",
                            "expression": latest_expression
                        })
                    }
                ]
            }
        ],
        # Add metadata fields
        "metadata": {
            "instance_id": instance_id,
            "timestamp": datetime.now().isoformat(),
            "image_filename": image_filename,
            "gcs_uri": gcs_uri,
            "conversation_history": conversation
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
    
    # Pick another random instance for the next one
    next_instance_idx = random.randint(0, len(INSTANCES)-1)
    
    # Redirect to next instance with success message
    return redirect(url_for('index', instance_idx=next_instance_idx,
                           status=status_message))

if __name__ == '__main__':
    app.run(debug=True, port=5001) 