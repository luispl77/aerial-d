import os
import json
import random
from flask import Flask, render_template, request, jsonify, send_file, make_response
import cv2
import base64
from datetime import datetime
from google import genai
from google.genai import types
from pathlib import Path
import xml.etree.ElementTree as ET

app = Flask(__name__)

# Hardcoded values
PROJECT_ID = "564504826453"
LOCATION = "us-central1"
MODEL_ID_LITE = "gemini-2.0-flash-lite-001"
MODEL_ID_FULL = "gemini-2.0-flash-001"
NUM_ENHANCED = 2
NUM_UNIQUE = 2
MAX_RETRIES = 3
MAX_REQUESTS_PER_MINUTE = 200
DELAY = 1.0

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))  # Get the actual project root
DATASET_ROOT = os.path.join(PROJECT_ROOT, "dataset/patches_rules_expressions_unique")
CLASSIFICATION_FILE = os.path.join(PROJECT_ROOT, "utils/classifications.json")
WORKSPACE_ROOT = "/tmp/u035679/aerial_seg_datagen"  # Add workspace root

def get_client():
    """Initialize Google generative AI client with custom model"""
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        http_options=types.HttpOptions(api_version="v1")
    )

# Initialize Gemini clients for both models
client = get_client()

def load_classifications():
    if os.path.exists(CLASSIFICATION_FILE):
        with open(CLASSIFICATION_FILE, 'r') as f:
            return json.load(f)
    return {"good": [], "bad": []}

def save_classifications(classifications):
    with open(CLASSIFICATION_FILE, 'w') as f:
        json.dump(classifications, f, indent=2)

def get_random_sample():
    # Get all XML files in both train and val directories
    train_xml_files = list(Path(DATASET_ROOT).rglob("train/annotations/*.xml"))
    val_xml_files = list(Path(DATASET_ROOT).rglob("val/annotations/*.xml"))
    xml_files = train_xml_files + val_xml_files
    
    print(f"Found {len(xml_files)} XML files")
    
    if not xml_files:
        print("No XML files found")
        return None, None
    
    # Pick a random XML file
    xml_path = random.choice(xml_files)
    print(f"Selected XML file: {xml_path}")
    
    # Determine split (train or val)
    if '/train/' in str(xml_path):
        split = 'train'
    elif '/val/' in str(xml_path):
        split = 'val'
    else:
        print(f"Could not determine split from path: {xml_path}")
        return None, None
    
    # Parse XML to get all objects with expressions
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get all objects that have expressions
    valid_objects = []
    for obj in root.findall('.//object'):
        expressions_elem = obj.find('expressions')
        if expressions_elem is not None:
            expressions = [expr.text for expr in expressions_elem.findall('expression') if expr.text and expr.text.strip()]
            if expressions:  # Only include objects that have at least one expression
                bbox_elem = obj.find('bndbox')
                if bbox_elem is not None:
                    bbox = [
                        int(bbox_elem.find('xmin').text),
                        int(bbox_elem.find('ymin').text),
                        int(bbox_elem.find('xmax').text),
                        int(bbox_elem.find('ymax').text)
                    ]
                    valid_objects.append({
                        'expressions': expressions,
                        'bbox': bbox,
                        'object_name': obj.find('name').text,
                        'object_id': obj.find('id').text
                    })
                    print(f"Found object {obj.find('id').text} with expressions: {expressions} and bbox: {bbox}")  # Debug print
    
    if not valid_objects:
        print(f"No valid objects found in {xml_path}")
        return None, None
    
    # Get corresponding image path in patches/{split}/images
    img_name = Path(xml_path).stem + '.jpg'
    img_path = os.path.join(PROJECT_ROOT, 'dataset/patches', split, 'images', img_name)
    print(f"Looking for image at: {img_path}")
    
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        # Try with .png extension
        img_path = os.path.join(PROJECT_ROOT, 'dataset/patches', split, 'images', Path(xml_path).stem + '.png')
        print(f"Trying with .png extension: {img_path}")
        if not os.path.exists(img_path):
            print(f"Image not found with .png extension either")
            return None, None
    
    print(f"Found image at: {img_path}")
    
    # Randomly select one of the valid objects
    selected_obj = random.choice(valid_objects)
    print(f"Selected object {selected_obj['object_id']} with expressions: {selected_obj['expressions']} and bbox: {selected_obj['bbox']}")
    
    return xml_path, str(img_path), selected_obj

def process_image_for_api(image_path, bbox=None):
    """Process image for API - highlight instance with bounding box if provided"""
    image = cv2.imread(image_path)
    if image is None:
        return None
    
    original_height, original_width = image.shape[:2]
    image = cv2.resize(image, (384, 384))
    
    if bbox:
        xmin, ymin, xmax, ymax = bbox
        scale_x = 384 / original_width
        scale_y = 384 / original_height
        
        xmin = int(xmin * scale_x)
        ymin = int(ymin * scale_y)
        xmax = int(xmax * scale_x)
        ymax = int(ymax * scale_y)
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
    
    return image

def generate_description_and_expressions(image_path, object_name, original_expressions, bbox=None):
    """Generate object description and enhanced expressions using both Gemini models"""
    # Process image if bounding box is provided
    temp_image = None
    try:
        if bbox:
            temp_image = process_image_for_api(image_path, bbox)
            if temp_image is None:
                return None, None, None, "Failed to process image"
        
        # Read image bytes
        if temp_image is not None:
            _, buffer = cv2.imencode('.jpg', temp_image)
            image_bytes = buffer.tobytes()
        else:
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()

        # Format original expressions for the prompt
        formatted_expressions = "\n".join([f"- {expr}" for expr in original_expressions])
        
        # Create the combined prompt for both description and expression enhancement
        prompt = (
            "You are an expert at creating natural language descriptions for objects in aerial imagery. "
            "Your task is to help create diverse and precise referring expressions for objects in the image. "
            "The target object is highlighted with a red bounding box\n\n"
            
            "You have three tasks:\n\n"
            
            f"TASK 1: For each original expression listed below, create EXACTLY {NUM_ENHANCED} language variations that:\n"
            "1. MUST PRESERVE ALL SPATIAL INFORMATION from the original expression:\n"
            "   - Absolute positions (e.g., \"in the top right\", \"near the center\")\n"
            "   - Relative positions (e.g., \"to the right of\", \"below\")\n"
            "2. Use natural, everyday language that a regular person would use\n"
            "   - Avoid overly formal or technical vocabulary\n"
            "   - Use common synonyms (e.g., \"car\" instead of \"automobile\")\n"
            "   - Keep the tone conversational and straightforward\n"
            "3. Ensure each variation uniquely identifies this object to avoid ambiguity\n\n"
            
            "TASK 2: Analyze the object's context and uniqueness factors:\n"
            "1. Examine the immediate surroundings of the object\n"
            "2. Identify distinctive features that could be used to uniquely identify this object:\n"
            "   - Nearby objects and their relationships\n"
            "   - Visual characteristics that distinguish it from similar objects\n"
            "   - Environmental context (roads, buildings, terrain) that provide reference points\n"
            "3. Consider how the original automated expressions could be improved\n"
            "4. Focus on features that would help someone locate this specific object without ambiguity\n\n"
            
            f"TASK 3: Generate EXACTLY {NUM_UNIQUE} new expressions that:\n"
            "1. MUST be based on one of the original expressions or their variations\n"
            "2. Add visual details ONLY when you are highly confident about them\n"
            "3. Each expression must uniquely identify this object\n"
            "4. Focus on describing the object's relationship with its immediate surroundings\n"
            "5. Maintain the core spatial information from the original expression\n\n"
            
            f"ORIGINAL EXPRESSIONS TO ENHANCE:\n{formatted_expressions}\n\n"
            
            "You must return your output in the following JSON format:\n"
            "{\n"
            "  \"enhanced_expressions\": [\n"
            "    {\n"
            "      \"original_expression\": \"<original expression>\",\n"
            "      \"variations\": [\n"
            "        \"<language variation 1>\",\n"
            "        \"<language variation 2>\"\n"
            "      ]\n"
            "    },\n"
            "    ...\n"
            "  ],\n"
            "  \"unique_description\": \"<detailed analysis of spatial context and uniqueness factors>\",\n"
            "  \"unique_expressions\": [\n"
            "    \"<new expression based on original 1>\",\n"
            "    \"<new expression based on original 2>\"\n"
            "  ]\n"
            "}\n"
        )
        
        user_prompt = f"Create language variations of the provided expressions while preserving spatial information, analyze the spatial context for uniqueness factors, and generate new unique expressions for this {object_name} (highlighted with a red bounding box)."
        
        # Create the request content using image bytes
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_text(text=user_prompt),
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/jpeg"
                    )
                ]
            )
        ]
        
        # Define the expected JSON output schema
        response_schema = {
            "type": "OBJECT",
            "properties": {
                "enhanced_expressions": {
                    "type": "ARRAY",
                    "items": {
                        "type": "OBJECT",
                        "properties": {
                            "original_expression": {
                                "type": "STRING"
                            },
                            "variations": {
                                "type": "ARRAY",
                                "minItems": NUM_ENHANCED,
                                "maxItems": NUM_ENHANCED,
                                "items": {
                                    "type": "STRING"
                                }
                            }
                        },
                        "required": ["original_expression", "variations"]
                    }
                },
                "unique_description": {
                    "type": "STRING"
                },
                "unique_expressions": {
                    "type": "ARRAY",
                    "minItems": NUM_UNIQUE,
                    "maxItems": NUM_UNIQUE,
                    "items": {
                        "type": "STRING"
                    }
                }
            },
            "required": ["enhanced_expressions", "unique_description", "unique_expressions"]
        }

        # Create GenerationConfig object including the schema
        generation_config_object = types.GenerateContentConfig(
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=4096,
            response_mime_type="application/json",
            response_schema=response_schema
        )

        # Make API calls to both models
        responses = {}
        for model_id in [MODEL_ID_LITE, MODEL_ID_FULL]:
            try:
                response = client.models.generate_content(
                    model=model_id,
                    contents=contents,
                    config=generation_config_object
                )
                
                # Parse the JSON response
                response_text = response.text
                parsed_output = json.loads(response_text)
                responses[model_id] = parsed_output
            except Exception as e:
                responses[model_id] = {"error": str(e)}
        
        return responses, None
        
    except Exception as e:
        return None, str(e)

@app.route('/')
def index():
    return render_template('classifier.html')

@app.route('/get_sample')
def get_sample():
    result = get_random_sample()
    if not result:
        return jsonify({"error": "No samples found"})
    
    xml_path, img_path, selected_obj = result
    
    # Process image
    image = process_image_for_api(img_path, selected_obj['bbox'])
    _, buffer = cv2.imencode('.jpg', image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Generate enhanced expressions using both models
    responses, error = generate_description_and_expressions(
        img_path, selected_obj['object_name'], selected_obj['expressions'], selected_obj['bbox']
    )
    
    if error:
        return jsonify({"error": f"Failed to generate enhanced expressions: {error}"})
    
    # Process responses from both models
    model_outputs = {}
    for model_id, response in responses.items():
        if "error" in response:
            model_outputs[model_id] = {"error": response["error"]}
            continue
            
        # Combine original and enhanced expressions
        all_expressions = []
        for expr in selected_obj['expressions']:
            all_expressions.append({
                "original": expr,
                "enhanced": next((item['variations'] for item in response['enhanced_expressions'] 
                                if item['original_expression'] == expr), [])
            })
        
        model_outputs[model_id] = {
            "expressions": all_expressions,
            "description": response['unique_description'],
            "unique_expressions": response['unique_expressions']
        }
    
    response_data = {
        "image": img_base64,
        "model_outputs": model_outputs,
        "xml_path": str(xml_path),
        "img_path": str(img_path),
        "object_id": selected_obj['object_id'],  # Add object ID to response
        "bbox": selected_obj['bbox']  # Add bbox to response
    }
    
    return jsonify(response_data)

@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    xml_path = data['xml_path']
    img_path = data['img_path']
    model_outputs = data['model_outputs']
    classification = data['classification']  # 'good' or 'bad'
    object_id = data.get('object_id')  # Get the object ID from the request
    bbox = data.get('bbox')  # Get the bbox from the request
    
    # Get the selected model from the request
    selected_model = data.get('selected_model', 'gemini-2.0-flash-lite-001')
    
    # Get the output for the selected model
    model_output = model_outputs.get(selected_model)
    if not model_output or 'error' in model_output:
        return jsonify({"error": f"No valid output for selected model {selected_model}"})
    
    classifications = load_classifications()
    classifications[classification].append({
        "xml_path": xml_path,
        "img_path": str(img_path),
        "expressions": model_output['expressions'],
        "description": model_output['description'],
        "unique_expressions": model_output['unique_expressions'],
        "model": selected_model,
        "bbox": bbox,
        "object_id": object_id,
        "timestamp": datetime.now().isoformat()
    })
    save_classifications(classifications)
    
    return jsonify({"status": "success"})

@app.route('/get_stats')
def get_stats():
    classifications = load_classifications()
    return jsonify({
        "good_count": len(classifications["good"]),
        "bad_count": len(classifications["bad"])
    })

@app.route('/view_classified')
def view_classified():
    classifications = load_classifications()
    return render_template('view_classified.html', 
                         good_samples=classifications["good"],
                         bad_samples=classifications["bad"])

@app.route('/get_image/<path:path>')
def get_image(path):
    try:
        # Get the bounding box from the classifications
        classifications = load_classifications()
        bbox = None
        object_id = None
        
        # Search through both good and bad samples
        for samples in [classifications["good"], classifications["bad"]]:
            for sample in samples:
                # Compare the image paths, handling both absolute and relative paths
                sample_path = sample["img_path"]
                if (sample_path == path or 
                    os.path.basename(sample_path) == os.path.basename(path) or
                    os.path.normpath(sample_path) == os.path.normpath(path)):
                    bbox = sample.get("bbox")
                    object_id = sample.get("object_id")
                    print(f"Found bbox {bbox} for object {object_id} in image: {path}")  # Debug print
                    break
            if bbox:
                break
        
        # Try different path combinations
        img_paths = [
            path,  # Try the path as is
            os.path.join(PROJECT_ROOT, path),  # Try with project root
            os.path.join(WORKSPACE_ROOT, path)  # Try with workspace root
        ]
        
        # If path contains 'patches', try to construct the correct path
        if 'patches' in path:
            path_parts = path.split('patches/')
            if len(path_parts) > 1:
                img_paths.append(os.path.join(PROJECT_ROOT, 'dataset/patches', path_parts[1]))
        
        # Try each path until we find the image
        img = None
        found_path = None
        for img_path in img_paths:
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    found_path = img_path
                    break
        
        if img is None:
            print(f"Image not found. Tried paths: {img_paths}")
            return "Image not found", 404
            
        # Draw bounding box if available
        if bbox:
            try:
                xmin, ymin, xmax, ymax = bbox
                # Ensure coordinates are within image bounds
                height, width = img.shape[:2]
                xmin = max(0, min(xmin, width-1))
                ymin = max(0, min(ymin, height-1))
                xmax = max(0, min(xmax, width-1))
                ymax = max(0, min(ymax, height-1))
                
                # Draw the bounding box
                cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)
                print(f"Drew bbox at: ({xmin}, {ymin}), ({xmax}, {ymax}) for object {object_id}")  # Debug print
            except Exception as e:
                print(f"Error drawing bbox: {e}")
        else:
            print(f"No bbox found for image: {found_path}")  # Debug print
        
        # Convert to JPEG and return
        _, buffer = cv2.imencode('.jpg', img)
        response = make_response(buffer.tobytes())
        response.headers['Content-Type'] = 'image/jpeg'
        return response
        
    except Exception as e:
        print(f"Error serving image: {e}")
        return "Error serving image", 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('utils/templates', exist_ok=True)
    app.run(debug=True, port=5001) 