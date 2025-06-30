#!/usr/bin/env python3
import os
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
from PIL import Image
from tqdm import tqdm
import re
import multiprocessing
import time
import queue
import datetime
import base64
from google import genai
from google.genai import types
from pydantic import BaseModel
from typing import List, Optional
import cv2
from collections import deque
import threading
import json
from multiprocessing import Value, Array
import ctypes
import numpy as np

# Set multiprocessing start method to 'spawn'
multiprocessing.set_start_method('spawn', force=True)

# Hardcoded values
PROJECT_ID = "564504826453"
LOCATION = "us-central1"
# MODEL_ID = "projects/564504826453/locations/us-central1/endpoints/1416053328831315968"
#MODEL_ID = "projects/564504826453/locations/us-central1/endpoints/1374957982231560192"
MODEL_ID = "gemini-2.0-flash-lite-001"
NUM_ENHANCED = 3
MAX_RETRIES = 3
MAX_REQUESTS_PER_MINUTE = 200
DELAY = 1.0
NUM_WORKERS = 4
VERBOSE = False
DEBUG = False

# Hardcoded paths
DATASET_FOLDER = "dataset"
PATCHES_FOLDER = os.path.join(DATASET_FOLDER, "patches", "images")
OUTPUT_DIR = "dataset/patches_rules_expressions_unique_llm_custom"
ANNOTATIONS_DIR = "dataset/patches_rules_expressions_unique/annotations"

class EnhancedExpression(BaseModel):
    original_expression: str
    enhanced_versions: List[str]

class CombinedOutput(BaseModel):
    description: str
    enhanced_expressions: List[EnhancedExpression]

class RateLimiter:
    """Simple rate limiter to ensure we don't exceed specified requests per minute"""
    def __init__(self, max_requests_per_minute):
        self.max_requests = max_requests_per_minute
        self.request_times = deque()
        self.lock = threading.Lock()
        
    def wait_if_needed(self):
        """Wait if needed to comply with rate limits"""
        with self.lock:
            now = time.time()
            
            # Remove timestamps older than a minute
            while self.request_times and now - self.request_times[0] > 60:
                self.request_times.popleft()
            
            # If we've reached the limit, wait
            if len(self.request_times) >= self.max_requests:
                # Calculate how long to wait
                wait_time = 60 - (now - self.request_times[0])
                if wait_time > 0:
                    time.sleep(wait_time)
                    now = time.time()  # Update now after sleeping
            
            # Record this request
            self.request_times.append(now)

class Stats:
    """Class to track and calculate statistics across processes"""
    def __init__(self):
        # These will be shared across processes
        self.num_instances = Value(ctypes.c_int, 0)
        self.num_api_requests = Value(ctypes.c_int, 0)
        self.num_enhanced_expressions = Value(ctypes.c_int, 0)
        self.num_original_expressions = Value(ctypes.c_int, 0)
        
        # For token tracking
        self.total_input_tokens = Value(ctypes.c_int, 0)
        self.total_output_tokens = Value(ctypes.c_int, 0)
        
        # For timing
        self.total_api_time = Value(ctypes.c_float, 0.0)

def log_with_timestamp(message, worker_id=None):
    """Add timestamp to log messages"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    worker_prefix = f"Worker {worker_id}: " if worker_id is not None else ""
    print(f"[{timestamp}] {worker_prefix}{message}")

def get_client():
    """Initialize Google generative AI client with custom model"""
    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
    )

def encode_image(image_path):
    """Encode image to base64 for API request"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_for_api(image_path, bbox=None):
    """Process image for API - highlight instance with bounding box if provided"""
    if bbox:
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        xmin, ymin, xmax, ymax = bbox
        
        # Draw a red rectangle around the target instance
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255), 1)
        
        # Save temp image
        temp_path = f"temp_{os.path.basename(image_path)}"
        cv2.imwrite(temp_path, image)
        return temp_path
    else:
        return image_path

def save_debug_output(output_dir, image_filename, obj_id, worker_id, raw_response):
    """Save raw API response to a debug file"""
    debug_dir = os.path.join(output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Create a filename with timestamp to avoid conflicts
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_filename = f"{image_filename}_obj_{obj_id}_worker_{worker_id}_{timestamp}.json"
    debug_path = os.path.join(debug_dir, debug_filename)
    
    # Write the raw response
    with open(debug_path, 'w', encoding='utf-8') as f:
        import json
        # Try to serialize the response if it's an object, otherwise convert to string
        try:
            json.dump(raw_response, f, indent=2)
        except (TypeError, ValueError):
            f.write(str(raw_response))
    
    return debug_path

def estimate_tokens(text):
    """Estimate the number of tokens in a text string"""
    if not text:
        return 0
    
    try:
        # Try to use tiktoken for accurate token counting
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")  # OpenAI's encoding
        return len(encoding.encode(text))
    except (ImportError, Exception):
        # Fall back to simple approximation if tiktoken isn't available
        # Simple approximation: ~4 characters per token on average
        return len(text) // 4

def update_stats(stats, worker_stats):
    """Update global stats with worker stats"""
    with stats.num_instances.get_lock():
        stats.num_instances.value += worker_stats.get('instances', 0)
        stats.num_api_requests.value += worker_stats.get('api_requests', 0)
        stats.num_enhanced_expressions.value += worker_stats.get('enhanced_expressions', 0)
        stats.num_original_expressions.value += worker_stats.get('original_expressions', 0)
        stats.total_input_tokens.value += worker_stats.get('input_tokens', 0)
        stats.total_output_tokens.value += worker_stats.get('output_tokens', 0)
        stats.total_api_time.value += worker_stats.get('api_time', 0.0)

def save_stats_to_file(stats, output_dir):
    """Calculate final statistics and write to file"""
    stats_file = os.path.join(output_dir, "generation_stats.txt")
    
    # Calculate derived statistics
    total_expressions = stats.num_original_expressions.value + stats.num_enhanced_expressions.value
    
    avg_input_tokens = 0
    if stats.num_api_requests.value > 0:
        avg_input_tokens = stats.total_input_tokens.value / stats.num_api_requests.value
    
    avg_output_tokens = 0
    if stats.num_api_requests.value > 0:
        avg_output_tokens = stats.total_output_tokens.value / stats.num_api_requests.value
    
    avg_total_tokens = avg_input_tokens + avg_output_tokens
    total_tokens = stats.total_input_tokens.value + stats.total_output_tokens.value
    
    # Calculate cost estimates
    input_cost = (stats.total_input_tokens.value / 1_000_000) * 0.10  # $0.10 per million input tokens
    output_cost = (stats.total_output_tokens.value / 1_000_000) * 0.40  # $0.40 per million output tokens
    total_cost = input_cost + output_cost
    
    avg_api_time = 0
    if stats.num_api_requests.value > 0:
        avg_api_time = stats.total_api_time.value / stats.num_api_requests.value
    
    # Format the statistics report
    report = f"""
GENERATION STATISTICS
=====================
Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

COUNTS
------
Total Instances Processed: {stats.num_instances.value}
Total API Requests: {stats.num_api_requests.value}
Total Original Expressions: {stats.num_original_expressions.value}
Total Enhanced Expressions: {stats.num_enhanced_expressions.value}
Combined Total Expressions: {total_expressions}

TOKEN USAGE
-----------
Average Input Tokens per Request: {avg_input_tokens:.2f}
Average Output Tokens per Request: {avg_output_tokens:.2f}
Average Total Tokens per Request: {avg_total_tokens:.2f}
Total Input Tokens: {stats.total_input_tokens.value}
Total Output Tokens: {stats.total_output_tokens.value}
Total Tokens: {total_tokens}

TIMING
------
Average API Request Time: {avg_api_time:.2f} seconds
Total API Request Time: {stats.total_api_time.value:.2f} seconds

COST ESTIMATE
------------
Input Cost (at $0.10 per million tokens): ${input_cost:.4f}
Output Cost (at $0.40 per million tokens): ${output_cost:.4f}
Total Estimated Cost: ${total_cost:.4f}
"""

    # Write to file
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    log_with_timestamp(f"Statistics saved to {stats_file}")
    
    # Also print to console
    print(report)

def generate_description_and_expressions(client, rate_limiter, image_path, object_name, original_expressions, 
                                        bbox=None, worker_id=None, worker_stats=None, 
                                        image_filename=None, obj_id=None):
    """Generate object description and enhanced expressions using Google Generative AI API"""
    # Always log start and end of generation regardless of verbose flag
    log_with_timestamp(f"START GENERATION - worker {worker_id}", worker_id)
    start_time = time.time()
    
    # Process image if bounding box is provided
    temp_image_path = None
    path_to_use = image_path # Path of the image file to use for the API call
    try:
        if bbox:
            temp_image_path = process_image_for_api(image_path, bbox)
            if not temp_image_path:
                return None, None, "Failed to process image"
            path_to_use = temp_image_path
        # else: path_to_use remains image_path
        
        # Read image bytes
        try:
            with open(path_to_use, "rb") as image_file:
                image_bytes = image_file.read()
        except Exception as e:
            log_with_timestamp(f"Error reading image file {path_to_use}: {e}", worker_id)
            return None, None, f"Failed to read image file: {e}"

        # Format original expressions for the prompt
        formatted_expressions = "\n".join([f"- {expr}" for expr in original_expressions])
        
        # Create the combined prompt for both description and expression enhancement
        prompt = (
            "You have two tasks:\n\n"
            "TASK 1: Describe the object highlighted with a red bounding box factually and concisely. Focus on:\n"
            "1. Observable visual features (shape, color, orientation) without speculation\n"
            "2. Definite spatial relationships to nearby objects that are clearly visible\n"
            "3. Distinctive identifying characteristics that would help locate this specific object\n"
            "4. Actual contextual elements (roads, buildings, terrain) that are directly observable\n"
            "Limit your description to 150-200 words. Only describe what you can see with high confidence.\n\n"
            
            f"TASK 2: For each original expression listed below, create EXACTLY {NUM_ENHANCED} enhanced versions that:\n"
            "1. MUST PRESERVE ALL SPATIAL INFORMATION from the original expression:\n"
            "   - Absolute positions (e.g., \"in the top right\", \"near the center\")\n"
            "   - Relative positions (e.g., \"to the right of\", \"below\")\n"
            "2. Add relevant visual details from what you observe in the image\n"
            "3. Ensure each expression uniquely identifies this object to avoid ambiguity\n"
            "4. Vary the length of the enhanced expressions as follows:\n"
            "   - First version: Make it concise - if original is already 20+ words, keep similar length but improve clarity\n"
            "   - Second version: Add 30-50% more detail than the original expression\n"
            "   - Third version: Make it comprehensive with twice the detail of the original\n\n"
            
            f"ORIGINAL EXPRESSIONS TO ENHANCE:\n{formatted_expressions}\n\n"
            
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
        
        user_prompt = f"Provide a factual description of this {object_name} (highlighted with a red bounding box), and enhance the provided expressions to make them more descriptive while preserving all spatial information."
        
        # Estimate input tokens: prompt + user prompt + (Image token cost is harder to estimate precisely with upload)
        input_token_estimate = estimate_tokens(prompt) + estimate_tokens(user_prompt) + 259 # Keep estimate
        
        # Call API with retries
        for attempt in range(MAX_RETRIES):
            try:
                # Wait if needed for rate limiting
                rate_limiter.wait_if_needed()
                
                api_start_time = time.time()

                # Create the request content using image bytes
                contents = [
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_text(text=user_prompt),
                            types.Part.from_bytes(
                                data=image_bytes,
                                mime_type="image/jpeg" # Assuming JPEG, adjust if needed
                            )
                        ]
                    )
                ]
                
                # Define the expected JSON output schema based on Pydantic models
                response_schema = types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        'description': types.Schema(type=types.Type.STRING),
                        'enhanced_expressions': types.Schema(
                            type=types.Type.ARRAY,
                            items=types.Schema(
                                type=types.Type.OBJECT,
                                properties={
                                    'original_expression': types.Schema(type=types.Type.STRING),
                                    'enhanced_versions': types.Schema(
                                        type=types.Type.ARRAY,
                                        items=types.Schema(type=types.Type.STRING)
                                    )
                                },
                                required=['original_expression', 'enhanced_versions']
                            )
                        )
                    },
                    required=['description', 'enhanced_expressions']
                )

                # Create GenerationConfig object including the schema
                generation_config_object = types.GenerateContentConfig(
                    temperature=0.2,
                    top_p=0.95,
                    max_output_tokens=2048,
                    response_mime_type="application/json",
                    response_schema=response_schema
                )

                # Make the API call to the custom model using client.models and config
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=contents,
                    config=generation_config_object
                )
                
                api_time = time.time() - api_start_time
                
                # Update stats
                if worker_stats is not None:
                    worker_stats['api_requests'] += 1
                    worker_stats['api_time'] += api_time
                    worker_stats['input_tokens'] += input_token_estimate
                
                # Parse the JSON response
                response_text = response.text
                
                # Save raw response for debugging if requested
                if DEBUG:
                    raw_response = {
                        "response_text": response_text,
                        "usage": getattr(response, "usage", None)
                    }
                    debug_path = save_debug_output(OUTPUT_DIR, image_filename, obj_id, worker_id, raw_response)
                    log_with_timestamp(f"Debug output saved to {debug_path}", worker_id)
                
                # Parse the JSON response into our Pydantic model
                try:
                    parsed_output = json.loads(response_text)
                    combined_output = CombinedOutput(**parsed_output)
                    
                    # Extract the results
                    description = combined_output.description
                    enhanced_expressions_data = combined_output.enhanced_expressions
                    
                    # Estimate output tokens
                    output_token_estimate = estimate_tokens(description)
                    for item in enhanced_expressions_data:
                        output_token_estimate += estimate_tokens(item.original_expression)
                        for version in item.enhanced_versions:
                            output_token_estimate += estimate_tokens(version)
                    
                    # Update output token stats
                    if worker_stats is not None:
                        worker_stats['output_tokens'] += output_token_estimate
                    
                    end_time = time.time()
                    log_with_timestamp(f"END GENERATION - worker {worker_id} - took {end_time - start_time:.2f} seconds", worker_id)
                    
                    return description, enhanced_expressions_data, None
                    
                except json.JSONDecodeError as e:
                    log_with_timestamp(f"Error parsing JSON response: {e}", worker_id)
                    if DEBUG:
                        debug_path = save_debug_output(OUTPUT_DIR, image_filename, obj_id, worker_id, 
                                                     {"error": "JSON parse error", "response_text": response_text})
                        log_with_timestamp(f"Invalid JSON saved to {debug_path}", worker_id)
                    
                    time.sleep(DELAY)
                    continue # Retry if JSON parsing failed
                
            except Exception as e:
                log_with_timestamp(f"Error on attempt {attempt+1}/{MAX_RETRIES}: {str(e)}", worker_id)
                time.sleep(DELAY)
                
                if DEBUG:
                    # Save error information
                    error_info = {
                        "error": str(e),
                        "attempt": attempt + 1,
                        "max_retries": MAX_RETRIES,
                        "timestamp": datetime.datetime.now().isoformat()
                    }
                    debug_path = save_debug_output(OUTPUT_DIR, image_filename, obj_id, worker_id, error_info)
                    log_with_timestamp(f"Error debug info saved to {debug_path}", worker_id)
        
        return None, None, f"Failed after {MAX_RETRIES} attempts"
        
    finally:
        # Clean up temp file if it exists
        if temp_image_path and temp_image_path.startswith("temp_") and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

def pretty_format_xml(element):
    """Format XML with proper indentation for readability without extra blank lines"""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    
    # Get pretty XML with indentation
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Remove extra blank lines (common issue with toprettyxml)
    pretty_xml = re.sub(r'>\s*\n\s*\n+', '>\n', pretty_xml)
    return pretty_xml

def worker_process(worker_id, file_queue, done_queue, stats=None):
    """Worker process that processes files from the queue"""
    try:
        # Set up API client for this process
        client = get_client()
        rate_limiter = RateLimiter(MAX_REQUESTS_PER_MINUTE // NUM_WORKERS)
        
        # Initialize worker-specific stats
        worker_stats = {
            'instances': 0,
            'api_requests': 0,
            'enhanced_expressions': 0,
            'original_expressions': 0,
            'input_tokens': 0,
            'output_tokens': 0,
            'api_time': 0.0
        }
        
        log_with_timestamp(f"Worker ready", worker_id)
        
        while True:
            try:
                # Get a file from the queue with a timeout
                xml_file = file_queue.get(timeout=5)
                
                # Process the file
                xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
                
                # Parse XML
                tree = ET.parse(xml_path)
                root = tree.getroot()
                
                # Get image filename (without extension)
                image_filename = os.path.splitext(root.find('filename').text)[0]
                
                # Process each object
                for obj in root.findall('object'):
                    obj_id = obj.find('id').text
                    object_name = obj.find('name').text
                    
                    # Get bounding box
                    bbox_elem = obj.find('bndbox')
                    bbox = None
                    if bbox_elem is not None:
                        xmin = int(bbox_elem.find('xmin').text)
                        ymin = int(bbox_elem.find('ymin').text)
                        xmax = int(bbox_elem.find('xmax').text)
                        ymax = int(bbox_elem.find('ymax').text)
                        bbox = [xmin, ymin, xmax, ymax]
                    else:
                        # Skip objects without bbox - no warning needed
                        continue
                    
                    # Check for expressions - silently skip if not found
                    expressions_elem = obj.find('expressions')
                    if expressions_elem is None:
                        # Silently skip objects without expressions
                        continue
                    
                    # Extract original expressions - silently skip empty ones
                    original_expressions = []
                    for expr in expressions_elem.findall('expression'):
                        if expr.text and expr.text.strip():
                            original_expressions.append(expr.text)
                    
                    if not original_expressions:
                        # Silently skip if no valid expressions were found
                        continue
                    
                    # Find the patch image - only warn if needed for debugging
                    patch_path = os.path.join(PATCHES_FOLDER, f"{image_filename}.jpg")
                    if not os.path.exists(patch_path):
                        patch_path = os.path.join(PATCHES_FOLDER, f"{image_filename}.png")
                    
                    if not os.path.exists(patch_path):
                        if DEBUG:  # Only log this warning in debug mode
                            log_with_timestamp(f"Warning: Patch not found: {patch_path}", worker_id)
                        continue
                    
                    try:
                        # Update stats - instance count
                        worker_stats['instances'] += 1
                        worker_stats['original_expressions'] += len(original_expressions)
                        
                        # Generate description and enhanced expressions
                        description, enhanced_expressions_data, error = generate_description_and_expressions(
                            client, rate_limiter, patch_path, object_name, original_expressions,
                            bbox, worker_id, worker_stats, image_filename, obj_id
                        )
                        
                        if description and enhanced_expressions_data:
                            # Add description to XML
                            desc_elem = ET.SubElement(obj, 'raw_llm_description')
                            desc_elem.text = description
                            
                            # Create enhanced_expressions element
                            enhanced_elem = ET.SubElement(obj, 'enhanced_expressions')
                            
                            # Map original expressions to their XML elements to find ids
                            original_expr_map = {}
                            for expr in expressions_elem.findall('expression'):
                                original_expr_map[expr.text] = expr.get('id', str(expressions_elem.findall('expression').index(expr)))
                            
                            # Add enhanced expressions, mapping to original expressions
                            for enhanced_expr_item in enhanced_expressions_data:
                                original_text = enhanced_expr_item.original_expression
                                enhanced_versions = enhanced_expr_item.enhanced_versions
                                
                                # Find the original expression ID
                                original_id = None
                                for expr in expressions_elem.findall('expression'):
                                    if expr.text == original_text:
                                        original_id = expr.get('id', str(expressions_elem.findall('expression').index(expr)))
                                        break
                                
                                # If original not found exactly, use best match based on original_expr_map
                                if original_id is None and original_expr_map:
                                    # Use the mapping we created earlier
                                    closest_match = min(original_expr_map.keys(), 
                                                     key=lambda x: abs(len(x) - len(original_text)))
                                    original_id = original_expr_map[closest_match]
                                
                                # If still no ID, generate one
                                if original_id is None:
                                    original_id = str(enhanced_expressions_data.index(enhanced_expr_item))
                                
                                # Count enhanced expressions for stats
                                worker_stats['enhanced_expressions'] += len(enhanced_versions)
                                
                                # Add each enhanced version with proper ID
                                for j, enhanced_text in enumerate(enhanced_versions):
                                    enhanced_expr = ET.SubElement(enhanced_elem, 'expression')
                                    enhanced_expr.set('id', f"{original_id}_{j}")
                                    enhanced_expr.set('base_id', original_id)
                                    enhanced_expr.text = enhanced_text
                        else:
                            log_with_timestamp(f"Failed to generate output for {patch_path}: {error}", worker_id)
                            
                    except Exception as e:
                        log_with_timestamp(f"Error processing {patch_path}: {e}", worker_id)
                
                # Process groups if present
                groups_elem = root.find('groups')
                if groups_elem is not None:
                    for group in groups_elem.findall('group'):
                        group_id = group.find('id').text
                        
                        # Extract instance IDs for this group
                        instance_ids_elem = group.find('instance_ids')
                        if instance_ids_elem is None or not instance_ids_elem.text:
                            continue
                            
                        # Get group category
                        category = group.find('category').text
                        
                        # Find expressions for this group
                        group_expressions_elem = group.find('expressions')
                        if group_expressions_elem is None:
                            continue
                            
                        # Extract original expressions
                        original_expressions = []
                        for expr in group_expressions_elem.findall('expression'):
                            if expr.text and expr.text.strip():
                                original_expressions.append(expr.text)
                        
                        if not original_expressions:
                            continue
                            
                        # Calculate bounding box for the entire group
                        # Option 1: Use the min/max of all instance bounding boxes
                        instance_ids = instance_ids_elem.text.split(',')
                        min_x, min_y = float('inf'), float('inf')
                        max_x, max_y = float('-inf'), float('-inf')
                        
                        # Find all objects in the group and get their bounding boxes
                        for obj in root.findall('object'):
                            obj_id = obj.find('id').text
                            if obj_id in instance_ids:
                                bbox_elem = obj.find('bndbox')
                                if bbox_elem is not None:
                                    xmin = int(bbox_elem.find('xmin').text)
                                    ymin = int(bbox_elem.find('ymin').text)
                                    xmax = int(bbox_elem.find('xmax').text)
                                    ymax = int(bbox_elem.find('ymax').text)
                                    
                                    min_x = min(min_x, xmin)
                                    min_y = min(min_y, ymin)
                                    max_x = max(max_x, xmax)
                                    max_y = max(max_y, ymax)
                        
                        # Skip if we couldn't find a valid bounding box
                        if min_x == float('inf') or min_y == float('inf') or max_x == float('-inf') or max_y == float('-inf'):
                            continue
                            
                        # Create bounding box for the group
                        group_bbox = [int(min_x), int(min_y), int(max_x), int(max_y)]
                        
                        try:
                            # Update stats
                            worker_stats['instances'] += 1
                            worker_stats['original_expressions'] += len(original_expressions)
                            
                            # Generate description and enhanced expressions for the group
                            description, enhanced_expressions_data, error = generate_description_and_expressions(
                                client, rate_limiter, patch_path, f"group of {category}", original_expressions,
                                group_bbox, worker_id, worker_stats, image_filename, f"group_{group_id}"
                            )
                            
                            if description and enhanced_expressions_data:
                                # Add description to group XML
                                desc_elem = ET.SubElement(group, 'raw_llm_description')
                                desc_elem.text = description
                                
                                # Create enhanced_expressions element for group
                                enhanced_elem = ET.SubElement(group, 'enhanced_expressions')
                                
                                # Map original expressions to their XML elements to find ids
                                original_expr_map = {}
                                for expr in group_expressions_elem.findall('expression'):
                                    original_expr_map[expr.text] = expr.get('id', str(group_expressions_elem.findall('expression').index(expr)))
                                
                                # Add enhanced expressions, mapping to original expressions
                                for enhanced_expr_item in enhanced_expressions_data:
                                    original_text = enhanced_expr_item.original_expression
                                    enhanced_versions = enhanced_expr_item.enhanced_versions
                                    
                                    # Find the original expression ID
                                    original_id = None
                                    for expr in group_expressions_elem.findall('expression'):
                                        if expr.text == original_text:
                                            original_id = expr.get('id', str(group_expressions_elem.findall('expression').index(expr)))
                                            break
                                    
                                    # If original not found exactly, use best match
                                    if original_id is None and original_expr_map:
                                        closest_match = min(original_expr_map.keys(), 
                                                         key=lambda x: abs(len(x) - len(original_text)))
                                        original_id = original_expr_map[closest_match]
                                    
                                    # If still no ID, generate one
                                    if original_id is None:
                                        original_id = str(enhanced_expressions_data.index(enhanced_expr_item))
                                    
                                    # Count enhanced expressions for stats
                                    worker_stats['enhanced_expressions'] += len(enhanced_versions)
                                    
                                    # Add each enhanced version with proper ID
                                    for j, enhanced_text in enumerate(enhanced_versions):
                                        enhanced_expr = ET.SubElement(enhanced_elem, 'expression')
                                        enhanced_expr.set('id', f"{original_id}_{j}")
                                        enhanced_expr.set('base_id', original_id)
                                        enhanced_expr.text = enhanced_text
                            else:
                                log_with_timestamp(f"Failed to generate output for group {group_id}: {error}", worker_id)
                                
                        except Exception as e:
                            log_with_timestamp(f"Error processing group {group_id}: {e}", worker_id)
                
                # Save updated XML
                output_annotations_dir = os.path.join(OUTPUT_DIR, 'annotations')
                output_xml_path = os.path.join(output_annotations_dir, xml_file)
                pretty_xml = pretty_format_xml(root)
                
                with open(output_xml_path, 'w', encoding='utf-8') as f:
                    f.write(pretty_xml)
                
                # Signal that the file has been processed
                done_queue.put((worker_id, xml_file))
                
            except queue.Empty:
                # No more files to process
                break
        
        # Send worker stats back to main process
        done_queue.put(('STATS', worker_id, worker_stats))
                
    except Exception as e:
        log_with_timestamp(f"Error: {e}", worker_id)
    finally:
        # Clean up
        log_with_timestamp("Worker shutting down", worker_id)

def main():
    # Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_annotations_dir = os.path.join(OUTPUT_DIR, 'annotations')
    os.makedirs(output_annotations_dir, exist_ok=True)
    
    # Create debug directory if needed
    if DEBUG:
        debug_dir = os.path.join(OUTPUT_DIR, 'debug')
        os.makedirs(debug_dir, exist_ok=True)
        log_with_timestamp(f"Debug mode enabled. Raw outputs will be saved to {debug_dir}")
    
    # Get list of annotation files
    annotation_files = [f for f in os.listdir(ANNOTATIONS_DIR) if f.endswith('.xml')]
    total_files = len(annotation_files)
    log_with_timestamp(f"Found {total_files} annotation files to process")
    
    # Create queues for communication
    file_queue = multiprocessing.Queue()
    done_queue = multiprocessing.Queue()
    
    # Add all files to the queue
    for f in annotation_files:
        file_queue.put(f)
    
    # Create shared statistics
    stats = Stats()
    
    # Start worker processes
    workers = []
    for i in range(NUM_WORKERS):
        p = multiprocessing.Process(
            target=worker_process,
            args=(i, file_queue, done_queue, stats)
        )
        workers.append(p)
        p.start()
        log_with_timestamp(f"Started worker {i}")
    
    # Monitor progress with tqdm
    with tqdm(total=total_files, desc="Processing annotations") as pbar:
        processed_count = 0
        worker_stats_received = 0
        
        while processed_count < total_files or worker_stats_received < NUM_WORKERS:
            try:
                # Wait for a file to be processed or stats to be reported
                message = done_queue.get(timeout=1)
                
                if message[0] == 'STATS':
                    # Received stats from a worker
                    worker_id = message[1]
                    worker_stats = message[2]
                    update_stats(stats, worker_stats)
                    worker_stats_received += 1
                    log_with_timestamp(f"Received stats from worker {worker_id}")
                else:
                    # Received processed file notification
                    worker_id, filename = message
                    processed_count += 1
                    pbar.set_postfix({"worker": worker_id, "file": os.path.basename(filename)})
                    pbar.update(1)
            except queue.Empty:
                # Check if workers are still alive
                alive_workers = [p for p in workers if p.is_alive()]
                if not alive_workers and processed_count >= total_files:
                    break
    
    # Wait for all workers to finish
    for p in workers:
        p.join()
    
    log_with_timestamp(f"Processed {processed_count} of {total_files} annotation files")
    
    # Save final statistics
    save_stats_to_file(stats, OUTPUT_DIR)
    
    log_with_timestamp(f"Generated descriptions and enhanced expressions saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main() 