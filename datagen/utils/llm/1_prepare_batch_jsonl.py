#!/usr/bin/env python3
import os
import sys
import xml.etree.ElementTree as ET
import json
import argparse
import time
from tqdm import tqdm
from google.cloud import storage
import cv2
import base64
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import threading
import datetime

# Constants
PROJECT_ID = "564504826453"
BUCKET_NAME = "aerial-bucket"
GCS_PATH_PREFIX = "batch_inference/images"
NUM_WORKERS = 8
DEFAULT_DATASET_FOLDER = "dataset"
NUM_ENHANCED = 2
NUM_UNIQUE = 2

def parse_args():
    parser = argparse.ArgumentParser(description='Prepare batch JSONL files for LLM processing')
    parser.add_argument('--dataset_folder', type=str, default=DEFAULT_DATASET_FOLDER,
                      help=f'Dataset folder name (default: {DEFAULT_DATASET_FOLDER})')
    parser.add_argument('--split', choices=['train', 'val', 'both'], default='both',
                      help='Which split to process (train, val, or both)')
    return parser.parse_args()

def log_with_timestamp(message):
    """Add timestamp to log messages"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def upload_image_to_gcs(local_path, blob_name, bucket):
    """Upload an image to Google Cloud Storage"""
    try:
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(local_path)
        return f"gs://{BUCKET_NAME}/{blob_name}"
    except Exception as e:
        log_with_timestamp(f"Error uploading {local_path}: {e}")
        return None

def process_image_for_api(image_path, bbox=None):
    """Process image for API - highlight instance with bounding box if provided"""
    image = cv2.imread(image_path)
    if image is None:
        return None, None
    
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
    
    filename = os.path.basename(image_path)
    base_name, ext = os.path.splitext(filename)
    temp_path = f"temp_{base_name}{ext}"
    cv2.imwrite(temp_path, image)
    return temp_path, base_name

def create_request_json(gcs_uri, object_name, original_expressions):
    """Create the JSON request for a single instance"""
    formatted_expressions = "\n".join([f"- {expr}" for expr in original_expressions])
    
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
    
    request = {
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt},
                        {"text": user_prompt},
                        {
                            "fileData": {
                                "fileUri": gcs_uri,
                                "mimeType": "image/jpeg"
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 1.0,
                "topP": 0.95,
                "maxOutputTokens": 4096,
                "responseMimeType": "application/json",
                "responseSchema": response_schema
            }
        }
    }
    
    return request

def process_file(xml_file, bucket, output_file, lock, split, patches_folder, annotations_dir):
    """Process a single XML file, upload images, and write requests to JSONL"""
    try:
        xml_path = os.path.join(annotations_dir, split, "annotations", xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        image_filename = os.path.splitext(root.find('filename').text)[0]
        requests = []
        
        # Process each object
        for obj in root.findall('object'):
            obj_id = obj.find('id').text
            object_name = obj.find('name').text
            
            bbox_elem = obj.find('bndbox')
            bbox = None
            if bbox_elem is not None:
                xmin = int(bbox_elem.find('xmin').text)
                ymin = int(bbox_elem.find('ymin').text)
                xmax = int(bbox_elem.find('xmax').text)
                ymax = int(bbox_elem.find('ymax').text)
                bbox = [xmin, ymin, xmax, ymax]
            else:
                continue
            
            expressions_elem = obj.find('expressions')
            if expressions_elem is None:
                continue
            
            original_expressions = []
            for expr in expressions_elem.findall('expression'):
                if expr.text and expr.text.strip():
                    original_expressions.append(expr.text)
            
            if not original_expressions:
                continue
            
            split_folder = os.path.join(patches_folder, split, "images")
            patch_path = os.path.join(split_folder, f"{image_filename}.jpg")
            if not os.path.exists(patch_path):
                patch_path = os.path.join(split_folder, f"{image_filename}.png")
            
            if not os.path.exists(patch_path):
                continue
            
            temp_image_path, base_name = process_image_for_api(patch_path, bbox)
            if temp_image_path:
                blob_name = f"{GCS_PATH_PREFIX}/{base_name}_obj_{obj_id}.jpg"
                gcs_uri = upload_image_to_gcs(temp_image_path, blob_name, bucket)
                
                if gcs_uri:
                    request = create_request_json(gcs_uri, object_name, original_expressions)
                    requests.append(request)
                
                if temp_image_path.startswith("temp_") and os.path.exists(temp_image_path):
                    os.remove(temp_image_path)
        
        # Process groups if present
        groups_elem = root.find('groups')
        if groups_elem is not None:
            for group in groups_elem.findall('group'):
                group_id = group.find('id').text
                
                instance_ids_elem = group.find('instance_ids')
                if instance_ids_elem is None or not instance_ids_elem.text:
                    continue
                    
                category = group.find('category').text
                group_expressions_elem = group.find('expressions')
                if group_expressions_elem is None:
                    continue
                    
                original_expressions = []
                for expr in group_expressions_elem.findall('expression'):
                    if expr.text and expr.text.strip():
                        original_expressions.append(expr.text)
                
                if not original_expressions:
                    continue
                    
                instance_ids = instance_ids_elem.text.split(',')
                min_x, min_y = float('inf'), float('inf')
                max_x, max_y = float('-inf'), float('-inf')
                
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
                
                if min_x == float('inf') or min_y == float('inf') or max_x == float('-inf') or max_y == float('-inf'):
                    continue
                    
                group_bbox = [int(min_x), int(min_y), int(max_x), int(max_y)]
                
                split_folder = os.path.join(patches_folder, split, "images")
                patch_path = os.path.join(split_folder, f"{image_filename}.jpg")
                if not os.path.exists(patch_path):
                    patch_path = os.path.join(split_folder, f"{image_filename}.png")
                
                if not os.path.exists(patch_path):
                    continue
                
                temp_image_path, base_name = process_image_for_api(patch_path, group_bbox)
                if temp_image_path:
                    blob_name = f"{GCS_PATH_PREFIX}/{base_name}_group_{group_id}.jpg"
                    gcs_uri = upload_image_to_gcs(temp_image_path, blob_name, bucket)
                    
                    if gcs_uri:
                        request = create_request_json(gcs_uri, f"group of {category}", original_expressions)
                        requests.append(request)
                    
                    if temp_image_path.startswith("temp_") and os.path.exists(temp_image_path):
                        os.remove(temp_image_path)
        
        if requests:
            with lock:
                with open(output_file, 'a', encoding='utf-8') as f:
                    for req in requests:
                        f.write(json.dumps(req) + '\n')
        
        return len(requests)
        
    except Exception as e:
        log_with_timestamp(f"Error processing {xml_file}: {e}")
        return 0

def main():
    args = parse_args()
    dataset_folder = args.dataset_folder
    
    # Construct paths using the dataset folder
    patches_folder = os.path.join(dataset_folder, "patches")
    annotations_dir = os.path.join(dataset_folder, "patches_rules_expressions_unique")
    jsonl_dir = os.path.join(dataset_folder, "batch_prediction")
    output_jsonl_path_train = os.path.join(jsonl_dir, "batch_prediction_requests_train.jsonl")
    output_jsonl_path_val = os.path.join(jsonl_dir, "batch_prediction_requests_val.jsonl")
    
    # Create output directory if it doesn't exist
    os.makedirs(jsonl_dir, exist_ok=True)
    
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    
    # Determine which splits to process
    splits_to_process = ['train', 'val'] if args.split == 'both' else [args.split]
    
    for split in splits_to_process:
        log_with_timestamp(f"Processing {split} split...")
        
        output_path = output_jsonl_path_train if split == 'train' else output_jsonl_path_val
        if os.path.exists(output_path):
            os.remove(output_path)
        
        split_annotations_dir = os.path.join(annotations_dir, split, "annotations")
        xml_files = [f for f in os.listdir(split_annotations_dir) if f.endswith('.xml')]
        
        if not xml_files:
            log_with_timestamp(f"No XML files found for {split} split")
            continue
        
        lock = threading.Lock()
        
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
            futures = []
            for xml_file in xml_files:
                future = executor.submit(process_file, xml_file, bucket, output_path, lock, split, patches_folder, annotations_dir)
                futures.append(future)
            
            total_requests = 0
            for future in tqdm(futures, desc=f"Processing {split} files"):
                try:
                    num_requests = future.result()
                    total_requests += num_requests
                except Exception as e:
                    log_with_timestamp(f"Error processing file: {str(e)}")
        
        log_with_timestamp(f"Completed {split} split - Generated {total_requests} requests")

if __name__ == "__main__":
    main() 