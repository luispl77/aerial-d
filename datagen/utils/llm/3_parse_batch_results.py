#!/usr/bin/env python3
import os
import sys
import json
import argparse
import xml.etree.ElementTree as ET
from xml.dom import minidom
import re
from tqdm import tqdm
import datetime
import glob

# Constants
DEFAULT_DATASET_FOLDER = "dataset"
NUM_ENHANCED = 2  # Same as in the original and batch scripts
NUM_UNIQUE = 2    # Same as in the original and batch scripts

def parse_args():
    parser = argparse.ArgumentParser(description='Parse batch prediction results')
    parser.add_argument('--dataset_folder', type=str, default=DEFAULT_DATASET_FOLDER,
                      help=f'Dataset folder name (default: {DEFAULT_DATASET_FOLDER})')
    parser.add_argument('--split', choices=['train', 'val', 'both'], default='both',
                      help='Which split to process (train, val, or both)')
    return parser.parse_args()

def log_with_timestamp(message):
    """Add timestamp to log messages"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def extract_image_info_from_gcs_uri(gcs_uri):
    """Extract image filename, object ID, and type from GCS URI"""
    # Pattern like: gs://bucket/batch_inference/images/P0001_obj_123.jpg
    match = re.search(r'/([^/]+)_(obj|group)_([^/]+)\.', gcs_uri)
    if match:
        image_name = match.group(1)  # P0001
        obj_type = match.group(2)    # obj or group
        obj_id = match.group(3)      # 123
        return image_name, obj_type, obj_id
    return None, None, None

def pretty_format_xml(element):
    """Format XML with proper indentation for readability without extra blank lines"""
    rough_string = ET.tostring(element, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    
    # Get pretty XML with indentation
    pretty_xml = reparsed.toprettyxml(indent="  ")
    
    # Remove extra blank lines (common issue with toprettyxml)
    pretty_xml = re.sub(r'>\s*\n\s*\n+', '>\n', pretty_xml)
    return pretty_xml

def parse_results_file(results_file):
    """Parse a batch prediction results file"""
    results = []
    with open(results_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(tqdm(f, desc="Parsing result lines")):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line)
                
                # Extract request data (for image URI)
                request = data.get('request', {})
                contents = request.get('contents', [{}])[0]
                parts = contents.get('parts', [])
                
                # Find the fileData part to get the image URI
                image_uri = None
                for part in parts:
                    if 'fileData' in part and part['fileData']:
                        image_uri = part['fileData'].get('fileUri')
                        break
                
                if not image_uri:
                    log_with_timestamp(f"No image URI found in line {line_num+1}")
                    continue
                
                # Extract image info
                image_name, obj_type, obj_id = extract_image_info_from_gcs_uri(image_uri)
                if not all([image_name, obj_type, obj_id]):
                    log_with_timestamp(f"Could not extract image info from URI: {image_uri}")
                    continue
                
                # Extract prediction data
                response = data.get('response', {})
                candidates = response.get('candidates', [])
                if not candidates:
                    log_with_timestamp(f"No candidates found in line {line_num+1}")
                    continue
                
                # Get content from first candidate
                content = candidates[0].get('content', {})
                parts = content.get('parts', [])
                if not parts:
                    log_with_timestamp(f"No content parts found in line {line_num+1}")
                    continue
                
                # Get the text part (JSON output)
                json_text = None
                for part in parts:
                    if 'text' in part:
                        json_text = part['text']
                        break
                
                if not json_text:
                    log_with_timestamp(f"No text found in content parts in line {line_num+1}")
                    continue
                
                # Parse the JSON output
                try:
                    parsed_output = json.loads(json_text)
                    
                    # Add image and object info to the parsed output
                    parsed_output['image_name'] = image_name
                    parsed_output['obj_type'] = obj_type
                    parsed_output['obj_id'] = obj_id
                    
                    results.append(parsed_output)
                except json.JSONDecodeError as e:
                    log_with_timestamp(f"Error parsing JSON in result line {line_num+1}: {e}")
                    continue
                
            except json.JSONDecodeError as e:
                log_with_timestamp(f"Error parsing result line {line_num+1}: {e}")
                continue
            except Exception as e:
                log_with_timestamp(f"Unexpected error parsing line {line_num+1}: {str(e)}")
                continue
    
    log_with_timestamp(f"Successfully parsed {len(results)} results")
    return results

def update_xml_with_results(results, annotations_dir, output_dir, split):
    """Update XML files with batch prediction results"""
    # Group results by image name
    results_by_image = {}
    for result in results:
        image_name = result['image_name']
        if image_name not in results_by_image:
            results_by_image[image_name] = []
        results_by_image[image_name].append(result)
    
    # Create output directory for annotations
    output_annotations_dir = os.path.join(output_dir, split, "annotations")
    os.makedirs(output_annotations_dir, exist_ok=True)
    
    # Process each image
    processed_files = 0
    for image_name, image_results in results_by_image.items():
        # Find the XML file for this image in the correct split directory
        split_annotations_dir = os.path.join(annotations_dir, split, "annotations")
        xml_files = glob.glob(os.path.join(split_annotations_dir, f"{image_name}*.xml"))
        if not xml_files:
            log_with_timestamp(f"No XML file found for image {image_name}")
            continue
        
        xml_file = xml_files[0]
        
        try:
            # Parse the XML file
            tree = ET.parse(xml_file)
            root = tree.getroot()
            
            # Process each result for this image
            for result in image_results:
                obj_type = result['obj_type']
                obj_id = result['obj_id']
                enhanced_expressions = result.get('enhanced_expressions', [])
                
                if obj_type == 'obj':
                    # Find the object element
                    for obj in root.findall('object'):
                        if obj.find('id').text == obj_id:
                            # Add enhanced expressions
                            expressions_elem = obj.find('expressions')
                            if expressions_elem is not None:
                                enhanced_elem = ET.SubElement(obj, 'enhanced_expressions')
                                
                                # Map original expressions to IDs
                                original_expr_map = {}
                                for expr in expressions_elem.findall('expression'):
                                    original_expr_map[expr.text] = expr.get('id', 
                                                                         str(expressions_elem.findall('expression').index(expr)))
                                
                                # Add each enhanced expression
                                for enhanced_expr_item in enhanced_expressions:
                                    original_text = enhanced_expr_item.get('original_expression', '')
                                    variations = enhanced_expr_item.get('variations', [])
                                    
                                    # Find original ID
                                    original_id = None
                                    for expr in expressions_elem.findall('expression'):
                                        if expr.text == original_text:
                                            original_id = expr.get('id', 
                                                                str(expressions_elem.findall('expression').index(expr)))
                                            break
                                    
                                    # If not found exactly, use best match
                                    if original_id is None and original_expr_map:
                                        closest_match = min(original_expr_map.keys(), 
                                                         key=lambda x: abs(len(x) - len(original_text)))
                                        original_id = original_expr_map[closest_match]
                                    
                                    # If still no ID, generate one
                                    if original_id is None:
                                        original_id = str(enhanced_expressions.index(enhanced_expr_item))
                                    
                                    # Add variations
                                    for j, variation in enumerate(variations):
                                        enhanced_expr = ET.SubElement(enhanced_elem, 'expression')
                                        enhanced_expr.set('id', f"{original_id}_var_{j}")
                                        enhanced_expr.set('base_id', original_id)
                                        enhanced_expr.text = variation
                                
                                # Add unique expressions
                                unique_expressions = result.get('unique_expressions', [])
                                if unique_expressions:
                                    # Add unique description first
                                    unique_desc = result.get('unique_description')
                                    if unique_desc:
                                        unique_desc_elem = ET.SubElement(obj, 'unique_description')
                                        unique_desc_elem.text = unique_desc
                                    
                                    unique_elem = ET.SubElement(obj, 'unique_expressions')
                                    for j, unique_expr in enumerate(unique_expressions):
                                        unique_expr_elem = ET.SubElement(unique_elem, 'expression')
                                        unique_expr_elem.set('id', f"unique_{j}")
                                        unique_expr_elem.text = unique_expr
                            
                            break
                
                elif obj_type == 'group':
                    # Find the group element
                    groups_elem = root.find('groups')
                    if groups_elem is not None:
                        for group in groups_elem.findall('group'):
                            if group.find('id').text == obj_id:
                                # Add enhanced expressions
                                group_expressions_elem = group.find('expressions')
                                if group_expressions_elem is not None:
                                    enhanced_elem = ET.SubElement(group, 'enhanced_expressions')
                                    
                                    # Map original expressions to IDs
                                    original_expr_map = {}
                                    for expr in group_expressions_elem.findall('expression'):
                                        original_expr_map[expr.text] = expr.get('id', 
                                                                             str(group_expressions_elem.findall('expression').index(expr)))
                                    
                                    # Add each enhanced expression
                                    for enhanced_expr_item in enhanced_expressions:
                                        original_text = enhanced_expr_item.get('original_expression', '')
                                        variations = enhanced_expr_item.get('variations', [])
                                        
                                        # Find original ID
                                        original_id = None
                                        for expr in group_expressions_elem.findall('expression'):
                                            if expr.text == original_text:
                                                original_id = expr.get('id', 
                                                                    str(group_expressions_elem.findall('expression').index(expr)))
                                                break
                                        
                                        # If not found exactly, use best match
                                        if original_id is None and original_expr_map:
                                            closest_match = min(original_expr_map.keys(), 
                                                             key=lambda x: abs(len(x) - len(original_text)))
                                            original_id = original_expr_map[closest_match]
                                        
                                        # If still no ID, generate one
                                        if original_id is None:
                                            original_id = str(enhanced_expressions.index(enhanced_expr_item))
                                        
                                        # Add variations
                                        for j, variation in enumerate(variations):
                                            enhanced_expr = ET.SubElement(enhanced_elem, 'expression')
                                            enhanced_expr.set('id', f"{original_id}_var_{j}")
                                            enhanced_expr.set('base_id', original_id)
                                            enhanced_expr.text = variation
                                
                                # Add unique expressions
                                unique_expressions = result.get('unique_expressions', [])
                                if unique_expressions:
                                    # Add unique description first
                                    unique_desc = result.get('unique_description')
                                    if unique_desc:
                                        unique_desc_elem = ET.SubElement(group, 'unique_description')
                                        unique_desc_elem.text = unique_desc
                                    
                                    unique_elem = ET.SubElement(group, 'unique_expressions')
                                    for j, unique_expr in enumerate(unique_expressions):
                                        unique_expr_elem = ET.SubElement(unique_elem, 'expression')
                                        unique_expr_elem.set('id', f"unique_{j}")
                                        unique_expr_elem.text = unique_expr
                                
                                break
            
            # Save the updated XML
            output_xml_path = os.path.join(output_annotations_dir, os.path.basename(xml_file))
            pretty_xml = pretty_format_xml(root)
            with open(output_xml_path, 'w', encoding='utf-8') as f:
                f.write(pretty_xml)
            
            processed_files += 1
            
        except Exception as e:
            log_with_timestamp(f"Error updating XML file {xml_file}: {e}")
    
    return processed_files

def main():
    args = parse_args()
    dataset_folder = args.dataset_folder
    
    # Construct paths using the dataset folder
    annotations_dir = os.path.join(dataset_folder, "patches_rules_expressions_unique")
    output_dir = os.path.join(dataset_folder, "patches_rules_expressions_unique_llm")
    results_file_train = os.path.join(dataset_folder, "batch_prediction/batch_inference_results_train.jsonl")
    results_file_val = os.path.join(dataset_folder, "batch_prediction/batch_inference_results_val.jsonl")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine which splits to process
    splits_to_process = ['train', 'val'] if args.split == 'both' else [args.split]
    
    # Process selected splits
    for split in splits_to_process:
        results_file = results_file_train if split == 'train' else results_file_val
        if not os.path.exists(results_file):
            log_with_timestamp(f"Results file not found: {results_file}")
            continue
            
        log_with_timestamp(f"Processing {split} split...")
        results = parse_results_file(results_file)
        if not results:
            log_with_timestamp(f"No results found in {results_file}")
            continue
            
        processed_files = update_xml_with_results(results, annotations_dir, output_dir, split)
        log_with_timestamp(f"Processed {processed_files} files for {split} split")
    
    log_with_timestamp("Processing complete!")

if __name__ == "__main__":
    main() 