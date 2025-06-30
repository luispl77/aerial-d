#!/usr/bin/env python3
import os
import random
import argparse
import json
import xml.etree.ElementTree as ET
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from transformers import pipeline

# Constants for the enhancement
NUM_ENHANCED = 3  # Number of enhanced expressions per original
NUM_UNIQUE = 2    # Number of unique expressions to generate

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Enhance aerial image annotations using Gemma 3')
    parser.add_argument('--dataset_path', type=str, default='/tmp/u035679/aerial_seg_clean/aeriald/patches/train',
                        help='Path to the dataset directory')
    parser.add_argument('--output_dir', type=str, default='./enhanced_annotations',
                        help='Directory to save enhanced annotations')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--cache_dir', type=str, default='./gemma_model',
                        help='Directory to cache the model')
    parser.add_argument('--specific_file', type=str, default=None,
                        help='Process a specific file instead of random selection (filename without extension)')
    parser.add_argument('--include_images', action='store_true',
                        help='Include images in the prompt')
    return parser.parse_args()

def get_random_file(args):
    """Get a random image file and its corresponding annotation file."""
    random.seed(args.random_seed)
    
    images_dir = os.path.join(args.dataset_path, 'images')
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.png')]
    
    random_image = random.choice(image_files)
    image_path = os.path.join(images_dir, random_image)
    
    annotation_file = random_image.replace('.png', '.xml')
    annotation_path = os.path.join(args.dataset_path, 'annotations', annotation_file)
    
    return image_path, annotation_path

def parse_annotations(annotation_path):
    """Extract annotations from XML file."""
    tree = ET.parse(annotation_path)
    root = tree.getroot()
    
    filename = root.find('filename').text
    objects = []
    
    for obj in root.findall('object'):
        # Get category
        category = None
        if obj.find('name') is not None:
            category = obj.find('name').text
        elif obj.find('n') is not None:
            category = obj.find('n').text
        else:
            category = "unknown"
        
        # Get object ID
        obj_id = obj.find('id').text if obj.find('id') is not None else "unknown"
        
        # Get bounding box
        bbox = {}
        if obj.find('bndbox') is not None:
            bbox = {
                'xmin': int(obj.find('bndbox/xmin').text),
                'ymin': int(obj.find('bndbox/ymin').text),
                'xmax': int(obj.find('bndbox/xmax').text),
                'ymax': int(obj.find('bndbox/ymax').text)
            }
        
        # Initialize expressions list
        expressions = []
        
        # Check for grid_position
        if obj.find('grid_position') is not None:
            grid_pos = obj.find('grid_position').text
            expressions.append(grid_pos)
        
        # Check for explicit expressions tag
        if obj.find('expressions') is not None:
            for expr in obj.findall('expressions/expression'):
                if expr.text is not None:
                    expressions.append(expr.text)
        
        # Check for color
        if obj.find('color') is not None and obj.find('color').text is not None:
            color = obj.find('color').text
            expressions.append(f"the {color} {category}")
        
        # Extract relationship descriptions
        if obj.find('relationships') is not None:
            for rel in obj.findall('relationships/relationship'):
                target_category = None
                direction = None
                
                if rel.find('target_category') is not None:
                    target_category = rel.find('target_category').text
                
                if rel.find('direction') is not None:
                    direction = rel.find('direction').text
                
                if target_category is not None and direction is not None:
                    relationship_expr = f"{direction} a {target_category.replace('_', ' ').lower()}"
                    expressions.append(relationship_expr)
        
        # Add to objects list if we have a bounding box and expressions
        if bbox and expressions:
            obj_dict = {
                'id': obj_id,
                'category': category,
                'bbox': bbox,
                'expressions': expressions
            }
            objects.append(obj_dict)
    
    return {
        'filename': filename,
        'objects': objects
    }

def create_prompt(object_name, original_expressions):
    """Create the detailed prompt for Gemma 3."""
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
    
    system_message = {
        "role": "system",
        "content": [
            {"type": "text", "text": prompt}
        ]
    }
    
    user_message = {
        "role": "user",
        "content": [
            {"type": "text", "text": f"Create language variations of the provided expressions while preserving spatial information, analyze the spatial context for uniqueness factors, and generate new unique expressions for this {object_name} (highlighted with a red bounding box)."}
        ]
    }
    
    return system_message, user_message

def visualize_and_save_object(image_path, bbox, output_path):
    """Visualize object with bounding box and save to file."""
    img = Image.open(image_path)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    rect = patches.Rectangle(
        (bbox['xmin'], bbox['ymin']),
        bbox['xmax'] - bbox['xmin'],
        bbox['ymax'] - bbox['ymin'],
        linewidth=2,
        edgecolor='r',
        facecolor='none'
    )
    ax.add_patch(rect)
    
    ax.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def enhance_annotations(image_path, annotation_data, pipe, output_dir, include_images=False):
    """Enhance annotations using Gemma 3 pipeline."""
    results = []
    
    for i, obj in enumerate(annotation_data['objects']):
        if not obj['expressions']:
            continue
        
        # Create visualization with bounding box
        os.makedirs(output_dir, exist_ok=True)
        viz_path = os.path.join(output_dir, f"{os.path.basename(image_path).split('.')[0]}_obj_{obj['id']}.png")
        visualize_and_save_object(image_path, obj['bbox'], viz_path)
        
        try:
            system_message, user_message = create_prompt(obj['category'], obj['expressions'])
            
            if include_images:
                # Load the image using PIL
                image = Image.open(viz_path)
                user_message["content"].insert(0, {"type": "image", "url": image})
            
            messages = [system_message, user_message]
            
            # Use pipeline for inference
            output = pipe(text=messages, max_new_tokens=500)
            
            # Extract the assistant's response from the pipeline output
            generated_text = output[0]["generated_text"]
            
            # Find the assistant's message in the generated text
            response = None
            if isinstance(generated_text, list):
                # Look for the message with role 'assistant'
                for message in generated_text:
                    if isinstance(message, dict) and message.get('role') == 'assistant':
                        response = message.get('content', '')
                        break
            
            if response is None:
                response = str(generated_text)  # Fallback to string representation
                
            print(f"\n\nExtracted response: {response[:200]}...")
            
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                enhanced_data = json.loads(json_str)
            else:
                enhanced_data = {"error": "No valid JSON found in response"}
                
        except json.JSONDecodeError:
            enhanced_data = {"error": "Invalid JSON in response", "raw_response": response}
        except Exception as e:
            enhanced_data = {"error": f"Processing error: {e}"}
        
        result = {
            "object_id": obj['id'],
            "category": obj['category'],
            "original_expressions": obj['expressions'],
            "enhanced_data": enhanced_data
        }
        results.append(result)
    
    return results

def main():
    """Main function."""
    args = parse_arguments()
    
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Create pipeline with specified GPU
    pipe = pipeline(
        "image-text-to-text",
        model="google/gemma-3-4b-it",
        device=device,
        use_fast=True,
        torch_dtype=torch.bfloat16
    )
    
    if args.specific_file:
        image_filename = f"{args.specific_file}.png"
        image_path = os.path.join(args.dataset_path, 'images', image_filename)
        annotation_path = os.path.join(args.dataset_path, 'annotations', f"{args.specific_file}.xml")
    else:
        image_path, annotation_path = get_random_file(args)
    
    annotation_data = parse_annotations(annotation_path)
    
    output_dir = os.path.join(args.output_dir, os.path.basename(image_path).split('.')[0])
    os.makedirs(output_dir, exist_ok=True)
    
    results = enhance_annotations(image_path, annotation_data, pipe, output_dir, args.include_images)
    
    output_file = os.path.join(output_dir, "enhanced_expressions.json")
    with open(output_file, 'w') as f:
        json.dump({
            "image": os.path.basename(image_path),
            "results": results
        }, f, indent=2)
    
    print(f"Enhancement completed. Results saved to {output_file}")

if __name__ == "__main__":
    main() 