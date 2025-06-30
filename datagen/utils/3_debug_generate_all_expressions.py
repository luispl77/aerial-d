import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys

def create_visualization(xml_path, image_path, output_path, txt_path):
    """Create a visualization of the image with bounding boxes and raw expressions"""
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Try to get a font
    try:
        font = ImageFont.truetype("arial.ttf", 14)
        small_font = ImageFont.truetype("arial.ttf", 12)
    except IOError:
        font = ImageFont.load_default()
        small_font = font
    
    # Generate unique colors for each instance
    instances = []
    for obj in root.findall('object'):
        obj_id = int(obj.find('id').text)
        category = obj.find('name').text
        
        # Get expressions - try different possible locations in XML
        expressions = []
        
        # Try expressions element first
        expressions_elem = obj.find('expressions')
        if expressions_elem is not None:
            for expr_elem in expressions_elem.findall('expression'):
                expressions.append(expr_elem.text)
        
        # If no expressions found, try raw_expressions
        if not expressions:
            raw_expr_elem = obj.find('raw_expressions')
            if raw_expr_elem is not None:
                for expr_elem in raw_expr_elem.findall('expression'):
                    expressions.append(expr_elem.text)
        
        # Get bounding box
        bbox = obj.find('bndbox')
        if bbox is None:
            continue
            
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        instances.append({
            'id': obj_id,
            'category': category,
            'expressions': expressions,
            'bbox': [xmin, ymin, xmax, ymax]
        })
    
    # Generate colors based on hue to ensure they're distinct
    colors = {}
    hex_colors = {}
    for i, instance in enumerate(instances):
        # Generate HSV color with good saturation and value for visibility
        hue = (i * 0.618033988749895) % 1.0  # golden ratio conjugate ensures good distribution
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        color = (int(r*255), int(g*255), int(b*255))
        hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        colors[instance['id']] = color
        hex_colors[instance['id']] = hex_color
    
    # Draw bounding boxes and write expressions
    for instance in instances:
        obj_id = instance['id']
        bbox = instance['bbox']
        color = colors[obj_id]
        
        # Draw bounding box
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=2)
        
        # Draw ID and category
        id_text = f"ID:{obj_id} {instance['category']}"
        draw.text((bbox[0], bbox[1]-15), id_text, fill=color, font=font)
        
        # Draw number of expressions if any
        if instance['expressions']:
            expr_count = len(instance['expressions'])
            draw.text((bbox[0], bbox[1]-30), f"Expressions: {expr_count}", fill=color, font=small_font)
    
    # Save the visualization image
    image.save(output_path)
    
    # Create text file with color-coded expressions
    with open(txt_path, 'w') as f:
        f.write("Raw Referring Expressions:\n\n")
        
        for instance in sorted(instances, key=lambda x: x['id']):
            obj_id = instance['id']
            hex_color = hex_colors[obj_id]
            
            f.write(f"[ID: {obj_id}] {hex_color} {instance['category']}\n")
            
            if instance['expressions']:
                for i, expr in enumerate(instance['expressions']):
                    f.write(f"  {i+1}. \"{expr}\"\n")
            else:
                f.write("  No expressions found\n")
            
            f.write("\n")  # Add blank line between instances

def main():
    # Hardcoded parameters
    input_dir = "dataset/patches_rules_expressions/annotations"  # Directory with XML files
    images_dir = "dataset/patches/images"  # Directory with original images
    output_dir = "debug/patches_rules_expressions/visualizations"  # Directory for visualizations
    texts_dir = "debug/patches_rules_expressions/expressions"  # Directory for text files
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(texts_dir, exist_ok=True)
    
    # Get list of annotation files
    annotation_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    
    # Prepare paths for visualization
    xml_paths = []
    image_paths = []
    output_paths = []
    text_paths = []
    
    for xml_file in annotation_files:
        xml_path = os.path.join(input_dir, xml_file)
        
        # Get corresponding image filename from XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_filename = root.find('filename').text
        
        image_path = os.path.join(images_dir, image_filename)
        output_path = os.path.join(output_dir, f"vis_{os.path.splitext(image_filename)[0]}.png")
        text_path = os.path.join(texts_dir, f"raw_expr_{os.path.splitext(image_filename)[0]}.txt")
        
        xml_paths.append(xml_path)
        image_paths.append(image_path)
        output_paths.append(output_path)
        text_paths.append(text_path)
    
    # Generate visualizations
    print(f"Generating visualizations for {len(annotation_files)} files...")
    for xml_path, image_path, output_path, text_path in tqdm(zip(xml_paths, image_paths, output_paths, text_paths), 
                                                           total=len(annotation_files),
                                                           desc="Generating visualizations"):
        if os.path.exists(image_path):
            try:
                create_visualization(xml_path, image_path, output_path, text_path)
            except Exception as e:
                print(f"Warning: Failed to create visualization for {os.path.basename(image_path)}: {e}")
        else:
            print(f"Warning: Image file not found: {image_path}")
    
    print(f"Visualizations saved to {output_dir}")
    print(f"Expression text files saved to {texts_dir}")

if __name__ == '__main__':
    main() 