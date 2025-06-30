import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
from PIL import Image, ImageDraw, ImageFont
import colorsys

def standardize_expression(expr):
    """Standardize class names inside expressions"""
    for original, standard in {
        'Large_Vehicle': 'large vehicle',
        'Small_Vehicle': 'small vehicle',
        'storage_tank': 'storage tank',
        'plane': 'plane',
        'ship': 'ship',
        'Swimming_pool': 'swimming pool',
        'Harbor': 'harbor',
        'tennis_court': 'tennis court',
        'Ground_Track_Field': 'ground track field',
        'Soccer_ball_field': 'soccer ball field',
        'baseball_diamond': 'baseball diamond',
        'Bridge': 'bridge',
        'basketball_court': 'basketball court',
        'Roundabout': 'roundabout',
        'Helicopter': 'helicopter'
    }.items():
        expr = expr.replace(original, standard)
    return expr.lower()

def create_debug_visualization(xml_path, image_path, output_path, txt_path):
    """Create visualization showing filtered expressions"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Load image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    
    # Extract instance information
    instances = []
    for obj in root.findall('object'):
        obj_id = int(obj.find('id').text)
        category = obj.find('name').text
        
        # Get expressions
        expressions = []
        expr_elem = obj.find('expressions')
        if expr_elem is not None:
            for expr in expr_elem.findall('expression'):
                expressions.append(expr.text)
        
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
    
    # Extract group information
    groups = []
    groups_elem = root.find('groups')
    if groups_elem is not None:
        for group in groups_elem.findall('group'):
            group_id = int(group.find('id').text)
            category = group.find('category').text if group.find('category') is not None else "Unknown"
            size = int(group.find('size').text) if group.find('size') is not None else 0
            
            # Get instance IDs in this group
            instance_ids_elem = group.find('instance_ids')
            instance_ids = []
            if instance_ids_elem is not None and instance_ids_elem.text:
                instance_ids = [int(id_str) for id_str in instance_ids_elem.text.split(',')]
            
            # Get centroid if available
            centroid = None
            centroid_elem = group.find('centroid')
            if centroid_elem is not None:
                x_elem = centroid_elem.find('x')
                y_elem = centroid_elem.find('y')
                if x_elem is not None and y_elem is not None:
                    centroid = (float(x_elem.text), float(y_elem.text))
            
            # Get expressions
            expressions = []
            expr_elem = group.find('expressions')
            if expr_elem is not None:
                for expr in expr_elem.findall('expression'):
                    expressions.append(expr.text)
            
            groups.append({
                'id': group_id,
                'category': category,
                'size': size,
                'instance_ids': instance_ids,
                'centroid': centroid,
                'expressions': expressions
            })
    
    # Generate colors
    colors = {}
    hex_colors = {}
    for i, instance in enumerate(instances):
        hue = (i * 0.618033988749895) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        color = (int(r*255), int(g*255), int(b*255))
        hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        colors[instance['id']] = color
        hex_colors[instance['id']] = hex_color
    
    # Generate colors for groups
    group_colors = {}
    group_hex_colors = {}
    for i, group in enumerate(groups):
        hue = ((i * 0.618033988749895) + 0.5) % 1.0  # Different offset for groups
        r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
        color = (int(r*255), int(g*255), int(b*255))
        hex_color = f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"
        group_colors[group['id']] = color
        group_hex_colors[group['id']] = hex_color
    
    # Draw bounding boxes only for instances with expressions
    for instance in instances:
        obj_id = instance['id']
        bbox = instance['bbox']
        color = colors[obj_id]
        
        # Only draw boxes for instances that have at least one expression
        if instance['expressions']:
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=color, width=2)
            draw.text((bbox[0], bbox[1]-15), f"ID:{obj_id}", fill=color, font=font)
        else:
            # Optional: Draw a light gray box for instances without expressions
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline=(200, 200, 200), width=1)
    
    # Draw group indicators
    for group in groups:
        if group['centroid'] and group['expressions']:
            color = group_colors[group['id']]
            cx, cy = group['centroid']
            
            # Draw group marker (circle)
            radius = 20
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), outline=color, width=2)
            draw.text((cx-10, cy-10), f"G:{group['id']}", fill=color, font=font)
            
            # Draw lines from centroid to instances (optional)
            for instance_id in group['instance_ids']:
                # Find the instance
                instance = next((inst for inst in instances if inst['id'] == instance_id), None)
                if instance:
                    # Draw a line from centroid to instance center
                    bbox = instance['bbox']
                    instance_center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
                    draw.line([cx, cy, instance_center[0], instance_center[1]], fill=color, width=1)
    
    # Save visualization
    image.save(output_path)
    
    # Create text report
    with open(txt_path, 'w') as f:
        f.write("Filtered Expressions Report:\n\n")
        
        # Report for individual instances
        f.write("INSTANCES:\n")
        f.write("==========\n\n")
        for instance in sorted(instances, key=lambda x: x['id']):
            obj_id = instance['id']
            hex_color = hex_colors[obj_id]
            
            # Indicate in the report whether this instance has expressions
            if instance['expressions']:
                f.write(f"[ID: {obj_id}] {hex_color} {instance['category']} ✓\n")
            else:
                f.write(f"[ID: {obj_id}] (No expressions) {instance['category']}\n")
            
            f.write("Expressions:\n")
            if instance['expressions']:
                for i, expr in enumerate(instance['expressions']):
                    f.write(f"  {i+1}. \"{expr}\"\n")
            else:
                f.write("  None\n")
            
            f.write("\n")
        
        # Report for groups
        if groups:
            f.write("\nGROUPS:\n")
            f.write("=======\n\n")
            for group in sorted(groups, key=lambda x: x['id']):
                group_id = group['id']
                hex_color = group_hex_colors[group_id]
                
                # Indicate in the report whether this group has expressions
                if group['expressions']:
                    f.write(f"[Group ID: {group_id}] {hex_color} {group['category']} (Size: {group['size']}) ✓\n")
                else:
                    f.write(f"[Group ID: {group_id}] (No expressions) {group['category']} (Size: {group['size']})\n")
                
                # List instances in this group
                f.write(f"Instances: {', '.join(map(str, group['instance_ids']))}\n")
                
                f.write("Expressions:\n")
                if group['expressions']:
                    for i, expr in enumerate(group['expressions']):
                        f.write(f"  {i+1}. \"{expr}\"\n")
                else:
                    f.write("  None\n")
                
                f.write("\n")

def main():
    # Hardcoded parameters
    input_dir = "dataset/patches_rules_expressions_unique/annotations"  # Directory with XML files
    images_dir = "dataset/patches/images"  # Directory with original images
    output_dir = "debug/patches_rules_expressions_unique/visualizations"  # Directory for visualizations
    texts_dir = "debug/patches_rules_expressions_unique/expressions"  # Directory for text files
    
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
        text_path = os.path.join(texts_dir, f"filter_report_{os.path.splitext(image_filename)[0]}.txt")
        
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
                create_debug_visualization(xml_path, image_path, output_path, text_path)
            except Exception as e:
                print(f"Warning: Failed to create visualization for {os.path.basename(image_path)}: {e}")
        else:
            print(f"Warning: Image file not found: {image_path}")
    
    print(f"Visualizations saved to {output_dir}")
    print(f"Filter reports saved to {texts_dir}")

if __name__ == '__main__':
    main() 