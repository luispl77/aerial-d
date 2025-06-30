import os
import numpy as np
import math
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import cv2 # Still needed for contour rendering
import json # Still needed for RLE parsing
import pycocotools.mask as mask_util # Still needed for RLE decoding
import argparse # Add argparse for command line arguments

# ============================================================

MAX_DEBUG_FILES = None # Set to None to process all files

# === Constants copied from 2_add_rules.py for consistency ===
# The entire color constants section is being removed since we'll just read colors from XML
# ============================================================

# === RLE Decode function copied from 2_add_rules.py ===
def rle_to_mask(rle_string, height, width):
    """Decode RLE string (stored as JSON) into a binary mask."""
    try:
        rle = json.loads(rle_string.replace("'", '"'))
        if isinstance(rle.get('counts'), str):
            rle['counts'] = rle['counts'].encode('utf-8')
        mask = mask_util.decode(rle)
        if mask.shape != (height, width):
             print(f"Warning: RLE decode shape mismatch ({mask.shape} vs {(height, width)}). Using empty mask.")
             return np.zeros((height, width), dtype=np.uint8)
        return mask
    except Exception as e:
        # print(f"Error decoding RLE: {e}. RLE String: '{rle_string[:100]}...'") # Removed print
        return np.zeros((height, width), dtype=np.uint8)
# =======================================================

# === Color Reading Function ===
def get_color_from_xml(obj):
    """Get color attribute from XML object element if it exists."""
    color_elem = obj.find('color')
    if color_elem is not None and color_elem.text:
        return color_elem.text
    return "None"  # Default if no color information exists
# ===============================

def draw_arrow(draw, start, end, color, width=1, arrow_size=10):
    """Draw an arrow from start to end points"""
    # Draw the main line
    draw.line([start, end], fill=color, width=width)
    
    # Calculate the arrow head
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    angle = math.atan2(dy, dx)
    
    # Calculate arrow head points
    angle_left = angle + math.pi * 3/4  # 135 degrees
    angle_right = angle - math.pi * 3/4  # -135 degrees
    
    # Points for the arrow head
    x_left = end[0] + arrow_size * math.cos(angle_left)
    y_left = end[1] + arrow_size * math.sin(angle_left)
    x_right = end[0] + arrow_size * math.cos(angle_right)
    y_right = end[1] + arrow_size * math.sin(angle_right)
    
    # Draw arrow head
    draw.line([end, (x_left, y_left)], fill=color, width=width)
    draw.line([end, (x_right, y_right)], fill=color, width=width)

def visualize_annotations(xml_path, image_path, output_path):
    """Create debug visualization of positions and clustering"""
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Load image
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Try to get a font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        
        # Dictionary to store cluster colors
        cluster_colors = {}
        
        # Map object IDs to their groups
        object_to_group = {}
        groups_elem = root.find('groups')
        if groups_elem is not None:
            for group_elem in groups_elem.findall('group'):
                group_id = int(group_elem.find('id').text)
                
                # Generate color for this group
                if group_id not in cluster_colors:
                    h = (group_id * 50) % 360  # Spaced hues
                    s = 100
                    v = 100
                    
                    # Convert HSV to RGB
                    h_i = int(h / 60)
                    f = h / 60 - h_i
                    p = v * (1 - s/100)
                    q = v * (1 - f * s/100)
                    t = v * (1 - (1 - f) * s/100)
                    
                    if h_i == 0:
                        r, g, b = v, t, p
                    elif h_i == 1:
                        r, g, b = q, v, p
                    elif h_i == 2:
                        r, g, b = p, v, t
                    elif h_i == 3:
                        r, g, b = p, q, v
                    elif h_i == 4:
                        r, g, b = t, p, v
                    else:
                        r, g, b = v, p, q
                    
                    # Scale to 0-255
                    cluster_colors[group_id] = (int(r * 255/100), int(g * 255/100), int(b * 255/100))
                
                # Process instance IDs
                instance_ids_text = group_elem.find('instance_ids').text
                if instance_ids_text:
                    instance_ids = [int(id_str) for id_str in instance_ids_text.split(',')]
                    for obj_id in instance_ids:
                        object_to_group[obj_id] = group_id
                
                # Draw group centroid if available
                centroid_elem = group_elem.find('centroid')
                if centroid_elem is not None:
                    centroid_x = float(centroid_elem.find('x').text)
                    centroid_y = float(centroid_elem.find('y').text)
                    
                    # Draw a cross at the centroid
                    cross_size = 10
                    color = cluster_colors[group_id]
                    draw.line([(centroid_x - cross_size, centroid_y), 
                              (centroid_x + cross_size, centroid_y)], 
                             fill=color, width=2)
                    draw.line([(centroid_x, centroid_y - cross_size), 
                              (centroid_x, centroid_y + cross_size)], 
                             fill=color, width=2)
                    
                    # Add group size label near centroid
                    size_elem = group_elem.find('size')
                    if size_elem is not None:
                        size = int(size_elem.text)
                        draw.text((centroid_x + cross_size + 5, centroid_y - 10), 
                                f"Size: {size}", fill=color, font=font)
        
        # Draw bounding boxes and cluster information
        all_color_debug_strings = [] # List to hold debug info for the text file
        for obj in root.findall('object'):
            obj_id = int(obj.find('id').text)
            category = obj.find('name').text
            
            # Get group ID from mapping
            group_id = object_to_group.get(obj_id, -obj_id - 1)  # Default to unique negative ID
            
            # Get color for this object
            if group_id not in cluster_colors:
                # Gray for outliers/ungrouped objects
                cluster_colors[group_id] = (100, 100, 100)
            
            box_color = cluster_colors[group_id]
            
            # Get bounding box
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Draw bounding box with cluster color
                draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=2)
                
                # Base label (NO calculated color needed here)
                label = f"ID:{obj_id} {category}"
                if group_id >= 0:
                    label += f" (Group:{group_id})"
                
                # --- Color Debugging (for text file) --- 
                # We still need the log file generated here, so keep analysis part for the log string
                color_debug_str_for_file = f"ID:{obj_id}"
                
                # Get color directly from XML
                xml_color = get_color_from_xml(obj)
                
                segmentation_elem = obj.find('segmentation')
                if segmentation_elem is not None and segmentation_elem.text:
                    # Store original image state before potential modification by drawing contours
                    original_cv_image_state = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                    
                    mask = rle_to_mask(segmentation_elem.text, height, width)
                    mask_bool = mask.astype(bool)
                    
                    # Initialize variables
                    pixel_count = 0
                    image_to_convert = original_cv_image_state.copy() # Start with a copy
                    
                    if np.any(mask_bool):
                        # Count pixels in mask
                        pixel_count = int(np.sum(mask_bool))
                        
                        # Draw contours on the image
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(image_to_convert, contours, -1, (0, 255, 0), 1) # Draw green contour
                    
                    # Include pixel count and the color from XML
                    pixel_count_str = f"Pixels:{pixel_count}"
                    color_debug_str_for_file += f" {pixel_count_str} XMLColor:{xml_color} Mask:Yes"
                    
                else:
                    image_to_convert = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR) # No contour drawn
                    color_debug_str_for_file += f" Pixels:N/A XMLColor:{xml_color} Mask:No"
                
                all_color_debug_strings.append(color_debug_str_for_file)
                # -----------------------------
                
                # --- Add determined color to the image label --- 
                # REMOVED - Not needed for this visualization
                # if determined_color != "None":
                #     # label += f" CalcColor:{determined_color}" 
                #     label += f" {determined_color}" # Just add the color name
                # ---------------------------------------------

                # Convert potentially modified CV image back to PIL for drawing text
                # This ensures contours are included in the saved image
                image = Image.fromarray(cv2.cvtColor(image_to_convert, cv2.COLOR_BGR2RGB))
                draw = ImageDraw.Draw(image) # Re-acquire draw object

                # Draw text (now includes calculated color, positioned below)
                # draw.text((xmin, ymin - 15), label, fill=box_color, font=font)
                draw.text((xmin, ymax + 5), label, fill=box_color, font=font) # Position below bbox
        
        # Draw grid lines on top (keeping the grid visualization)
        grid_color = (200, 200, 200, 128)  # Semi-transparent gray
        third_w = width / 3
        third_h = height / 3
        
        # Draw vertical grid lines
        draw.line([(third_w, 0), (third_w, height)], fill=grid_color, width=1)
        draw.line([(2*third_w, 0), (2*third_w, height)], fill=grid_color, width=1)
        
        # Draw horizontal grid lines
        draw.line([(0, third_h), (width, third_h)], fill=grid_color, width=1)
        draw.line([(0, 2*third_h), (width, 2*third_h)], fill=grid_color, width=1)
        
        # Add a legend for groups
        legend_x = 10
        legend_y = 10
        for group_id, color in sorted(cluster_colors.items()):
            if group_id >= 0:  # Only show actual groups in legend
                draw.rectangle([legend_x, legend_y, legend_x + 20, legend_y + 20], fill=color)
                draw.text((legend_x + 25, legend_y + 5), f"Group {group_id}", fill=color, font=font)
                legend_y += 25
        
        # Draw group relationships
        rels_elem = groups_elem.find('relationships') if groups_elem is not None else None
        if rels_elem is not None:
            # First pass: collect all relationships
            relationships = []
            for rel in rels_elem.findall('relationship'):
                # In our XML structure, we only have source_id and target_id without types
                # Assume all relationships are between groups by default
                source_id = int(rel.find('source_id').text)
                target_id = int(rel.find('target_id').text)
                direction = rel.find('direction').text
                distance = float(rel.find('distance').text)
                relationships.append(('group', source_id, 'group', target_id, direction, distance))

            # Second pass: draw relationships with bidirectional awareness
            for rel in relationships:
                source_type, source_id, target_type, target_id, direction, distance = rel
                
                # Check if this relationship has a reverse
                has_reverse = False
                for other_rel in relationships:
                    if (other_rel[0] == target_type and other_rel[1] == target_id and 
                        other_rel[2] == source_type and other_rel[3] == source_id):
                        has_reverse = True
                        break

                # Get source centroid
                if source_type == 'group':
                    source_group = None
                    for group in groups_elem.findall('group'):
                        if int(group.find('id').text) == source_id:
                            source_group = group
                            break
                    if source_group is None:
                        continue
                    centroid_elem = source_group.find('centroid')
                    if centroid_elem is None:
                        continue
                    source_x = float(centroid_elem.find('x').text)
                    source_y = float(centroid_elem.find('y').text)
                else:  # instance
                    source_obj = None
                    for obj in root.findall('object'):
                        if int(obj.find('id').text) == source_id:
                            source_obj = obj
                            break
                    if source_obj is None:
                        continue
                    bbox = source_obj.find('bndbox')
                    if bbox is None:
                        continue
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    source_x = (xmin + xmax) / 2
                    source_y = (ymin + ymax) / 2

                # Get target centroid
                if target_type == 'group':
                    target_group = None
                    for group in groups_elem.findall('group'):
                        if int(group.find('id').text) == target_id:
                            target_group = group
                            break
                    if target_group is None:
                        continue
                    centroid_elem = target_group.find('centroid')
                    if centroid_elem is None:
                        continue
                    target_x = float(centroid_elem.find('x').text)
                    target_y = float(centroid_elem.find('y').text)
                else:  # instance
                    target_obj = None
                    for obj in root.findall('object'):
                        if int(obj.find('id').text) == target_id:
                            target_obj = obj
                            break
                    if target_obj is None:
                        continue
                    bbox = target_obj.find('bndbox')
                    if bbox is None:
                        continue
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    target_x = (xmin + xmax) / 2
                    target_y = (ymin + ymax) / 2

                # Draw relationship line with arrow
                if has_reverse:
                    # Bidirectional relationship - use solid line
                    arrow_color = (255, 0, 0, 200)  # Red with transparency
                    line_width = 2
                else:
                    # Unidirectional relationship - use dashed line
                    arrow_color = (255, 165, 0, 200)  # Orange with transparency
                    line_width = 1
                
                # Draw the line
                draw_arrow(draw, (source_x, source_y), (target_x, target_y), arrow_color, width=line_width, arrow_size=8)
                
                # Draw relationship info - position text differently based on direction
                # Calculate vector from source to target
                dx = target_x - source_x
                dy = target_y - source_y
                # Position text at 1/3 along the line from source to target
                text_x = source_x + dx * 0.33
                text_y = source_y + dy * 0.33
                # Add slight offset to avoid covering the line
                text_x += 5 if dx >= 0 else -5
                text_y += 5 if dy >= 0 else -5
                
                rel_text = f"{direction} ({distance:.1f}px)"
                draw.text((text_x, text_y), rel_text, fill=arrow_color, font=font)
        
        # Draw instance relationships
        for obj in root.findall('object'):
            obj_id = int(obj.find('id').text)
            rels_elem = obj.find('relationships')
            if rels_elem is None:
                continue
                
            # Get source centroid
            bbox = obj.find('bndbox')
            if bbox is None:
                continue
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            source_x = (xmin + xmax) / 2
            source_y = (ymin + ymax) / 2
            
            for rel in rels_elem.findall('relationship'):
                target_id = int(rel.find('target_id').text)
                direction = rel.find('direction').text
                distance = float(rel.find('distance').text)
                
                # Find target object
                target_obj = None
                for target in root.findall('object'):
                    if int(target.find('id').text) == target_id:
                        target_obj = target
                        break
                
                if target_obj is None:
                    continue
                
                # Get target centroid
                bbox = target_obj.find('bndbox')
                if bbox is None:
                    continue
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                target_x = (xmin + xmax) / 2
                target_y = (ymin + ymax) / 2
                
                # Draw relationship line with arrow
                arrow_color = (0, 0, 255, 150)  # Blue with transparency
                draw_arrow(draw, (source_x, source_y), (target_x, target_y), arrow_color, width=1, arrow_size=5)
                
                # Draw relationship info (only for a few relationships to avoid cluttering)
                if distance < 100:  # Only show text for close relationships
                    mid_x = (source_x + target_x) / 2
                    mid_y = (source_y + target_y) / 2
                    rel_text = f"{direction} ({distance:.1f}px)"
                    draw.text((mid_x + 5, mid_y), rel_text, fill=arrow_color, font=font)
        
        # Draw no-man's land region
        alpha = 0.2  # Same alpha value used in the rule calculation
        no_mans_land_width = alpha * width
        center_inner_left = width/2 - no_mans_land_width/2
        center_inner_right = width/2 + no_mans_land_width/2
        center_inner_top = height/2 - no_mans_land_width/2
        center_inner_bottom = height/2 + no_mans_land_width/2
        
        # Define the corners of the no-man's land rectangle
        inner_rect = [center_inner_left, center_inner_top, center_inner_right, center_inner_bottom]
        
        # Draw inner rectangle with dashed line
        dash_length = 5
        dash_gap = 5
        dash_color = (255, 255, 0)  # Yellow
        
        # Draw dashed horizontal lines
        for x in range(int(inner_rect[0]), int(inner_rect[2]), dash_length + dash_gap):
            draw.line([(x, inner_rect[1]), (min(x + dash_length, inner_rect[2]), inner_rect[1])], fill=dash_color, width=1)
            draw.line([(x, inner_rect[3]), (min(x + dash_length, inner_rect[2]), inner_rect[3])], fill=dash_color, width=1)
        
        # Draw dashed vertical lines
        for y in range(int(inner_rect[1]), int(inner_rect[3]), dash_length + dash_gap):
            draw.line([(inner_rect[0], y), (inner_rect[0], min(y + dash_length, inner_rect[3]))], fill=dash_color, width=1)
            draw.line([(inner_rect[2], y), (inner_rect[2], min(y + dash_length, inner_rect[3]))], fill=dash_color, width=1)
        
        # Save image
        image.save(output_path)
        return True, all_color_debug_strings # Return success and the list of debug strings
        
    except Exception as e:
        # print(f"Error creating visualization: {e}") # Removed print
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        return False, [] # Return failure and empty list

def generate_visualizations(xml_paths, image_paths, output_paths):
    """Generate visualizations for multiple images"""
    for xml_path, image_path, output_path in tqdm(zip(xml_paths, image_paths, output_paths), 
                                                 total=len(xml_paths),
                                                 desc="Generating visualizations"):
        if os.path.exists(image_path):
            success, color_debug_strings = visualize_annotations(xml_path, image_path, output_path)
            if success:
                # Write the debug strings to a corresponding text file
                txt_output_path = os.path.splitext(output_path)[0] + ".txt"
                try:
                    with open(txt_output_path, 'w') as f:
                        if color_debug_strings:
                            f.write("\n".join(color_debug_strings))
                        else:
                            f.write("No color debug information generated.")
                except Exception as e:
                    # print(f"Error writing debug text file {txt_output_path}: {e}") # Removed print
                    pass # Silently ignore write error
            else:
                # print(f"Warning: Failed to create visualization for {os.path.basename(image_path)}") # Removed print
                pass # Silently ignore creation failure
        else:
            # print(f"Warning: Image file not found: {image_path}") # Removed print
            pass # Silently ignore missing image

def visualize_group_relationships(xml_path, image_path, output_path):
    """Create debug visualization focusing only on group relationships"""
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Load image
    try:
        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Try to get a font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        
        # Dictionary to store cluster colors
        cluster_colors = {}
        
        # Map object IDs to their groups
        object_to_group = {}
        groups_elem = root.find('groups')
        if groups_elem is not None:
            for group_elem in groups_elem.findall('group'):
                group_id = int(group_elem.find('id').text)
                
                # Generate color for this group
                if group_id not in cluster_colors:
                    h = (group_id * 50) % 360  # Spaced hues
                    s = 100
                    v = 100
                    
                    # Convert HSV to RGB
                    h_i = int(h / 60)
                    f = h / 60 - h_i
                    p = v * (1 - s/100)
                    q = v * (1 - f * s/100)
                    t = v * (1 - (1 - f) * s/100)
                    
                    if h_i == 0:
                        r, g, b = v, t, p
                    elif h_i == 1:
                        r, g, b = q, v, p
                    elif h_i == 2:
                        r, g, b = p, v, t
                    elif h_i == 3:
                        r, g, b = p, q, v
                    elif h_i == 4:
                        r, g, b = t, p, v
                    else:
                        r, g, b = v, p, q
                    
                    # Scale to 0-255
                    cluster_colors[group_id] = (int(r * 255/100), int(g * 255/100), int(b * 255/100))
                
                # Process instance IDs
                instance_ids_text = group_elem.find('instance_ids').text
                if instance_ids_text:
                    instance_ids = [int(id_str) for id_str in instance_ids_text.split(',')]
                    for obj_id in instance_ids:
                        object_to_group[obj_id] = group_id
                
                # Draw group centroid if available
                centroid_elem = group_elem.find('centroid')
                if centroid_elem is not None:
                    centroid_x = float(centroid_elem.find('x').text)
                    centroid_y = float(centroid_elem.find('y').text)
                    
                    # Draw a cross at the centroid
                    cross_size = 10
                    color = cluster_colors[group_id]
                    draw.line([(centroid_x - cross_size, centroid_y), 
                              (centroid_x + cross_size, centroid_y)], 
                             fill=color, width=2)
                    draw.line([(centroid_x, centroid_y - cross_size), 
                              (centroid_x, centroid_y + cross_size)], 
                             fill=color, width=2)
                    
                    # Add group size label near centroid
                    size_elem = group_elem.find('size')
                    if size_elem is not None:
                        size = int(size_elem.text)
                        draw.text((centroid_x + cross_size + 5, centroid_y - 10), 
                                f"Size: {size}", fill=color, font=font)
        
        # Draw bounding boxes and cluster information
        for obj in root.findall('object'):
            obj_id = int(obj.find('id').text)
            category = obj.find('name').text
            
            # Get group ID from mapping
            group_id = object_to_group.get(obj_id, -obj_id - 1)  # Default to unique negative ID
            
            # Get color for this object
            if group_id not in cluster_colors:
                # Gray for outliers/ungrouped objects
                cluster_colors[group_id] = (100, 100, 100)
            
            box_color = cluster_colors[group_id]
            
            # Get bounding box
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Draw bounding box with cluster color
                draw.rectangle([xmin, ymin, xmax, ymax], outline=box_color, width=2)
                
                # Draw ID, category and cluster info
                label = f"ID:{obj_id} {category}"
                if group_id >= 0:
                    label += f" (Group:{group_id})"
                
                # --- Add Color Information --- 
                color_elem = obj.find('color')
                if color_elem is not None and color_elem.text:
                    label += f" Color:{color_elem.text}"
                # -----------------------------
                
                draw.text((xmin, ymin-15), label, fill=box_color, font=font)
        
        # Add a legend for groups
        legend_x = 10
        legend_y = 10
        for group_id, color in sorted(cluster_colors.items()):
            if group_id >= 0:  # Only show actual groups in legend
                draw.rectangle([legend_x, legend_y, legend_x + 20, legend_y + 20], fill=color)
                draw.text((legend_x + 25, legend_y + 5), f"Group {group_id}", fill=color, font=font)
                legend_y += 25
        
        # Draw group relationships
        rels_elem = groups_elem.find('relationships') if groups_elem is not None else None
        if rels_elem is not None:
            # First pass: collect all relationships
            relationships = []
            for rel in rels_elem.findall('relationship'):
                # In our XML structure, we only have source_id and target_id without types
                # Assume all relationships are between groups by default
                source_id = int(rel.find('source_id').text)
                target_id = int(rel.find('target_id').text)
                direction = rel.find('direction').text
                distance = float(rel.find('distance').text)
                relationships.append(('group', source_id, 'group', target_id, direction, distance))

            # Second pass: draw relationships with bidirectional awareness
            for rel in relationships:
                source_type, source_id, target_type, target_id, direction, distance = rel
                
                # Check if this relationship has a reverse
                has_reverse = False
                for other_rel in relationships:
                    if (other_rel[0] == target_type and other_rel[1] == target_id and 
                        other_rel[2] == source_type and other_rel[3] == source_id):
                        has_reverse = True
                        break

                # Get source centroid
                if source_type == 'group':
                    source_group = None
                    for group in groups_elem.findall('group'):
                        if int(group.find('id').text) == source_id:
                            source_group = group
                            break
                    if source_group is None:
                        continue
                    centroid_elem = source_group.find('centroid')
                    if centroid_elem is None:
                        continue
                    source_x = float(centroid_elem.find('x').text)
                    source_y = float(centroid_elem.find('y').text)
                else:  # instance
                    source_obj = None
                    for obj in root.findall('object'):
                        if int(obj.find('id').text) == source_id:
                            source_obj = obj
                            break
                    if source_obj is None:
                        continue
                    bbox = source_obj.find('bndbox')
                    if bbox is None:
                        continue
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    source_x = (xmin + xmax) / 2
                    source_y = (ymin + ymax) / 2

                # Get target centroid
                if target_type == 'group':
                    target_group = None
                    for group in groups_elem.findall('group'):
                        if int(group.find('id').text) == target_id:
                            target_group = group
                            break
                    if target_group is None:
                        continue
                    centroid_elem = target_group.find('centroid')
                    if centroid_elem is None:
                        continue
                    target_x = float(centroid_elem.find('x').text)
                    target_y = float(centroid_elem.find('y').text)
                else:  # instance
                    target_obj = None
                    for obj in root.findall('object'):
                        if int(obj.find('id').text) == target_id:
                            target_obj = obj
                            break
                    if target_obj is None:
                        continue
                    bbox = target_obj.find('bndbox')
                    if bbox is None:
                        continue
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    target_x = (xmin + xmax) / 2
                    target_y = (ymin + ymax) / 2

                # Draw relationship line with arrow
                if has_reverse:
                    # Bidirectional relationship - use solid line
                    arrow_color = (255, 0, 0, 200)  # Red with transparency
                    line_width = 2
                else:
                    # Unidirectional relationship - use dashed line
                    arrow_color = (255, 165, 0, 200)  # Orange with transparency
                    line_width = 1
                
                # Draw the line
                draw_arrow(draw, (source_x, source_y), (target_x, target_y), arrow_color, width=line_width, arrow_size=8)
                
                # Draw relationship info - position text differently based on direction
                # Calculate vector from source to target
                dx = target_x - source_x
                dy = target_y - source_y
                # Position text at 1/3 along the line from source to target
                text_x = source_x + dx * 0.33
                text_y = source_y + dy * 0.33
                # Add slight offset to avoid covering the line
                text_x += 5 if dx >= 0 else -5
                text_y += 5 if dy >= 0 else -5
                
                rel_text = f"{direction} ({distance:.1f}px)"
                draw.text((text_x, text_y), rel_text, fill=arrow_color, font=font)
        
        # Save image
        image.save(output_path)
        return True
        
    except Exception as e:
        # print(f"Error creating group visualization: {e}") # Removed print
        return False

def generate_group_visualizations(xml_paths, image_paths, output_paths):
    """Generate group relationship visualizations for multiple images"""
    for xml_path, image_path, output_path in tqdm(zip(xml_paths, image_paths, output_paths), 
                                                 total=len(xml_paths),
                                                 desc="Generating group visualizations"):
        if os.path.exists(image_path):
            success = visualize_group_relationships(xml_path, image_path, output_path)
            if not success:
                # print(f"Warning: Failed to create group visualization for {os.path.basename(image_path)}") # Removed print
                pass # Silently ignore creation failure
        else:
            # print(f"Warning: Image file not found: {image_path}") # Removed print
            pass # Silently ignore missing image

# === New Function for ID-Only Visualization ===
def visualize_id_only(xml_path, image_path, output_path):
    """Create debug visualization showing only ID, mask outline, and color"""
    try:
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Load image using PIL
        image = Image.open(image_path).convert("RGB") # Ensure RGB
        draw = ImageDraw.Draw(image)
        width, height = image.size
        
        # Try to get a font
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except IOError:
            font = ImageFont.load_default()
        
        # Process each object
        for obj in root.findall('object'):
            obj_id = int(obj.find('id').text)
            
            # Determine color for bbox and text (e.g., white -> yellow)
            draw_color = (255, 255, 0) # Yellow for better visibility
            mask_contour_color = (0, 255, 0) # Green for mask outline
            
            # Get bounding box coordinates (still needed for label placement)
            bbox = obj.find('bndbox')
            if bbox is not None:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                
                # Ensure label is reset for each object within the bbox check
                label = f"{obj_id}" # Removed "ID:"

                # Handle mask drawing if segmentation exists
                segmentation_elem = obj.find('segmentation')
                image_array_for_cv = np.array(image)
                cv_image = cv2.cvtColor(image_array_for_cv, cv2.COLOR_RGB2BGR)
                needs_reconversion = False
                
                if segmentation_elem is not None and segmentation_elem.text:
                    mask = rle_to_mask(segmentation_elem.text, height, width)
                    mask_bool = mask.astype(bool)
                    
                    if np.any(mask_bool):
                        # Draw contours
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(cv_image, contours, -1, mask_contour_color, 1) # Draw on CV image
                        needs_reconversion = True
                
                # If mask was drawn, convert back to PIL
                if needs_reconversion:
                    image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
                    draw = ImageDraw.Draw(image) # Re-acquire draw object
                
                # Draw the ID label BELOW (as white)
                id_label_color = (255, 255, 255) # White for ID
                draw.text((xmin, ymax + 5), label, fill=id_label_color, font=font)
                
                # Get color from XML
                xml_color = get_color_from_xml(obj)
                
                # Only draw color if it's not "None"
                if xml_color != "None":
                    # Draw the color label ABOVE in yellow
                    draw.text((xmin, ymin - 15), xml_color, fill=draw_color, font=font)

         # Save the simplified image
        image.save(output_path)
        return True
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return False
# ===========================================

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate debug visualizations for annotations')
    parser.add_argument('--start', type=int, default=0, help='Starting file index (0-based)')
    parser.add_argument('--end', type=int, default=None, help='Ending file index (exclusive, None for all)')
    parser.add_argument('--limit', type=int, default=None, help='Maximum number of files to process')
    parser.add_argument('--xml-dir', type=str, default="dataset/patches_rules/annotations", help='Directory containing XML files')
    parser.add_argument('--img-dir', type=str, default="dataset/patches/images", help='Directory containing images')
    parser.add_argument('--output-dir', type=str, default="debug/patches_rules/debug_visualizations", help='Output directory for debug files')
    parser.add_argument('--group-dir', type=str, default="debug/patches_rules/group_visualizations", help='Output directory for group visualizations')
    parser.add_argument('--id-dir', type=str, default="debug/patches_rules/id_only_visualizations", help='Output directory for ID-only visualizations')
    args = parser.parse_args()
    
    # Use parsed arguments
    input_dir = args.xml_dir
    images_dir = args.img_dir
    output_dir = args.output_dir
    group_output_dir = args.group_dir
    id_only_output_dir = args.id_dir
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(group_output_dir, exist_ok=True)
    os.makedirs(id_only_output_dir, exist_ok=True)
    
    # Get list of annotation files
    annotation_files = [f for f in os.listdir(input_dir) if f.endswith('.xml')]
    annotation_files.sort() # Ensure alphabetical order
    
    # Limit number of files based on arguments
    total_files = len(annotation_files)
    start_idx = min(args.start, total_files)
    
    if args.end is not None:
        end_idx = min(args.end, total_files)
    else:
        end_idx = total_files
    
    if args.limit is not None:
        end_idx = min(start_idx + args.limit, end_idx)
    
    annotation_files = annotation_files[start_idx:end_idx]
    
    if not annotation_files:
        print("No XML files found to process.")
        return
        
    print(f"Processing {len(annotation_files)} files (range {start_idx} to {end_idx-1} out of {total_files})...")
    
    # Prepare paths for visualization
    xml_paths = []
    image_paths = []
    output_paths = []
    group_output_paths = []
    id_only_output_paths = []
    
    for xml_file in annotation_files:
        xml_path = os.path.join(input_dir, xml_file)
        
        # Get corresponding image filename from XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            image_filename_elem = root.find('filename')
            if image_filename_elem is None or not image_filename_elem.text:
                 continue
            image_filename = image_filename_elem.text
        except ET.ParseError:
             continue
        
        image_path = os.path.join(images_dir, image_filename)
        output_path = os.path.join(output_dir, f"vis_{os.path.splitext(image_filename)[0]}.png")
        group_output_path = os.path.join(group_output_dir, f"group_vis_{os.path.splitext(image_filename)[0]}.png")
        id_only_output_path = os.path.join(id_only_output_dir, f"id_vis_{os.path.splitext(image_filename)[0]}.png")
        
        xml_paths.append(xml_path)
        image_paths.append(image_path)
        output_paths.append(output_path)
        group_output_paths.append(group_output_path)
        id_only_output_paths.append(id_only_output_path)
    
    # Generate all visualizations (and debug text files)
    print(f"Generating detailed visualizations and text logs...")
    generate_visualizations(xml_paths, image_paths, output_paths)
    print(f"Detailed visualizations saved to {output_dir}")
    
    # Generate group visualizations
    print(f"Generating group visualizations...")
    generate_group_visualizations(xml_paths, image_paths, group_output_paths)
    print(f"Group visualizations saved to {group_output_dir}")
    
    # Generate ID-only visualizations
    print(f"Generating ID-only visualizations...")
    generate_id_only_visualizations(xml_paths, image_paths, id_only_output_paths)
    print(f"ID-only visualizations saved to {id_only_output_dir}")

# === New Function to Generate ID-Only Visualizations ===
def generate_id_only_visualizations(xml_paths, image_paths, output_paths):
    """Generate ID-only visualizations for multiple images"""
    for xml_path, image_path, output_path in tqdm(zip(xml_paths, image_paths, output_paths),
                                                 total=len(xml_paths),
                                                 desc="Generating ID-only visualizations"):
        if os.path.exists(image_path):
            success = visualize_id_only(xml_path, image_path, output_path)
            if not success:
                # Error is printed within visualize_id_only
                pass 
        else:
            # print(f"Warning: Image file not found for ID-only vis: {image_path}") # Removed print
            pass # Silently ignore missing image
# ======================================================

if __name__ == '__main__':
    main() 