import os
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
from sklearn.cluster import DBSCAN
import json
import pycocotools.mask as mask_util
import multiprocessing
from multiprocessing import Pool, Value, Manager
import time

# === Color Constants ===
HUE_RANGES = {
    "red": ([0, 10], [170, 179]),
    "orange": ([11, 20],),
    "yellow": ([20, 34],),
    "green": ([35, 85],),
    "cyan": ([86, 100],),
    "blue": ([101, 130],),
    "purple": ([131, 145],),
    "magenta": ([146, 169],),
}
# HSV Ranges: H: 0-179, S: 0-255, V: 0-255
# Thresholds on a 0-100 scale for S and V
ACHROMATIC_SATURATION_THRESHOLD_100 = 25
ACHROMATIC_LIGHT_DARK_THRESHOLD_V_100 = 54
MIN_PIXELS_FOR_COLOR = 100
# Dominance Thresholds (as fractions)
ACHROMATIC_DOMINANCE_THRESHOLD = 0.70
CHROMATIC_DOMINANCE_THRESHOLD = 0.70
SINGLE_HUE_MIN_PERC = 60.0
# ===============================

def get_instance_info(xml_root):
    """Extract instance information from XML annotation, including cutoff flag"""
    instances = []
    image_width = int(xml_root.find('size/width').text)
    image_height = int(xml_root.find('size/height').text)
    
    for obj in xml_root.findall('object'):
        obj_id = int(obj.find('id').text)
        category = obj.find('name').text
        
        bbox = obj.find('bndbox')
        if bbox is not None:
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            centroid_x = (xmin + xmax) / 2
            centroid_y = (ymin + ymax) / 2
            
            width = xmax - xmin
            height = ymax - ymin
            area = width * height
            
            # Check cutoff flag
            is_cutoff_elem = obj.find('is_cutoff')
            is_cutoff = False
            if is_cutoff_elem is not None and is_cutoff_elem.text.lower() == 'true':
                is_cutoff = True
            
            # Get segmentation RLE if available
            segmentation = None
            segmentation_elem = obj.find('segmentation')
            if segmentation_elem is not None and segmentation_elem.text:
                segmentation = segmentation_elem.text
            
            instances.append({
                'id': obj_id,
                'category': category,
                'centroid': (centroid_x, centroid_y),
                'bbox': [xmin, ymin, xmax, ymax],
                'width': width,
                'height': height,
                'area': area,
                'is_cutoff': is_cutoff,
                'segmentation': segmentation,  # Add segmentation data
                'obj_element': obj  # Store reference to XML element
            })
    
    return instances, image_width, image_height

def rle_to_mask(rle_string, height, width, bbox=None):
    """Decode RLE string (stored as JSON) into a binary mask."""
    if not rle_string:
        return np.zeros((height, width), dtype=np.uint8)
    try:
        # Attempt to fix common JSON issues before parsing
        corrected_rle_string = rle_string.replace("'", '"')
        rle = json.loads(corrected_rle_string)

        # Ensure 'counts' is bytes for pycocotools
        if isinstance(rle.get('counts'), str):
            rle['counts'] = rle['counts'].encode('utf-8')

        # Decode the full mask first
        full_mask = mask_util.decode(rle)
        
        # If bbox is provided, crop the mask to the bbox
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            mask = full_mask[ymin:ymax, xmin:xmax]
        else:
            mask = full_mask

        # Final shape check
        if mask.shape != (height, width):
            print(f"Warning: RLE decode shape mismatch ({mask.shape} vs {(height, width)}). Using empty mask.")
            return np.zeros((height, width), dtype=np.uint8)
            
        return mask
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from RLE string: {e}")
        print(f"Problematic RLE string (start): {rle_string[:100]}...")
        return np.zeros((height, width), dtype=np.uint8)
    except Exception as e:
        print(f"Error decoding RLE: {e}. RLE String (start): '{rle_string[:100]}...'")
        return np.zeros((height, width), dtype=np.uint8)

def mask_to_rle(mask):
    """Convert binary mask to RLE format"""
    rle = mask_util.encode(np.asfortranarray(mask))
    # Convert binary data to string representation
    rle_string = {'size': rle['size'], 'counts': rle['counts'].decode('utf-8')}
    return rle_string

def determine_instance_color(instance_pixels_hsv):
    """Analyzes HSV pixels of an instance to determine its dominant color (light, dark, or specific hue)."""
    total_masked_pixels = instance_pixels_hsv.shape[0]
    determined_color = "None"
    is_ambiguous = False
    possible_colors = []
    
    if total_masked_pixels == 0:
        return determined_color, True, ["light", "dark"]  # Mark as ambiguous if no pixels

    if total_masked_pixels < MIN_PIXELS_FOR_COLOR:
        return determined_color, True, ["light", "dark"]  # Mark as ambiguous if too few pixels

    # --- Extract H, S, V and scale S, V to 0-100 --- 
    h_values = instance_pixels_hsv[:, 0]
    s_values_100 = instance_pixels_hsv[:, 1] * (100.0 / 255.0)
    v_values_100 = instance_pixels_hsv[:, 2] * (100.0 / 255.0)
    
    # --- Classify each pixel --- 
    is_achromatic = s_values_100 < ACHROMATIC_SATURATION_THRESHOLD_100
    is_light = is_achromatic & (v_values_100 >= ACHROMATIC_LIGHT_DARK_THRESHOLD_V_100)
    is_dark = is_achromatic & (v_values_100 < ACHROMATIC_LIGHT_DARK_THRESHOLD_V_100)
    is_chromatic = ~is_achromatic
    
    # --- Calculate counts and percentages --- 
    light_pixels = np.sum(is_light)
    dark_pixels = np.sum(is_dark)
    chromatic_pixels = np.sum(is_chromatic)

    light_perc = (light_pixels / total_masked_pixels) * 100
    dark_perc = (dark_pixels / total_masked_pixels) * 100
    chromatic_perc = (chromatic_pixels / total_masked_pixels) * 100
    
    # --- Determine final color based on dominance --- 
    if light_perc >= ACHROMATIC_DOMINANCE_THRESHOLD * 100:
        determined_color = "light"
    elif dark_perc >= ACHROMATIC_DOMINANCE_THRESHOLD * 100:
        determined_color = "dark"
    # Check if light wins and light+chromatic together exceed the threshold
    elif (light_perc + chromatic_perc) >= ACHROMATIC_DOMINANCE_THRESHOLD * 100 and light_perc > dark_perc and light_perc > chromatic_perc:
        determined_color = "light"
    # Check if dark wins and dark+chromatic together exceed the threshold
    elif (dark_perc + chromatic_perc) >= ACHROMATIC_DOMINANCE_THRESHOLD * 100 and dark_perc > light_perc and dark_perc > chromatic_perc:
        determined_color = "dark"
    elif chromatic_perc >= CHROMATIC_DOMINANCE_THRESHOLD * 100:
        if chromatic_pixels > 0:
            # --- Color Category Analysis ---
            chromatic_hues = h_values[is_chromatic]
            hist = cv2.calcHist([chromatic_hues], [0], None, [180], [0, 180])
            total_chromatic_hist = np.sum(hist)
            
            if total_chromatic_hist > 0:
                # Group histogram bins by color category
                color_categories = {}
                for i in range(180):
                    bin_value = hist[i][0]
                    if bin_value > 0:  # Skip empty bins
                        # Find which color category this hue belongs to
                        category = None
                        for color_name, ranges in HUE_RANGES.items():
                            for hue_range in ranges:
                                low, high = hue_range
                                if low > high:  # wrap-around
                                    if i >= low or i <= high:
                                        category = color_name
                                        break
                                elif low <= i <= high:
                                    category = color_name
                                    break
                            if category:
                                break
                        
                        # Use "unknown" for any hue that doesn't match our ranges
                        category = category or "unknown"
                        
                        # Add to category total
                        if category not in color_categories:
                            color_categories[category] = 0
                        color_categories[category] += bin_value
                
                # Find the dominant color category and its percentage
                if color_categories:
                    dominant_category = max(color_categories, key=color_categories.get)
                    dominant_category_percentage = (color_categories[dominant_category] / total_chromatic_hist) * 100
                    
                    # Check if dominant category exceeds threshold
                    if dominant_category_percentage >= SINGLE_HUE_MIN_PERC:
                        determined_color = dominant_category
                    else:
                        determined_color = "None"
                else:
                    determined_color = "None"
            else:
                determined_color = "None"
        else:
            determined_color = "None"
    else:
        determined_color = "None"
    
    # Check for color ambiguity
    if determined_color == "None":
        is_ambiguous = True
        possible_colors = ["light", "dark"]
    
    return determined_color, is_ambiguous, possible_colors

def add_color_information(xml_root, image_path):
    """Add color information to each object in the XML based on instance analysis."""
    try:
        # Load image and convert to HSV
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            return
        
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = image.shape[:2]
        
        # Process each object in the XML
        for obj in xml_root.findall('object'):
            # Get segmentation RLE if available
            segmentation_elem = obj.find('segmentation')
            if segmentation_elem is None or not segmentation_elem.text:
                # If no segmentation, mark as ambiguous
                color_elem = ET.SubElement(obj, 'color')
                color_elem.text = "None"
                is_ambiguous_elem = ET.SubElement(obj, 'is_ambiguous')
                is_ambiguous_elem.text = "true"
                possible_colors_elem = ET.SubElement(obj, 'possible_colors')
                possible_colors_elem.text = "light,dark"
                continue
                
            rle_string = segmentation_elem.text
            
            # Convert RLE to mask
            mask = rle_to_mask(rle_string, height, width)
            mask_bool = mask.astype(bool)
            
            if not np.any(mask_bool):
                # If empty mask, mark as ambiguous
                color_elem = ET.SubElement(obj, 'color')
                color_elem.text = "None"
                is_ambiguous_elem = ET.SubElement(obj, 'is_ambiguous')
                is_ambiguous_elem.text = "true"
                possible_colors_elem = ET.SubElement(obj, 'possible_colors')
                possible_colors_elem.text = "light,dark"
                continue
            
            # Extract HSV values for the masked area
            instance_pixels_hsv = hsv_image[mask_bool]
            
            # Determine color
            color, is_ambiguous, possible_colors = determine_instance_color(instance_pixels_hsv)
            
            # Always add color information
            color_elem = ET.SubElement(obj, 'color')
            color_elem.text = color
            
            # Add is_ambiguous flag
            is_ambiguous_elem = ET.SubElement(obj, 'is_ambiguous')
            is_ambiguous_elem.text = str(is_ambiguous).lower()
            
            # Add possible colors if ambiguous
            if is_ambiguous:
                possible_colors_elem = ET.SubElement(obj, 'possible_colors')
                possible_colors_elem.text = ','.join(possible_colors)
    except Exception as e:
        print(f"Error adding color information: {e}")
        # If there's an error, mark all objects as ambiguous
        for obj in xml_root.findall('object'):
            color_elem = ET.SubElement(obj, 'color')
            color_elem.text = "None"
            is_ambiguous_elem = ET.SubElement(obj, 'is_ambiguous')
            is_ambiguous_elem.text = "true"
            possible_colors_elem = ET.SubElement(obj, 'possible_colors')
            possible_colors_elem.text = "light,dark"

def determine_grid_position(instance, image_width, image_height, alpha=0.2, border_delta=0.10):
    """Determine grid position label based on the 3x3 grid with borderline detection"""
    x, y = instance['centroid']
    
    # Calculate grid boundaries
    third_w = image_width / 3
    third_h = image_height / 3
    
    # Calculate borderline regions
    border_w = image_width * border_delta
    border_h = image_height * border_delta
    
    # Determine vertical position and check for borderline
    is_vertical_borderline = False
    possible_verticals = []
    if y < third_h:
        vertical = "top"
        possible_verticals = ["top"]
        # Check if near bottom border of top section
        if y > third_h - border_h:
            is_vertical_borderline = True
            possible_verticals.append("center")
    elif y < 2*third_h:
        vertical = "center"
        possible_verticals = ["center"]
        # Check if near top or bottom border of center section
        if y < third_h + border_h:
            is_vertical_borderline = True
            possible_verticals.append("top")
        if y > 2*third_h - border_h:
            is_vertical_borderline = True
            possible_verticals.append("bottom")
    else:
        vertical = "bottom"
        possible_verticals = ["bottom"]
        # Check if near top border of bottom section
        if y < 2*third_h + border_h:
            is_vertical_borderline = True
            possible_verticals.append("center")
    
    # Determine horizontal position and check for borderline
    is_horizontal_borderline = False
    possible_horizontals = []
    if x < third_w:
        horizontal = "left"
        possible_horizontals = ["left"]
        # Check if near right border of left section
        if x > third_w - border_w:
            is_horizontal_borderline = True
            possible_horizontals.append("center")
    elif x < 2*third_w:
        horizontal = "center"
        possible_horizontals = ["center"]
        # Check if near left or right border of center section
        if x < third_w + border_w:
            is_horizontal_borderline = True
            possible_horizontals.append("left")
        if x > 2*third_w - border_w:
            is_horizontal_borderline = True
            possible_horizontals.append("right")
    else:
        horizontal = "right"
        possible_horizontals = ["right"]
        # Check if near left border of right section
        if x < 2*third_w + border_w:
            is_horizontal_borderline = True
            possible_horizontals.append("center")
    
    # Generate all possible position combinations
    possible_positions = []
    for v in possible_verticals:
        for h in possible_horizontals:
            if v == "center" and h == "center":
                possible_positions.append("in the center")
            else:
                possible_positions.append(f"in the {v} {h}")
    
    # Combine positions for the main position
    if vertical == "center" and horizontal == "center":
        position = "in the center"
    else:
        position = f"in the {vertical} {horizontal}"
    
    # Return position, borderline status, and possible positions
    return position, (is_vertical_borderline or is_horizontal_borderline), possible_positions

def find_mask_boundary_intersection(mask, start_point, end_point):
    """
    Find the intersection point of a line with a mask boundary using Bresenham's line algorithm.
    
    Args:
        mask: Binary mask of the instance
        start_point: (x,y) tuple of the starting point (centroid)
        end_point: (x,y) tuple of the ending point (target centroid)
        
    Returns:
        (x,y) tuple of the intersection point, or None if no intersection found
    """
    x1, y1 = map(int, start_point)
    x2, y2 = map(int, end_point)
    
    # Bresenham's line algorithm
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx - dy
    
    # Start from the centroid
    x, y = x1, y1
    
    # Track if we've been inside the mask
    was_inside = False
    
    while True:
        # Check if we're within mask bounds
        if 0 <= x < mask.shape[1] and 0 <= y < mask.shape[0]:
            is_inside = mask[y, x] == 1
            
            # If we were inside and now we're outside, we found the boundary
            if was_inside and not is_inside:
                return (x, y)
                
            was_inside = is_inside
        
        # If we've reached the end point, we're done
        if x == x2 and y == y2:
            break
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return None

def is_bbox_contained(bbox1, bbox2):
    """Check if one bounding box is contained within another"""
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Check if bbox1 is contained in bbox2
    contained_1_in_2 = (x1_min >= x2_min and y1_min >= y2_min and 
                       x1_max <= x2_max and y1_max <= y2_max)
    
    # Check if bbox2 is contained in bbox1
    contained_2_in_1 = (x2_min >= x1_min and y2_min >= y1_min and 
                       x2_max <= x1_max and y2_max <= y1_max)
    
    return contained_1_in_2 or contained_2_in_1

def is_centroid_contained(instance1, instance2):
    """Check if the centroid of one instance is inside the bounding box of another"""
    x1, y1 = instance1['centroid']
    x2, y2 = instance2['centroid']
    
    bbox1 = instance1['bbox']
    bbox2 = instance2['bbox']
    
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Check if centroid of instance1 is inside bbox of instance2
    centroid1_in_bbox2 = (x2_min <= x1 <= x2_max and y2_min <= y1 <= y2_max)
    
    # Check if centroid of instance2 is inside bbox of instance1
    centroid2_in_bbox1 = (x1_min <= x2 <= x1_max and y1_min <= y2 <= y1_max)
    
    return centroid1_in_bbox2 or centroid2_in_bbox1

def calculate_relationships(instance, all_instances, image_width, image_height, alpha=0.2, base_max_distance=150):
    """Calculate relationships between this instance and nearby instances only"""
    relationships = []
    x, y = instance['centroid']
    
    # Calculate source instance's average size
    source_width = instance['width']
    source_height = instance['height']
    source_avg_size = (source_width + source_height) / 2
    source_size_factor = source_avg_size / 2  # Half of average size
    
    for other in all_instances:
        if other['id'] == instance['id']:
            continue
        
        # Skip if one object is contained within the other (bbox or centroid containment)
        if is_bbox_contained(instance['bbox'], other['bbox']) or is_centroid_contained(instance, other):
            continue
        
        other_x, other_y = other['centroid']
        other_category = other['category']
        
        # Calculate target instance's average size
        other_width = other['width']
        other_height = other['height']
        other_avg_size = (other_width + other_height) / 2
        other_size_factor = other_avg_size / 2  # Half of average size
        
        # Calculate dynamic max distance based on both instances' sizes
        dynamic_max_distance = base_max_distance + source_size_factor + other_size_factor
        
        # Calculate distance using centroids
        distance = np.sqrt((other_x - x)**2 + (other_y - y)**2)
        
        # Only include relationships for nearby instances
        if distance > dynamic_max_distance:
            continue
        
        # Calculate angle
        angle = np.arctan2(y - other_y, other_x - x) * 180 / np.pi
        
        # Define angle ranges for each direction with overlap zones
        angle_ranges = {
            "to the left of": (-22.5, 22.5),
            "to the bottom left of": (22.5, 67.5),
            "below": (67.5, 112.5),
            "to the bottom right of": (112.5, 157.5),
            "to the right of": (157.5, 180),
            "to the right of": (-180, -157.5),
            "to the top right of": (-157.5, -112.5),
            "above": (-112.5, -67.5),
            "to the top left of": (-67.5, -22.5)
        }
        
        # Define overlap zones (in degrees)
        overlap_zone = 15.0  # 15 degrees overlap on each side
        
        # Determine if the angle is in an overlap zone
        is_borderline = False
        possible_directions = []
        
        for direction, (min_angle, max_angle) in angle_ranges.items():
            # Check if angle is within the main range
            if min_angle <= angle <= max_angle:
                possible_directions.append(direction)
                
                # Check if angle is near the boundaries
                if abs(angle - min_angle) < overlap_zone or abs(angle - max_angle) < overlap_zone:
                    is_borderline = True
                    
                    # Add the adjacent direction if we're near a boundary
                    if abs(angle - min_angle) < overlap_zone:
                        # Find the direction with the previous range
                        for prev_dir, (prev_min, prev_max) in angle_ranges.items():
                            if prev_max == min_angle:
                                possible_directions.append(prev_dir)
                                break
                    if abs(angle - max_angle) < overlap_zone:
                        # Find the direction with the next range
                        for next_dir, (next_min, next_max) in angle_ranges.items():
                            if next_min == max_angle:
                                possible_directions.append(next_dir)
                                break
        
        # Remove duplicates while preserving order
        possible_directions = list(dict.fromkeys(possible_directions))
        
        # Ensure we have at least one direction
        if not possible_directions:
            # If no direction was found (shouldn't happen), use the closest one
            closest_direction = min(angle_ranges.items(), 
                                 key=lambda x: min(abs(angle - x[1][0]), abs(angle - x[1][1])))
            possible_directions = [closest_direction[0]]
        
        # If borderline, use all possible directions
        if is_borderline:
            for direction in possible_directions:
                relationships.append({
                    'target_id': other['id'],
                    'target_category': other_category,
                    'direction': direction,
                    'distance': distance,
                    'is_cutoff': other.get('is_cutoff', False),
                    'is_borderline': True,
                    'possible_directions': possible_directions
                })
        else:
            # If not borderline, use the single direction
            relationships.append({
                'target_id': other['id'],
                'target_category': other_category,
                'direction': possible_directions[0],
                'distance': distance,
                'is_cutoff': other.get('is_cutoff', False),
                'is_borderline': False,
                'possible_directions': [possible_directions[0]]
            })
    
    # Sort by distance
    relationships.sort(key=lambda x: x['distance'])
    return relationships

def calculate_size_relationships(instances, image_width, image_height, size_threshold=1.5):
    """Calculate largest/smallest attributes within each class, only checking cutoff for smallest"""
    # Group instances by category
    instances_by_category = {}
    for inst in instances:
        category = inst['category']
        if category not in instances_by_category:
            instances_by_category[category] = []
        instances_by_category[category].append(inst)
    
    # Process each category
    for category, cat_instances in instances_by_category.items():
        if len(cat_instances) < 2:
            continue  # Need at least 2 for comparison
            
        # Sort by area
        sorted_instances = sorted(cat_instances, key=lambda x: x['area'])
        
        # Check if largest is significantly larger than others
        largest = sorted_instances[-1]
        second_largest = sorted_instances[-2]
        if largest['area'] > size_threshold * second_largest['area']:
            largest['size_attribute'] = 'largest'
        
        # For smallest, only consider fully visible instances
        visible_instances = []
        for inst in cat_instances:
            xmin, ymin, xmax, ymax = inst['bbox']
            if (xmin >= 0 and ymin >= 0 and 
                xmax <= image_width and ymax <= image_height):
                visible_instances.append(inst)
        
        # Only proceed with smallest if we have enough visible instances
        if len(visible_instances) >= 2:
            sorted_visible = sorted(visible_instances, key=lambda x: x['area'])
            smallest = sorted_visible[0]
            second_smallest = sorted_visible[1]
            if smallest['area'] * size_threshold < second_smallest['area']:
                smallest['size_attribute'] = 'smallest'

def bbox_distance(bbox1, bbox2):
    """Calculate minimum distance between two bounding boxes"""
    # Extract coordinates
    x1_min, y1_min, x1_max, y1_max = bbox1
    x2_min, y2_min, x2_max, y2_max = bbox2
    
    # Calculate x-axis distance
    if x1_max < x2_min:  # bbox1 is left of bbox2
        x_dist = x2_min - x1_max
    elif x2_max < x1_min:  # bbox1 is right of bbox2
        x_dist = x1_min - x2_max
    else:  # Boxes overlap on x-axis
        x_dist = 0
    
    # Calculate y-axis distance
    if y1_max < y2_min:  # bbox1 is above bbox2
        y_dist = y2_min - y1_max
    elif y2_max < y1_min:  # bbox1 is below bbox2
        y_dist = y1_min - y2_max
    else:  # Boxes overlap on y-axis
        y_dist = 0
    
    # Return Euclidean distance (or just min distance)
    return math.sqrt(x_dist**2 + y_dist**2)

def cluster_instances(instances, eps=100, min_samples=1, max_samples=10):
    """Cluster instances using DBSCAN with minimum bbox distance, return groups"""
    if len(instances) < 1:  # Changed from min_samples to 1
        return {}, {}  # Return empty groups and empty debug info
    
    # For debugging
    debug_instance_categories = {}
    for inst in instances:
        debug_instance_categories[inst['id']] = inst['category']
    
    # Group instances by category (no special handling for vehicles anymore)
    category_groups = {}
    for i, instance in enumerate(instances):
        # Skip cutoff instances
        if instance.get('is_cutoff', False):
            continue
            
        category = instance['category']
        
        if category not in category_groups:
            category_groups[category] = []
        
        # Store the instance and its index in the original list
        category_groups[category].append((i, instance))
    
    # Store groups mapping group_id -> list of instance IDs
    instance_groups = {}
    
    # For debugging - track which category each group comes from
    debug_group_categories = {}
    
    # Generate a unique ID for each category to avoid collisions
    # We'll use 1000 as the multiplier to leave plenty of room for clusters
    category_id_map = {}
    for i, category in enumerate(category_groups.keys()):
        category_id_map[category] = (i + 1) * 1000
    
    # First pass: Process each category group separately and create multi-instance groups
    for category, group_instances in category_groups.items():
        if len(group_instances) < 1:  # Changed from min_samples to 1
            continue  # Skip if no instances in this category
        
        # Extract indices and instances
        indices = [idx for idx, _ in group_instances]
        inst_list = [inst for _, inst in group_instances]
        
        # Debug info
        inst_ids = [inst['id'] for inst in inst_list]
        
        # Precompute distance matrix between instances in this category
        n = len(inst_list)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate distance between bounding boxes
                bbox1 = inst_list[i]['bbox']
                bbox2 = inst_list[j]['bbox']
                dist = bbox_distance(bbox1, bbox2)
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist  # Symmetric
        
        # Apply DBSCAN using precomputed distances with min_samples=1
        db = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed').fit(distance_matrix)
        instance_labels = db.labels_
        
        # Handle oversized clusters
        unique_labels = np.unique([l for l in instance_labels if l != -1])
        for label in unique_labels:
            # Find all instances in this cluster
            cluster_indices = [i for i, l in enumerate(instance_labels) if l == label]
            
            # If cluster exceeds max size, mark all as outliers
            if len(cluster_indices) > max_samples:
                for idx in cluster_indices:
                    instance_labels[idx] = -1
        
        # Get the category base ID for generating unique group IDs
        category_base_id = category_id_map[category]
        
        # Update valid cluster labels
        for i, label in enumerate(instance_labels):
            if label != -1:  # Only update actual clusters, not outliers
                # Create a truly unique group ID using category base ID
                group_id = category_base_id + label
                
                if group_id not in instance_groups:
                    instance_groups[group_id] = []
                    # For debugging - record which category this group comes from
                    debug_group_categories[group_id] = category
                
                # Add the actual instance ID to the group
                instance_id = inst_list[i]['id']
                instance_groups[group_id].append(instance_id)
    
    # Second pass: Create single instance groups only for instances that have relationships with multi-instance groups
    # First, collect all instances that are in multi-instance groups
    multi_instance_group_members = set()
    for group_id, instance_ids in instance_groups.items():
        if len(instance_ids) > 1:  # This is a multi-instance group
            multi_instance_group_members.update(instance_ids)
    
    # Now process each category again to handle single instances
    for category, group_instances in category_groups.items():
        # Extract instances that weren't assigned to any group
        unassigned_instances = []
        for _, instance in group_instances:
            if not any(instance['id'] in group for group in instance_groups.values()):
                unassigned_instances.append(instance)
        
        # For each unassigned instance, check if it has relationships with multi-instance group members
        for instance in unassigned_instances:
            # Calculate relationships with all instances
            relationships = calculate_relationships(instance, instances, 0, 0, base_max_distance=150)
            
            # Check if any relationship is with a multi-instance group member
            has_relationship_with_group = False
            for rel in relationships:
                if rel['target_id'] in multi_instance_group_members:
                    has_relationship_with_group = True
                    break
            
            # Only create a single instance group if it has relationships with multi-instance groups
            if has_relationship_with_group:
                # Create a unique ID for this individual instance
                instance_id = instance['id']
                single_group_id = category_id_map[category] + 500000 + len(instance_groups)
                
                if single_group_id not in instance_groups:
                    instance_groups[single_group_id] = []
                    # For debugging - record which category this group comes from
                    debug_group_categories[single_group_id] = category
                
                instance_groups[single_group_id].append(instance_id)
    
    # For debugging - verify group categories
    for group_id, instance_ids in instance_groups.items():
        categories = set(debug_instance_categories[inst_id] for inst_id in instance_ids)
        expected_category = debug_group_categories.get(group_id, "UNKNOWN")
        
        if len(categories) > 1:
            print(f"\nANOMALY DETECTED: Group {group_id} (expected category {expected_category}) contains mixed categories: {categories}")
            print(f"Instance IDs: {instance_ids}")
            print(f"This should never happen as we're clustering by category!")
            print("-" * 80)
    
    return instance_groups, debug_group_categories

def add_groups_to_xml(xml_root, groups, all_instances, image_width, image_height, debug_group_categories):
    """Add groups section to XML with their relationships"""
    # Create groups element
    groups_elem = ET.SubElement(xml_root, 'groups')
    
    # Add each group
    for group_id, instance_ids in groups.items():
        group_elem = ET.SubElement(groups_elem, 'group')
        
        # Add group ID
        id_elem = ET.SubElement(group_elem, 'id')
        id_elem.text = str(group_id)
        
        # Add instance IDs as comma-separated list
        instances_elem = ET.SubElement(group_elem, 'instance_ids')
        instances_elem.text = ','.join(str(id) for id in instance_ids)
        
        # Add group size
        size_elem = ET.SubElement(group_elem, 'size')
        size_elem.text = str(len(instance_ids))
        
        # Generate combined RLE for all instances in the group
        combined_mask = np.zeros((image_height, image_width), dtype=np.uint8)
        
        for obj in xml_root.findall('object'):
            obj_id = int(obj.find('id').text)
            if obj_id in instance_ids:
                # Get segmentation RLE if available
                segmentation_elem = obj.find('segmentation')
                if segmentation_elem is not None and segmentation_elem.text:
                    rle_string = segmentation_elem.text
                    instance_mask = rle_to_mask(rle_string, image_height, image_width)
                    # Add this instance mask to the combined mask
                    combined_mask = np.logical_or(combined_mask, instance_mask).astype(np.uint8)
        
        # Convert combined mask to RLE and add to XML
        if np.any(combined_mask):
            combined_rle = mask_to_rle(combined_mask)
            segmentation_elem = ET.SubElement(group_elem, 'segmentation')
            segmentation_elem.text = str(combined_rle)
        
        # Calculate and add group centroid
        centroids = []
        for obj in xml_root.findall('object'):
            obj_id = int(obj.find('id').text)
            if obj_id in instance_ids:
                bbox = obj.find('bndbox')
                if bbox is not None:
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    centroid_x = (xmin + xmax) / 2
                    centroid_y = (ymin + ymax) / 2
                    centroids.append((centroid_x, centroid_y))
        
        if centroids:
            # Calculate mean centroid
            mean_x = sum(x for x, y in centroids) / len(centroids)
            mean_y = sum(y for x, y in centroids) / len(centroids)
            
            # Add centroid coordinates
            centroid_elem = ET.SubElement(group_elem, 'centroid')
            ET.SubElement(centroid_elem, 'x').text = str(mean_x)
            ET.SubElement(centroid_elem, 'y').text = str(mean_y)
            
            # Add grid position for the group centroid
            group_instance = {
                'centroid': (mean_x, mean_y),
                'bbox': [0, 0, 0, 0]  # Dummy bbox since we only need centroid
            }
            grid_position = determine_group_grid_position(group_instance, image_width, image_height)
            grid_elem = ET.SubElement(group_elem, 'grid_position')
            grid_elem.text = grid_position
        
        # Add category if all instances have the same category
        categories = set()
        category_instances = {}
        
        for obj in xml_root.findall('object'):
            obj_id = int(obj.find('id').text)
            if obj_id in instance_ids:
                category = obj.find('name').text
                categories.add(category)
                
                # For debug info, track which instance IDs belong to which category
                if category not in category_instances:
                    category_instances[category] = []
                category_instances[category].append(obj_id)
        
        # Add category element
        category_elem = ET.SubElement(group_elem, 'category')
        
        # Handle different group types
        if group_id >= 2000000:  # Special pair groups
            category_elem.text = "vehicle_pair"
        elif group_id >= 1000000:  # Class-level groups
            # For class-level groups, use the actual category from the instances
            # Since these are class-level groups, all instances should have the same category
            if len(categories) == 1:
                category_elem.text = categories.pop()
            else:
                # This should never happen for class-level groups, but just in case
                print(f"\nWARNING: Class-level group {group_id} contains mixed categories: {categories}!")
                print(f"Filename: {xml_root.find('filename').text}")
                print(f"Instance IDs in this group: {instance_ids}")
                # Use the most common category as fallback
                most_common_category = max(category_instances.items(), key=lambda x: len(x[1]))[0]
                category_elem.text = most_common_category
        elif len(categories) == 1:  # Regular groups with same category
            category_elem.text = categories.pop()
        else:
            # Print warning for mixed categories in regular groups
            print(f"\nWARNING: Group {group_id} contains mixed categories: {categories}!")
            print(f"This indicates a bug in clustering as we should only group same-category instances.")
            print(f"Filename: {xml_root.find('filename').text}")
            print(f"Instance IDs in this group: {instance_ids}")
            
            # Show which instance belongs to which category
            for cat, ids in category_instances.items():
                print(f"  {cat}: {ids}")
            
            # Show bounding boxes to check proximity
            print("Bounding boxes:")
            for obj in xml_root.findall('object'):
                obj_id = int(obj.find('id').text)
                if obj_id in instance_ids:
                    bbox = obj.find('bndbox')
                    if bbox is not None:
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)
                        print(f"  Instance {obj_id}: [{xmin}, {ymin}, {xmax}, {ymax}]")
            
            # Use the most common category as fallback
            most_common_category = max(category_instances.items(), key=lambda x: len(x[1]))[0]
            category_elem.text = most_common_category

    # Add group relationships
    group_relationships = calculate_group_relationships(groups, all_instances, image_width, image_height)
    if group_relationships:
        group_rels_elem = ET.SubElement(groups_elem, 'relationships')
        for rel in group_relationships:
            rel_elem = ET.SubElement(group_rels_elem, 'relationship')
            
            ET.SubElement(rel_elem, 'source_id').text = str(rel['source_id'])
            ET.SubElement(rel_elem, 'target_id').text = str(rel['target_id'])
            ET.SubElement(rel_elem, 'direction').text = rel['direction']
            ET.SubElement(rel_elem, 'distance').text = str(rel['distance'])

def add_position_and_relationships_to_xml(instances, image_width, image_height, alpha=0.2, base_max_distance=150):
    """Add position and relationship information to XML objects"""
    for instance in instances:
        obj_element = instance['obj_element']
        
        # Add grid position and borderline status
        grid_position, is_borderline, possible_positions = determine_grid_position(instance, image_width, image_height, alpha)
        grid_elem = ET.SubElement(obj_element, 'grid_position')
        grid_elem.text = grid_position
        
        # Add borderline status
        borderline_elem = ET.SubElement(obj_element, 'is_borderline')
        borderline_elem.text = str(is_borderline).lower()
        
        # Add possible positions if borderline
        if is_borderline:
            possible_positions_elem = ET.SubElement(obj_element, 'possible_positions')
            possible_positions_elem.text = ','.join(possible_positions)
        
        # Add relationships
        relationships = calculate_relationships(instance, instances, image_width, image_height, alpha, base_max_distance)
        rels_elem = ET.SubElement(obj_element, 'relationships')
        
        # Add each relationship
        for rel in relationships:
            rel_elem = ET.SubElement(rels_elem, 'relationship')
            
            target_id_elem = ET.SubElement(rel_elem, 'target_id')
            target_id_elem.text = str(rel['target_id'])
            
            target_category_elem = ET.SubElement(rel_elem, 'target_category')
            target_category_elem.text = rel['target_category']
            
            direction_elem = ET.SubElement(rel_elem, 'direction')
            direction_elem.text = rel['direction']
            
            distance_elem = ET.SubElement(rel_elem, 'distance')
            distance_elem.text = str(rel['distance'])
            
            # Add cutoff flag to relationship
            is_cutoff_elem = ET.SubElement(rel_elem, 'is_cutoff')
            is_cutoff_elem.text = str(rel['is_cutoff']).lower()
            
            # Add borderline status for relationship
            is_borderline_elem = ET.SubElement(rel_elem, 'is_borderline')
            is_borderline_elem.text = str(rel['is_borderline']).lower()
            
            # Add possible directions if borderline
            if rel['is_borderline']:
                possible_directions_elem = ET.SubElement(rel_elem, 'possible_directions')
                possible_directions_elem.text = ','.join(rel['possible_directions'])

def assign_extreme_positions(instances, image_width, image_height, margin_ratio=0.05):
    """Assign topmost, bottommost, leftmost, rightmost if sufficiently separated from next candidate"""
    # Group instances by category
    instances_by_category = {}
    for inst in instances:
        category = inst['category']
        if category not in instances_by_category:
            instances_by_category[category] = []
        instances_by_category[category].append(inst)

    # Process each category separately
    for category, cat_instances in instances_by_category.items():
        if len(cat_instances) < 2:
            continue  # Skip categories with only one instance

        # Sort by coordinates
        sorted_by_y = sorted(cat_instances, key=lambda x: x['centroid'][1])
        sorted_by_x = sorted(cat_instances, key=lambda x: x['centroid'][0])

        margin_y = image_height * margin_ratio
        margin_x = image_width * margin_ratio

        candidates = {
            'topmost': (sorted_by_y[0], sorted_by_y[1], margin_y),
            'bottommost': (sorted_by_y[-1], sorted_by_y[-2], margin_y),
            'leftmost': (sorted_by_x[0], sorted_by_x[1], margin_x),
            'rightmost': (sorted_by_x[-1], sorted_by_x[-2], margin_x),
        }

        for label, (first, second, margin) in candidates.items():
            coord_idx = 1 if label in ['topmost', 'bottommost'] else 0
            first_val = first['centroid'][coord_idx]
            second_val = second['centroid'][coord_idx]

            # Compute separation
            if label in ['topmost', 'leftmost']:
                is_separated = (second_val - first_val) > margin
            else:  # bottommost, rightmost
                is_separated = (first_val - second_val) > margin

            if is_separated:
                first['extreme_position'] = label

    # Add extreme positions to XML
    for inst in instances:
        if 'extreme_position' in inst:
            elem = ET.SubElement(inst['obj_element'], 'extreme_position')
            elem.text = inst['extreme_position']

def calculate_group_relationships(groups, all_instances, image_width, image_height, base_max_distance=150):
    """Calculate relationships between groups and single instances"""
    relationships = []
    
    # First create a list of all entities from the groups structure
    entities = []
    
    # Process all groups (including single-instance groups)
    for group_id, instance_ids in groups.items():
        # Skip class-level groups (ID >= 1000000) and pair groups (ID >= 2000000)
        if group_id >= 1000000:
            continue
            
        # Calculate group centroid and bounding box
        centroids = []
        categories = set()
        bboxes = []
        
        for inst in all_instances:
            if inst['id'] in instance_ids:
                centroids.append(inst['centroid'])
                categories.add(inst['category'])
                bboxes.append(inst['bbox'])
        
        if not centroids:
            continue
            
        # Calculate mean centroid
        mean_x = sum(x for x,y in centroids) / len(centroids)
        mean_y = sum(y for x,y in centroids) / len(centroids)
        
        # Calculate group bounding box (min/max of all instance bboxes)
        if bboxes:
            min_x = min(bbox[0] for bbox in bboxes)
            min_y = min(bbox[1] for bbox in bboxes)
            max_x = max(bbox[2] for bbox in bboxes)
            max_y = max(bbox[3] for bbox in bboxes)
            group_bbox = [min_x, min_y, max_x, max_y]
        else:
            group_bbox = None
        
        entity = {
            'id': group_id,
            'centroid': (mean_x, mean_y),
            'bbox': group_bbox,
            'size': len(instance_ids),
            'is_single_instance': len(instance_ids) == 1  # Flag to identify single instance groups
        }
        
        # Add category if all instances have the same category
        if len(categories) == 1:
            entity['category'] = categories.pop()
            
        entities.append(entity)
    
    # Now calculate relationships between entities
    for i, entity in enumerate(entities):
        x, y = entity['centroid']
        
        for j, other in enumerate(entities):
            # Skip self-relationships
            if i == j:
                continue
            
            # Skip relationships between two single instance groups
            if entity['is_single_instance'] and other['is_single_instance']:
                continue
            
            # Skip if one group is contained within the other (bbox containment)
            if entity.get('bbox') and other.get('bbox'):
                if is_bbox_contained(entity['bbox'], other['bbox']):
                    continue
                
            other_x, other_y = other['centroid']
            
            # Calculate distance
            distance = np.sqrt((other_x - x)**2 + (other_y - y)**2)
            
            if distance > base_max_distance:
                continue
            
            # Calculate direction (same as individual instances)
            angle = np.arctan2(y - other_y, other_x - x) * 180 / np.pi
            
            if -22.5 <= angle < 22.5:
                direction = "to the left of"
            elif 22.5 <= angle < 67.5:
                direction = "to the bottom left of"
            elif 67.5 <= angle < 112.5:
                direction = "below"
            elif 112.5 <= angle < 157.5:
                direction = "to the bottom right of"
            elif 157.5 <= angle <= 180 or -180 <= angle < -157.5:
                direction = "to the right of"
            elif -157.5 <= angle < -112.5:
                direction = "to the top right of"
            elif -112.5 <= angle < -67.5:
                direction = "above"
            else:
                direction = "to the top left of"
            
            relationships.append({
                'source_id': entity['id'],
                'target_id': other['id'],
                'direction': direction,
                'distance': distance
            })
    
    return relationships

def determine_group_grid_position(instance, image_width, image_height):
    """Determine grid position label based on the 3x3 grid, without borderline detection"""
    x, y = instance['centroid']
    
    # Calculate grid boundaries
    third_w = image_width / 3
    third_h = image_height / 3
    
    # Determine vertical position
    if y < third_h:
        vertical = "top"
    elif y < 2*third_h:
        vertical = "center"
    else:
        vertical = "bottom"
    
    # Determine horizontal position
    if x < third_w:
        horizontal = "left"
    elif x < 2*third_w:
        horizontal = "center"
    else:
        horizontal = "right"
    
    # Combine positions
    if vertical == "center" and horizontal == "center":
        position = "in the center"
    else:
        position = f"in the {vertical} {horizontal}"
    
    return position

def create_class_level_groups(instances):
    """Create groups for all instances of each class in the patch"""
    class_groups = {}
    
    # Group instances by category, including cutoff instances
    for instance in instances:
        category = instance['category']
        if category not in class_groups:
            class_groups[category] = []
        class_groups[category].append(instance['id'])
    
    # Create a group for each class that has multiple instances
    groups = {}
    for category, instance_ids in class_groups.items():
        if len(instance_ids) > 1:  # Only create groups for classes with multiple instances
            # Use a special ID range for class-level groups (starting from 1000000)
            group_id = 1000000 + len(groups)
            groups[group_id] = instance_ids
    
    return groups

def create_special_pair_groups(instances):
    """Create special pair groups for small and large vehicles"""
    special_pairs = {}
    
    # Find small and large vehicles, including cutoff instances
    small_vehicles = []
    large_vehicles = []
    
    for instance in instances:
        category = instance['category']
        if category == 'Small_Vehicle':
            small_vehicles.append(instance['id'])
        elif category == 'Large_Vehicle':
            large_vehicles.append(instance['id'])
    
    # Create a special pair group if both types exist
    if small_vehicles and large_vehicles:
        # Use a special ID range for special pair groups (starting from 2000000)
        group_id = 2000000
        special_pairs[group_id] = small_vehicles + large_vehicles
    
    return special_pairs

def process_single_file(args):
    """Process a single XML file with all its annotations"""
    xml_path, output_xml_path, split_images_dir, alpha, size_threshold, eps, min_samples, max_samples = args
    
    try:
        # Parse XML
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get instances with references to XML elements
        instances, image_width, image_height = get_instance_info(root)
        
        # Assign extreme positions
        assign_extreme_positions(instances, image_width, image_height)

        # Calculate size relationships
        calculate_size_relationships(instances, image_width, image_height, size_threshold)
        
        # Cluster instances and get groups
        groups, debug_group_categories = cluster_instances(instances, eps, min_samples, max_samples)
        
        # Add class-level groups
        class_groups = create_class_level_groups(instances)
        groups.update(class_groups)
        
        # Add special pair groups
        special_pairs = create_special_pair_groups(instances)
        groups.update(special_pairs)  # Now special_pairs has the same structure as other groups
        
        # Add groups section to XML
        add_groups_to_xml(root, groups, instances, image_width, image_height, debug_group_categories)

        # Add position and relationship info to XML
        add_position_and_relationships_to_xml(instances, image_width, image_height, alpha)
        
        # Extract image path from XML
        image_filename = root.find('filename').text
        image_path = os.path.join(split_images_dir, image_filename)
        
        # Add color information
        if os.path.exists(image_path):
            add_color_information(root, image_path)
        else:
            print(f"Warning: Image file not found: {image_path}")
        
        # Save updated XML
        tree.write(output_xml_path)
        return True
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False

def main():
    # Hardcoded parameters
    input_dir = "dataset/patches"
    output_dir = "dataset/patches_rules"  # Second level
    alpha = 0.2
    size_threshold = 1.5  # Area must be 1.5x larger/smaller to be considered significant
    images_dir = "dataset/patches"
    
    # DBSCAN parameters
    eps = 40  # Maximum distance between two samples to be in the same cluster
    min_samples = 2  # Minimum number of samples in a cluster
    max_samples = 8  # Maximum samples in a cluster
    
    # Get number of CPU cores
    num_workers = multiprocessing.cpu_count()
    print(f"\nUsing {num_workers} worker processes")
    
    splits = ['train', 'val']
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        
        # Set up split-specific paths
        split_input_dir = os.path.join(input_dir, split, 'annotations')
        split_output_dir = os.path.join(output_dir, split)
        split_images_dir = os.path.join(images_dir, split, 'images')
        
        # Create output directories
        annotations_dir = os.path.join(split_output_dir, 'annotations')
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Get list of annotation files
        annotation_files = [f for f in os.listdir(split_input_dir) if f.endswith('.xml')]
        
        # Prepare arguments for parallel processing
        process_args = []
        for xml_file in annotation_files:
            xml_path = os.path.join(split_input_dir, xml_file)
            output_xml_path = os.path.join(annotations_dir, xml_file)
            process_args.append((xml_path, output_xml_path, split_images_dir, alpha, size_threshold, eps, min_samples, max_samples))
        
        # Set up progress bar
        pbar = tqdm(total=len(annotation_files), desc=f"Annotating {split} instances")
        
        # Start time
        start_time = time.time()
        
        # Process files in parallel
        with Pool(processes=num_workers) as pool:
            for _ in pool.imap_unordered(process_single_file, process_args):
                pbar.update(1)
        
        # End time
        end_time = time.time()
        
        pbar.close()
        print(f"\nProcessed {len(annotation_files)} files for {split} split")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    print(f"\nAll splits processed successfully!")
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main() 