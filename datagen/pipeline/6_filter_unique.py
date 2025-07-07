import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
from collections import defaultdict
import shutil
from PIL import Image, ImageDraw, ImageFont
import colorsys
import re
import multiprocessing
from multiprocessing import Pool, Value, Manager
import time

# Shared replacements dictionary
CLASS_REPLACEMENTS = {
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
}

def pluralize(word):
    """Properly pluralize a word"""
    if word.endswith('y'):
        return word[:-1] + 'ies'
    elif word.endswith('s') or word.endswith('sh') or word.endswith('ch'):
        return word + 'es'
    else:
        return word + 's'

def standardize_class_name(name, plural=False):
    """Standardize class names to be more natural language friendly"""
    standardized = CLASS_REPLACEMENTS.get(name, name).lower()
    
    # Handle pluralization
    if plural:
        standardized = pluralize(standardized)
    
    return standardized

def standardize_expression(expr, group_size=None):
    """
    Standardize class names inside expressions and handle group expressions
    Uses regex with word boundaries to avoid duplicate pluralization
    """
    # First standardize all class names
    for original, standard in CLASS_REPLACEMENTS.items():
        expr = re.sub(r'\b' + re.escape(original) + r'\b', standard, expr, flags=re.IGNORECASE)
    
    # Convert "group of 1 X" to just "X" in the entire expression
    for standard in CLASS_REPLACEMENTS.values():
        expr = re.sub(r'\bgroup of 1 ' + re.escape(standard) + r'\b', standard, expr, flags=re.IGNORECASE)
    
    # Handle pluralization for current group if size > 1
    if group_size is not None and group_size > 1:
        for standard in CLASS_REPLACEMENTS.values():
            pattern = r'\bgroup of ' + str(group_size) + r' ' + re.escape(standard) + r'\b'
            plural_form = f"group of {group_size} {pluralize(standard)}"
            expr = re.sub(pattern, plural_form, expr, flags=re.IGNORECASE)
    
    # Handle pluralization in relationship expressions 
    for standard in CLASS_REPLACEMENTS.values():
        # Find all "group of N" where N > 1
        for match in re.finditer(r'\bgroup of (\d+) ' + re.escape(standard) + r'\b', expr, re.IGNORECASE):
            if int(match.group(1)) > 1:
                # Only pluralize if not already pluralized
                if not match.group(0).endswith('s'):
                    replacement = f"group of {match.group(1)} {pluralize(standard)}"
                    expr = expr[:match.start()] + replacement + expr[match.end():]
    
    # Handle "all X" expressions
    for standard in CLASS_REPLACEMENTS.values():
        # Find "all X" patterns and ensure proper pluralization
        pattern = r'\ball ' + re.escape(standard) + r'\b'
        if re.search(pattern, expr, re.IGNORECASE):
            plural_form = f"all {pluralize(standard)}"
            expr = re.sub(pattern, plural_form, expr, flags=re.IGNORECASE)
    
    return expr.lower()

def remove_color_expressions_from_ambiguous(filtered_expressions, instances):
    """
    Remove all color expressions from ambiguous objects after duplicate filtering.
    
    Args:
        filtered_expressions: Dict mapping instance IDs to their filtered expressions
        instances: List of dicts containing instance information including is_ambiguous flag
        
    Returns:
        Dict mapping instance IDs to their filtered expressions with color expressions removed from ambiguous objects
    """
    # Create a set of ambiguous object IDs
    ambiguous_objects = {inst['id'] for inst in instances if inst.get('is_ambiguous', False)}
    
    # Create new dict for final expressions
    final_expressions = defaultdict(list)
    
    # Process each instance's expressions
    for inst_id, expressions in filtered_expressions.items():
        if inst_id in ambiguous_objects:
            # For ambiguous objects, only keep non-color expressions
            for expr in expressions:
                # Split into words and check each word
                words = expr.lower().split()
                # Keep if no word is a color term
                if not any(word in ['dark', 'light', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta'] for word in words):
                    final_expressions[inst_id].append(expr)
        else:
            # For non-ambiguous objects, keep all expressions
            final_expressions[inst_id].extend(expressions)
    
    return final_expressions

def filter_duplicate_expressions(instances):
    """
    Filter out expressions that appear more than once across all instances.
    If an expression appears multiple times, ALL instances of it are removed.
    
    Args:
        instances: List of dicts, each with 'id', 'expressions', and optional 'dummy_expression_ids' keys
        
    Returns:
        Dict mapping instance IDs to their filtered expressions with preserved IDs
    """
    expression_counts = defaultdict(int)
    normalized_to_original = defaultdict(list)

    # First pass: count normalized expressions
    for instance in instances:
        for i, expr in enumerate(instance['expressions']):
            norm_expr = standardize_expression(expr)
            expression_counts[norm_expr] += 1
            normalized_to_original[norm_expr].append((instance['id'], expr, i))  # Store original index

    # Final pass: keep only expressions that appear exactly once
    filtered_expressions = defaultdict(list)
    filtered_indices = defaultdict(list)  # Store the original indices of kept expressions
    for norm_expr, count in expression_counts.items():
        if count == 1:
            inst_id, original_expr, original_idx = normalized_to_original[norm_expr][0]
            filtered_expressions[inst_id].append(standardize_expression(original_expr))
            filtered_indices[inst_id].append(original_idx)

    # Remove dummy expressions using original indices
    for instance in instances:
        inst_id = instance['id']
        if inst_id in filtered_expressions and instance.get('dummy_expression_ids'):
            try:
                dummy_ids = [int(id_str) for id_str in instance['dummy_expression_ids'].split(',')]
                # Keep only non-dummy expressions using original indices
                kept_expressions = []
                for expr, orig_idx in zip(filtered_expressions[inst_id], filtered_indices[inst_id]):
                    if orig_idx not in dummy_ids:
                        kept_expressions.append(expr)
                filtered_expressions[inst_id] = kept_expressions
            except (AttributeError, ValueError):
                # If there's any error parsing dummy IDs, keep all expressions
                pass

    return filtered_expressions

def filter_group_expressions(groups):
    """
    Filter out group expressions that appear more than once across all groups.
    If an expression appears multiple times, ALL instances of it are removed.
    
    Args:
        groups: List of dicts, each with 'id', 'size', 'expressions' keys
        
    Returns:
        Dict mapping group IDs to their filtered expressions with preserved IDs
    """
    expression_counts = defaultdict(int)
    normalized_to_original = defaultdict(list)

    # First pass: count normalized expressions
    for group in groups:
        for i, expr in enumerate(group['expressions']):
            norm_expr = standardize_expression(expr, group['size'])
            expression_counts[norm_expr] += 1
            normalized_to_original[norm_expr].append((group['id'], expr, i))  # Store original index

    # Second pass: keep only unique normalized expressions
    filtered_expressions = defaultdict(list)
    for norm_expr, count in expression_counts.items():
        if count == 1:
            group_id, original_expr, _ = normalized_to_original[norm_expr][0]
            group = next(g for g in groups if g['id'] == group_id)
            filtered_expressions[group_id].append(standardize_expression(original_expr, group['size']))

    return filtered_expressions

def clean_xml_annotations(xml_root):
    """Remove useless annotations from the XML, keeping only essential elements"""
    # Elements to remove from each object
    useless_elements = [
        'relationships', 'grid_position', 'is_borderline', 'is_cutoff', 
        'is_ambiguous', 'color', 'extreme_position', 'possible_positions',
        'dummy_expression_ids', 'iscrowd'
    ]
    
    for obj in xml_root.findall('object'):
        # Remove useless elements
        for elem_name in useless_elements:
            elem = obj.find(elem_name)
            if elem is not None:
                obj.remove(elem)
    
    # Clean up groups as well
    groups_elem = xml_root.find('groups')
    if groups_elem is not None:
        for group in groups_elem.findall('group'):
            # Remove unnecessary elements from groups
            elements_to_remove = ['grid_position', 'instance_ids', 'relationships']
            for elem_name in elements_to_remove:
                elem = group.find(elem_name)
                if elem is not None:
                    group.remove(elem)
    
    # Remove nodes element if it exists and is empty
    nodes_elem = xml_root.find('nodes')
    if nodes_elem is not None and len(nodes_elem) == 0:
        xml_root.remove(nodes_elem)
    
    # Remove relationships element at the end of the file if it exists
    relationships_elem = xml_root.find('relationships')
    if relationships_elem is not None:
        xml_root.remove(relationships_elem)

def remove_objects_without_expressions(xml_root, filtered_instance_expressions, filtered_group_expressions):
    """Remove objects and groups that don't have any expressions after filtering"""
    
    # Remove objects without expressions
    objects_to_remove = []
    for obj in xml_root.findall('object'):
        obj_id = int(obj.find('id').text)
        if obj_id not in filtered_instance_expressions or len(filtered_instance_expressions[obj_id]) == 0:
            objects_to_remove.append(obj)
    
    for obj in objects_to_remove:
        xml_root.remove(obj)
    
    # Remove groups without expressions
    groups_elem = xml_root.find('groups')
    if groups_elem is not None:
        groups_to_remove = []
        for group in groups_elem.findall('group'):
            group_id = int(group.find('id').text)
            if group_id not in filtered_group_expressions or len(filtered_group_expressions[group_id]) == 0:
                groups_to_remove.append(group)
        
        for group in groups_to_remove:
            groups_elem.remove(group)
        
        # Remove the entire groups element if it's empty
        if len(groups_elem.findall('group')) == 0:
            xml_root.remove(groups_elem)

def update_xml_with_filtered(xml_root, filtered_expressions):
    """Update XML with filtered expressions and standardize class names"""
    for obj in xml_root.findall('object'):
        obj_id = int(obj.find('id').text)

        # Standardize class name
        name_elem = obj.find('name')
        if name_elem is not None:
            name_elem.text = standardize_class_name(name_elem.text)

        # Remove existing expressions
        expr_elem = obj.find('expressions')
        if expr_elem is not None:
            obj.remove(expr_elem)

        if obj_id in filtered_expressions:
            # Add updated expressions
            expressions_elem = ET.SubElement(obj, 'expressions')
            for i, expr in enumerate(filtered_expressions[obj_id]):
                expr_elem = ET.SubElement(expressions_elem, 'expression')
                expr_elem.set('id', str(i))
                expr_elem.text = expr

def update_xml_with_filtered_groups(xml_root, filtered_group_expressions):
    """Update XML with filtered group expressions"""
    groups_elem = xml_root.find('groups')
    if groups_elem is not None:
        for group in groups_elem.findall('group'):
            group_id = int(group.find('id').text)
            
            # Remove existing expressions
            expr_elem = group.find('expressions')
            if expr_elem is not None:
                group.remove(expr_elem)

            if group_id in filtered_group_expressions:
                # Add updated expressions
                expressions_elem = ET.SubElement(group, 'expressions')
                for i, expr in enumerate(filtered_group_expressions[group_id]):
                    expr_elem = ET.SubElement(expressions_elem, 'expression')
                    expr_elem.set('id', str(i))
                    expr_elem.text = expr

def process_single_file(args):
    """Process a single XML file to filter expressions"""
    xml_path, output_xml_path = args
    
    try:
        # Parse XML and check for raw_expressions
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Extract instances and their expressions
        instances = []
        cutoff_ids = set()  # Keep track of cutoff object IDs
        
        for obj in root.findall('object'):
            obj_id = int(obj.find('id').text)
            expressions = []

            # Check if object is cutoff
            is_cutoff = obj.find('is_cutoff')
            if is_cutoff is not None and is_cutoff.text.lower() == 'true':
                cutoff_ids.add(obj_id)

            # Get expressions
            expr_elem = obj.find('expressions')
            if expr_elem is not None:
                for e in expr_elem.findall('expression'):
                    expressions.append(e.text)

            # Get dummy expression IDs if any
            dummy_ids_elem = obj.find('dummy_expression_ids')
            dummy_expression_ids = dummy_ids_elem.text if dummy_ids_elem is not None else None

            instances.append({
                'id': obj_id,
                'expressions': expressions,
                'dummy_expression_ids': dummy_expression_ids
            })
        
        # Extract groups and their expressions
        groups = []
        groups_elem = root.find('groups')
        if groups_elem is not None:
            for group in groups_elem.findall('group'):
                group_id = int(group.find('id').text)
                size = int(group.find('size').text)
                expressions = []
                
                expr_elem = group.find('expressions')
                if expr_elem is not None:
                    for e in expr_elem.findall('expression'):
                        expressions.append(e.text)
                
                groups.append({
                    'id': group_id,
                    'size': size,
                    'expressions': expressions
                })
        
        # Filter expressions for both instances and groups
        filtered_instance_expressions = filter_duplicate_expressions(instances)
        filtered_group_expressions = filter_group_expressions(groups)
        
        # Remove expressions for cutoff objects after uniqueness filtering
        for cutoff_id in cutoff_ids:
            if cutoff_id in filtered_instance_expressions:
                del filtered_instance_expressions[cutoff_id]
        
        # Check if we have any valid expressions left
        if not filtered_instance_expressions and not filtered_group_expressions:
            # No valid expressions left, return False to indicate this file should be removed
            return False
        
        # Update XML with filtered expressions
        update_xml_with_filtered(root, filtered_instance_expressions)
        update_xml_with_filtered_groups(root, filtered_group_expressions)
        
        # Remove objects and groups without expressions
        remove_objects_without_expressions(root, filtered_instance_expressions, filtered_group_expressions)
        
        # Clean up useless annotations
        clean_xml_annotations(root)
        
        # Save updated XML
        tree.write(output_xml_path)
        return True
    except Exception as e:
        print(f"Error processing {xml_path}: {e}")
        return False

def clean_unused_images(images_dir, annotations_dir, split):
    """
    Remove image files that don't have corresponding XML annotations.
    
    Args:
        images_dir: Directory containing image files
        annotations_dir: Directory containing XML annotation files
        split: Dataset split (train or val)
    
    Returns:
        int: Number of removed image files
    """
    # Get paths
    split_images_dir = os.path.join(images_dir, split, 'images')
    split_annotations_dir = os.path.join(annotations_dir, split, 'annotations')
    
    # Check if directories exist
    if not os.path.exists(split_images_dir):
        print(f"ERROR: Images directory {split_images_dir} does not exist!")
        return 0
        
    if not os.path.exists(split_annotations_dir):
        print(f"ERROR: Annotations directory {split_annotations_dir} does not exist!")
        return 0
    
    # Get list of XML files (without extension)
    xml_files = set()
    for f in os.listdir(split_annotations_dir):
        if f.endswith('.xml'):
            base_name = os.path.splitext(f)[0]
            xml_files.add(base_name)
    
    # Check image files and remove those without XML
    removed_count = 0
    for f in os.listdir(split_images_dir):
        if f.endswith('.png') or f.endswith('.jpg'):
            base_name = os.path.splitext(f)[0]
            if base_name not in xml_files:
                image_path = os.path.join(split_images_dir, f)
                try:
                    os.remove(image_path)
                    removed_count += 1
                except Exception as e:
                    print(f"Error removing {image_path}: {e}")
    
    return removed_count

def main():
    # Hardcoded parameters
    input_dir = "dataset/patches_rules_expressions"
    output_dir = "dataset/patches_rules_expressions_unique"
    images_dir = "dataset/patches"
    
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
        
        # Check that the input directory exists and contains XML files
        if not os.path.exists(split_input_dir):
            print(f"ERROR: Input directory {split_input_dir} does not exist!")
            continue
            
        annotation_files = [f for f in os.listdir(split_input_dir) if f.endswith('.xml')]
        if not annotation_files:
            print(f"ERROR: No XML files found in {split_input_dir}")
            continue
            
        print(f"Found {len(annotation_files)} annotation files to process in {split_input_dir}")
        
        # Prepare arguments for parallel processing
        process_args = []
        for xml_file in annotation_files:
            xml_path = os.path.join(split_input_dir, xml_file)
            output_xml_path = os.path.join(annotations_dir, xml_file)
            process_args.append((xml_path, output_xml_path))
        
        # Set up progress bar
        pbar = tqdm(total=len(annotation_files), desc=f"Filtering {split} expressions")
        
        # Start time
        start_time = time.time()
        
        # Process files in parallel
        with Pool(processes=num_workers) as pool:
            results = []
            for result in pool.imap_unordered(process_single_file, process_args):
                results.append(result)
                pbar.update(1)
        
        # End time
        end_time = time.time()
        
        pbar.close()
        
        # Remove files that had no valid expressions
        removed_count = 0
        for i, result in enumerate(results):
            if not result:  # If process_single_file returned False
                xml_file = annotation_files[i]
                # Remove the XML file
                output_xml_path = os.path.join(annotations_dir, xml_file)
                if os.path.exists(output_xml_path):
                    os.remove(output_xml_path)
                
                # Remove the corresponding image file
                image_name = os.path.splitext(xml_file)[0]
                for ext in ['.jpg', '.png']:
                    image_path = os.path.join(split_images_dir, image_name + ext)
                    if os.path.exists(image_path):
                        os.remove(image_path)
                        removed_count += 1
                        break
        
        print(f"\nProcessed {len(annotation_files)} files for {split} split")
        print(f"Removed {removed_count} files with no valid expressions")
        print(f"Processing time: {end_time - start_time:.2f} seconds")
        
        # Clean up unused images
        print(f"\nCleaning up unused images for {split} split...")
        removed_images = clean_unused_images(images_dir, output_dir, split)
        print(f"Removed {removed_images} images without corresponding annotations")
    
    print(f"\nAll splits processed successfully!")
    print(f"Results saved to {output_dir}")

if __name__ == '__main__':
    main() 