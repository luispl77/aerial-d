import os
import xml.etree.ElementTree as ET
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import colorsys
import multiprocessing
from multiprocessing import Pool, Value, Manager
import time
import re

def get_instance_info(xml_root):
    """Extract instance information from XML annotation"""
    instances = []
    
    for obj in xml_root.findall('object'):
        obj_id = int(obj.find('id').text)
        category = obj.find('name').text
        
        # Get grid position
        grid_position = obj.find('grid_position').text
        
        # Get extreme position
        extreme_elem = obj.find('extreme_position')
        extreme_position = extreme_elem.text if extreme_elem is not None else None
        
        # Get size attribute
        size_elem = obj.find('size_attribute')
        size_attribute = size_elem.text if size_elem is not None else None
        
        # Get color attribute and ambiguity info
        color_elem = obj.find('color')
        color = color_elem.text if color_elem is not None else None
        
        is_ambiguous_elem = obj.find('is_ambiguous')
        is_ambiguous = is_ambiguous_elem.text.lower() == 'true' if is_ambiguous_elem is not None else False
        
        possible_colors_elem = obj.find('possible_colors')
        possible_colors = possible_colors_elem.text.split(',') if possible_colors_elem is not None else []
        
        # Get cutoff flag
        is_cutoff_elem = obj.find('is_cutoff')
        is_cutoff = is_cutoff_elem.text.lower() == 'true' if is_cutoff_elem is not None else False
        
        # Get relationships
        relationships = []
        for rel_elem in obj.findall('relationships/relationship'):
            rel = {
                'target_id': int(rel_elem.find('target_id').text),
                'target_category': rel_elem.find('target_category').text,
                'direction': rel_elem.find('direction').text,
                'distance': float(rel_elem.find('distance').text)
            }
            # Get cutoff flag for relationship
            is_cutoff_elem = rel_elem.find('is_cutoff')
            rel['is_cutoff'] = is_cutoff_elem.text.lower() == 'true' if is_cutoff_elem is not None else False
            
            # Get borderline status for relationship
            is_borderline_elem = rel_elem.find('is_borderline')
            rel['is_borderline'] = is_borderline_elem.text.lower() == 'true' if is_borderline_elem is not None else False
            
            # Get possible directions if borderline
            if rel['is_borderline']:
                possible_directions_elem = rel_elem.find('possible_directions')
                rel['possible_directions'] = possible_directions_elem.text.split(',') if possible_directions_elem is not None else []
            else:
                rel['possible_directions'] = [rel['direction']]
            
            relationships.append(rel)
        
        # Get bounding box
        bbox = obj.find('bndbox')
        if bbox is not None:
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            
            instances.append({
                'id': obj_id,
                'category': category,
                'grid_position': grid_position,
                'extreme_position': extreme_position,
                'size_attribute': size_attribute,
                'color': color,
                'is_ambiguous': is_ambiguous,
                'possible_colors': possible_colors,
                'is_cutoff': is_cutoff,
                'relationships': relationships,
                'bbox': [xmin, ymin, xmax, ymax],
                'obj_element': obj  # Store reference to XML element
            })
    
    return instances

def get_group_info(xml_root):
    """Extract group information from XML annotation"""
    groups = []
    group_relationships = {}
    
    groups_elem = xml_root.find('groups')
    if groups_elem is not None:
        # First get all groups
        for group_elem in groups_elem.findall('group'):
            group_id = int(group_elem.find('id').text)
            instance_ids = [int(id_str) for id_str in group_elem.find('instance_ids').text.split(',')]
            size = int(group_elem.find('size').text)
            category = group_elem.find('category').text
            grid_position = group_elem.find('grid_position').text
            
            # Get centroid
            centroid_elem = group_elem.find('centroid')
            centroid = {
                'x': float(centroid_elem.find('x').text),
                'y': float(centroid_elem.find('y').text)
            }
            
            groups.append({
                'id': group_id,
                'instance_ids': instance_ids,
                'size': size,
                'category': category,
                'grid_position': grid_position,
                'centroid': centroid,
                'group_element': group_elem
            })
        
        # Then get all relationships
        rels_elem = groups_elem.find('relationships')
        if rels_elem is not None:
            for rel in rels_elem.findall('relationship'):
                source_id = int(rel.find('source_id').text)
                target_id = int(rel.find('target_id').text)
                direction = rel.find('direction').text
                
                if source_id not in group_relationships:
                    group_relationships[source_id] = []
                group_relationships[source_id].append({
                    'target_id': target_id,
                    'direction': direction
                })
    
    return groups, group_relationships

def generate_all_expressions(instances, groups=None, group_relationships=None):
    """Generate all possible raw expressions for each instance and group"""
    all_expressions = {}
    dummy_expression_ids = {}  # Track which expressions are dummy expressions
    
    # Generate expressions for individual instances
    for instance in instances:
        obj_id = instance['id']
        category = instance['category']
        grid_position = instance['grid_position']
        extreme_position = instance.get('extreme_position')
        size_attribute = instance.get('size_attribute')
        color = instance.get('color')
        is_ambiguous = instance.get('is_ambiguous', False)
        possible_colors = instance.get('possible_colors', [])
        is_borderline = instance.get('is_borderline', False)
        possible_positions = instance.get('possible_positions', '').split(',') if is_borderline else []
        is_cutoff = instance.get('is_cutoff', False)  # Get cutoff flag
        expressions = []
        dummy_ids = []  # Track dummy expression IDs for this instance
        
        # Helper function to generate position variants for borderline objects
        def get_position_variants(base_expr):
            if is_borderline and possible_positions:
                return [re.sub(r'in the [^,]+', f'in the {pos}', base_expr) for pos in possible_positions]
            return [base_expr]
        
        # Helper function to generate relationship variants for borderline relationships
        def get_relationship_variants(base_expr, rel):
            if rel.get('is_borderline', False) and rel.get('possible_directions'):
                return [re.sub(r'that is [^,]+ a', f'that is {dir} a', base_expr) for dir in rel['possible_directions']]
            return [base_expr]
        
        # Helper function to add expression and track if it's a dummy
        def add_expression(expr, is_dummy=False):
            expressions.append(expr)
            if is_dummy or is_cutoff:  # Mark as dummy if object is cutoff or expression is ambiguous
                dummy_ids.append(len(expressions) - 1)  # Store the index of the dummy expression
        
        # Case 1: Just the category
        for expr in get_position_variants(f"the {category}"):
            add_expression(expr)
        
        # Case 2: Category + position
        for expr in get_position_variants(f"the {category} {grid_position}"):
            add_expression(expr)
        
        # Case 3: Category + position + relationship
        for rel in instance['relationships']:
            # Mark as dummy if either object is cutoff
            is_dummy = is_cutoff or rel.get('is_cutoff', False)
            base_expr = f"the {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
            
            # Generate variants for both position and relationship if either is borderline
            position_variants = get_position_variants(base_expr)
            for pos_expr in position_variants:
                for rel_expr in get_relationship_variants(pos_expr, rel):
                    add_expression(rel_expr, is_dummy=is_dummy)
        
        # Case 4: Extreme position + category
        if extreme_position:
            base_expr = f"the {extreme_position} {category}"
            for expr in get_position_variants(base_expr):
                add_expression(expr)
            
            base_expr = f"the {extreme_position} {category} {grid_position}"
            for expr in get_position_variants(base_expr):
                add_expression(expr)
            
            for rel in instance['relationships']:
                is_dummy = is_cutoff or rel.get('is_cutoff', False)
                base_expr = f"the {extreme_position} {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
                
                # Generate variants for both position and relationship if either is borderline
                position_variants = get_position_variants(base_expr)
                for pos_expr in position_variants:
                    for rel_expr in get_relationship_variants(pos_expr, rel):
                        add_expression(rel_expr, is_dummy=is_dummy)
        
        # New cases for size attributes - only combined with other attributes
        if size_attribute:
            # Case 5: Size + category + position
            base_expr = f"the {size_attribute} {category} {grid_position}"
            for expr in get_position_variants(base_expr):
                add_expression(expr)
            
            # Case 6: Size + category + position + relationship
            for rel in instance['relationships']:
                is_dummy = is_cutoff or rel.get('is_cutoff', False)
                base_expr = f"the {size_attribute} {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
                
                # Generate variants for both position and relationship if either is borderline
                position_variants = get_position_variants(base_expr)
                for pos_expr in position_variants:
                    for rel_expr in get_relationship_variants(pos_expr, rel):
                        add_expression(rel_expr, is_dummy=is_dummy)
            
            # Case 7: Size + extreme position combinations (if both exist)
            if extreme_position:
                base_expr = f"the {size_attribute} {extreme_position} {category} {grid_position}"
                for expr in get_position_variants(base_expr):
                    add_expression(expr)
                
                for rel in instance['relationships']:
                    is_dummy = is_cutoff or rel.get('is_cutoff', False)
                    base_expr = f"the {size_attribute} {extreme_position} {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
                    
                    # Generate variants for both position and relationship if either is borderline
                    position_variants = get_position_variants(base_expr)
                    for pos_expr in position_variants:
                        for rel_expr in get_relationship_variants(pos_expr, rel):
                            add_expression(rel_expr, is_dummy=is_dummy)
        
        # Handle color expressions
        if color or possible_colors:
            # If color is ambiguous, generate expressions for all possible colors
            colors_to_use = possible_colors if is_ambiguous else [color]
            
            # Define chromatic colors to exclude for buildings
            chromatic_colors = {"red", "orange", "yellow", "green", "cyan", "blue", "purple", "magenta"}
            
            # Filter out chromatic colors for buildings and water, but keep achromatic colors (light, dark)
            if category.lower() == "building" or category.lower() == "water":
                colors_to_use = [c for c in colors_to_use if c not in chromatic_colors]
            
            # Skip color expressions entirely if no valid colors remain after filtering
            if not colors_to_use:
                pass  # Skip the entire color expression section
            else:
                # For unambiguous colors, also generate expressions without color
                if not is_ambiguous:
                    # Case 8: Just category (without color)
                    base_expr = f"the {category}"
                    for expr in get_position_variants(base_expr):
                        add_expression(expr)
                    
                    # Case 9: Category + position (without color)
                    base_expr = f"the {category} {grid_position}"
                    for expr in get_position_variants(base_expr):
                        add_expression(expr)
                    
                    # Case 10: Category + position + relationship (without color)
                    for rel in instance['relationships']:
                        is_dummy = is_cutoff or rel.get('is_cutoff', False)
                        base_expr = f"the {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
                        
                        # Generate variants for both position and relationship if either is borderline
                        position_variants = get_position_variants(base_expr)
                        for pos_expr in position_variants:
                            for rel_expr in get_relationship_variants(pos_expr, rel):
                                add_expression(rel_expr, is_dummy=is_dummy)
                    
                    # Case 11: Extreme position combinations (without color)
                    if extreme_position:
                        base_expr = f"the {extreme_position} {category}"
                        for expr in get_position_variants(base_expr):
                            add_expression(expr)
                        
                        base_expr = f"the {extreme_position} {category} {grid_position}"
                        for expr in get_position_variants(base_expr):
                            add_expression(expr)
                        
                        for rel in instance['relationships']:
                            is_dummy = is_cutoff or rel.get('is_cutoff', False)
                            base_expr = f"the {extreme_position} {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
                            
                            # Generate variants for both position and relationship if either is borderline
                            position_variants = get_position_variants(base_expr)
                            for pos_expr in position_variants:
                                for rel_expr in get_relationship_variants(pos_expr, rel):
                                    add_expression(rel_expr, is_dummy=is_dummy)
                    
                    # Case 12: Size combinations (without color)
                    if size_attribute:
                        base_expr = f"the {size_attribute} {category} {grid_position}"
                        for expr in get_position_variants(base_expr):
                            add_expression(expr)
                        
                        for rel in instance['relationships']:
                            is_dummy = is_cutoff or rel.get('is_cutoff', False)
                            base_expr = f"the {size_attribute} {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
                            
                            # Generate variants for both position and relationship if either is borderline
                            position_variants = get_position_variants(base_expr)
                            for pos_expr in position_variants:
                                for rel_expr in get_relationship_variants(pos_expr, rel):
                                    add_expression(rel_expr, is_dummy=is_dummy)
                        
                        if extreme_position:
                            base_expr = f"the {size_attribute} {extreme_position} {category} {grid_position}"
                            for expr in get_position_variants(base_expr):
                                add_expression(expr)
                            
                            for rel in instance['relationships']:
                                is_dummy = is_cutoff or rel.get('is_cutoff', False)
                                base_expr = f"the {size_attribute} {extreme_position} {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
                                
                                # Generate variants for both position and relationship if either is borderline
                                position_variants = get_position_variants(base_expr)
                                for pos_expr in position_variants:
                                    for rel_expr in get_relationship_variants(pos_expr, rel):
                                        add_expression(rel_expr, is_dummy=is_dummy)
                
                # Now generate expressions with colors
                for current_color in colors_to_use:
                    # Case 13: Color + category
                    base_expr = f"the {current_color} {category}"
                    for expr in get_position_variants(base_expr):
                        add_expression(expr, is_dummy=is_ambiguous or is_cutoff)
                    
                    # Case 14: Color + category + position
                    base_expr = f"the {current_color} {category} {grid_position}"
                    for expr in get_position_variants(base_expr):
                        add_expression(expr, is_dummy=is_ambiguous or is_cutoff)
                    
                    # Case 15: Color + category + position + relationship
                    for rel in instance['relationships']:
                        is_dummy = is_ambiguous or is_cutoff or rel.get('is_cutoff', False)
                        base_expr = f"the {current_color} {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
                        
                        # Generate variants for both position and relationship if either is borderline
                        position_variants = get_position_variants(base_expr)
                        for pos_expr in position_variants:
                            for rel_expr in get_relationship_variants(pos_expr, rel):
                                add_expression(rel_expr, is_dummy=is_dummy)
                    
                    # Case 16: Color + extreme position combinations
                    if extreme_position:
                        base_expr = f"the {current_color} {extreme_position} {category}"
                        for expr in get_position_variants(base_expr):
                            add_expression(expr, is_dummy=is_ambiguous or is_cutoff)
                        
                        base_expr = f"the {current_color} {extreme_position} {category} {grid_position}"
                        for expr in get_position_variants(base_expr):
                            add_expression(expr, is_dummy=is_ambiguous or is_cutoff)
                        
                        for rel in instance['relationships']:
                            is_dummy = is_ambiguous or is_cutoff or rel.get('is_cutoff', False)
                            base_expr = f"the {current_color} {extreme_position} {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
                            
                            # Generate variants for both position and relationship if either is borderline
                            position_variants = get_position_variants(base_expr)
                            for pos_expr in position_variants:
                                for rel_expr in get_relationship_variants(pos_expr, rel):
                                    add_expression(rel_expr, is_dummy=is_dummy)
                    
                    # Case 17: Color + size combinations
                    if size_attribute:
                        base_expr = f"the {current_color} {size_attribute} {category} {grid_position}"
                        for expr in get_position_variants(base_expr):
                            add_expression(expr, is_dummy=is_ambiguous or is_cutoff)
                        
                        for rel in instance['relationships']:
                            is_dummy = is_ambiguous or is_cutoff or rel.get('is_cutoff', False)
                            base_expr = f"the {current_color} {size_attribute} {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
                            
                            # Generate variants for both position and relationship if either is borderline
                            position_variants = get_position_variants(base_expr)
                            for pos_expr in position_variants:
                                for rel_expr in get_relationship_variants(pos_expr, rel):
                                    add_expression(rel_expr, is_dummy=is_dummy)
                        
                        if extreme_position:
                            base_expr = f"the {current_color} {size_attribute} {extreme_position} {category} {grid_position}"
                            for expr in get_position_variants(base_expr):
                                add_expression(expr, is_dummy=is_ambiguous or is_cutoff)
                            
                            for rel in instance['relationships']:
                                is_dummy = is_ambiguous or is_cutoff or rel.get('is_cutoff', False)
                                base_expr = f"the {current_color} {size_attribute} {extreme_position} {category} {grid_position} that is {rel['direction']} a {rel['target_category']}"
                                
                                # Generate variants for both position and relationship if either is borderline
                                position_variants = get_position_variants(base_expr)
                                for pos_expr in position_variants:
                                    for rel_expr in get_relationship_variants(pos_expr, rel):
                                        add_expression(rel_expr, is_dummy=is_dummy)
        
        all_expressions[obj_id] = expressions
        if dummy_ids:  # Only store dummy IDs if there are any
            dummy_expression_ids[obj_id] = dummy_ids
    
    # Generate expressions for groups if provided
    if groups:
        # First, identify which groups are single instance, multi-instance, class-level, and special pairs
        single_instance_groups = set()
        multi_instance_groups = set()
        class_level_groups = set()
        special_pair_groups = set()
        
        for group in groups:
            if len(group['instance_ids']) == 1:
                single_instance_groups.add(group['id'])
            elif group['id'] >= 2000000:  # Special pair groups have IDs >= 2000000
                special_pair_groups.add(group['id'])
            elif group['id'] >= 1000000:  # Class-level groups have IDs >= 1000000
                class_level_groups.add(group['id'])
            else:
                multi_instance_groups.add(group['id'])
        
        for group in groups:
            group_id = group['id']
            instance_ids = group['instance_ids']
            size = group['size']
            category = group['category']
            grid_position = group['grid_position']
            expressions = []
            
            # Handle special pair groups
            if group_id in special_pair_groups:
                # Generate expressions for small and large vehicles pair
                base_expr = "all small and large vehicles in the image"
                expressions.append(base_expr)
                
            # Handle class-level groups
            elif group_id in class_level_groups:
                # Generate only the basic semantic segmentation expression in singular form
                base_expr = f"all {category} in the image"
                expressions.append(base_expr)
                
            # For single instance groups, only generate expressions if they have relationships with multi-instance groups
            elif group_id in single_instance_groups:
                has_multi_instance_relationship = False
                if group_relationships and group_id in group_relationships:
                    for rel in group_relationships[group_id]:
                        if rel['target_id'] in multi_instance_groups:
                            has_multi_instance_relationship = True
                            target_group = next((g for g in groups if g['id'] == rel['target_id']), None)
                            if target_group:
                                base_expr = f"the {category} {grid_position} that is {rel['direction']} a group of {target_group['size']} {target_group['category']}"
                                expressions.append(base_expr)
                
                # If no relationships with multi-instance groups, skip this group
                if not has_multi_instance_relationship:
                    continue
            else:
                # For multi-instance groups, generate all expressions
                # Case 1: Basic group expression
                base_expr = f"the group of {size} {category} {grid_position}"
                expressions.append(base_expr)
                
                # Case 2: Group with extreme position
                if group.get('extreme_position'):
                    base_expr = f"the {group['extreme_position']} group of {size} {category} {grid_position}"
                    expressions.append(base_expr)
                
                # Case 3: Group with relationships
                if group_relationships and group_id in group_relationships:
                    for rel in group_relationships[group_id]:
                        target_group = next((g for g in groups if g['id'] == rel['target_id']), None)
                        if target_group:
                            base_expr = f"the group of {size} {category} {grid_position} that is {rel['direction']} a group of {target_group['size']} {target_group['category']}"
                            expressions.append(base_expr)
            
            if expressions:  # Only add if we have expressions
                all_expressions[f"group_{group_id}"] = expressions
    
    return all_expressions, dummy_expression_ids

def add_expressions_to_xml(xml_root, all_expressions, dummy_expression_ids):
    """Add all raw expressions to XML objects and groups"""
    # Add expressions to individual objects
    for obj in xml_root.findall('object'):
        obj_id = int(obj.find('id').text)

        if obj_id in all_expressions:
            # Remove existing expressions if any
            existing = obj.find('expressions')
            if existing is not None:
                obj.remove(existing)

            # Add expressions element
            expressions_elem = ET.SubElement(obj, 'expressions')

            # Add each expression as a child element
            for i, expr in enumerate(all_expressions[obj_id]):
                expr_elem = ET.SubElement(expressions_elem, 'expression')
                expr_elem.set('id', str(i))
                expr_elem.text = expr
            
            # Add dummy expression IDs if any
            if obj_id in dummy_expression_ids:
                dummy_ids_elem = ET.SubElement(obj, 'dummy_expression_ids')
                dummy_ids_elem.text = ','.join(map(str, dummy_expression_ids[obj_id]))
    
    # Add expressions to groups
    groups_elem = xml_root.find('groups')
    if groups_elem is not None:
        for group_elem in groups_elem.findall('group'):
            group_id = int(group_elem.find('id').text)
            group_key = f"group_{group_id}"
            
            if group_key in all_expressions:
                # Remove existing expressions if any
                existing = group_elem.find('expressions')
                if existing is not None:
                    group_elem.remove(existing)
                
                # Add expressions element
                expressions_elem = ET.SubElement(group_elem, 'expressions')
                
                # Add each expression as a child element
                for i, expr in enumerate(all_expressions[group_key]):
                    expr_elem = ET.SubElement(expressions_elem, 'expression')
                    expr_elem.set('id', str(i))
                    expr_elem.text = expr
                
                # Add dummy expression IDs if any
                if group_key in dummy_expression_ids:
                    dummy_ids_elem = ET.SubElement(group_elem, 'dummy_expression_ids')
                    dummy_ids_elem.text = ','.join(map(str, dummy_expression_ids[group_key]))

def process_single_file(args):
    """Process a single XML file to generate expressions"""
    xml_path, output_xml_path = args
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get instances and groups
    instances = get_instance_info(root)
    groups, group_relationships = get_group_info(root)
    
    # Generate all possible expressions
    all_expressions, dummy_expression_ids = generate_all_expressions(instances, groups, group_relationships)
    
    # Add expressions to XML
    add_expressions_to_xml(root, all_expressions, dummy_expression_ids)
    
    # Save updated XML
    tree.write(output_xml_path)
    return True

def main():
    # Hardcoded parameters
    input_dir = "dataset/patches_rules"
    output_dir = "dataset/patches_rules_expressions"
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
            process_args.append((xml_path, output_xml_path))
        
        # Set up progress bar
        pbar = tqdm(total=len(annotation_files), desc=f"Generating {split} expressions")
        
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