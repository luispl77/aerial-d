#!/usr/bin/env python3
"""
Script to fix duplicated expressions in LoveDA files caused by running LLM step twice.

The problem: LLM was run twice, treating enhanced/unique expressions as new originals.
Pattern:
- Correct: 2 original -> 1 enhanced + 2 unique per original (6 total)
- Wrong: Those 6 were treated as originals -> 6 more enhanced + 2 more unique (14+ total)

Solution: Keep only the first set (original + first round of enhanced/unique).
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path
import argparse
import logging
import random

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def analyze_expressions(expressions_elem):
    """Analyze the structure of expressions to identify the correct cutoff."""
    if expressions_elem is None:
        return 0, []
    
    expressions = list(expressions_elem)
    
    # Parse through expressions to find the end of first round
    # Pattern: originals (with id) -> enhanced -> unique (first batch)
    # We need to stop after we see the first complete set of unique expressions
    
    original_count = 0
    first_enhanced_start = -1
    first_unique_start = -1
    first_unique_count = 0
    
    # Count originals and find phase transitions
    for i, expr in enumerate(expressions):
        expr_type = expr.get('type', 'original')
        
        if 'id' in expr.attrib:  # Original expression
            original_count += 1
        elif expr_type == 'enhanced' and first_enhanced_start == -1:
            first_enhanced_start = i
        elif expr_type == 'unique' and first_unique_start == -1:
            first_unique_start = i
        elif expr_type == 'unique' and first_unique_start != -1:
            first_unique_count += 1
        elif expr_type == 'enhanced' and first_unique_start != -1:
            # We've hit enhanced again after unique, so first round is complete
            break
    
    # For single original, expect: 1 original + 1 enhanced + 2 unique = 4 total
    # For multiple originals, expect: N original + N enhanced + 2 unique
    if original_count == 1:
        expected_unique_count = 2
    else:
        expected_unique_count = 2  # Always 2 unique per object, regardless of original count
    
    # Find cutoff point: after we've seen expected number of unique expressions
    cutoff_point = len(expressions)
    
    if first_unique_start != -1:
        unique_seen = 0
        for i in range(first_unique_start, len(expressions)):
            if expressions[i].get('type', 'original') == 'unique':
                unique_seen += 1
                if unique_seen >= expected_unique_count:
                    cutoff_point = i + 1
                    break
            elif expressions[i].get('type', 'original') == 'enhanced':
                # Hit enhanced after unique, means we're in second round
                cutoff_point = i
                break
    
    logging.debug(f"Analysis: {original_count} originals, cutoff at position {cutoff_point}/{len(expressions)}")
    
    return cutoff_point, expressions

def fix_expressions(expressions_elem, dry_run=False):
    """Fix expressions by keeping only the first round of generation."""
    if expressions_elem is None:
        return False
    
    keep_count, all_expressions = analyze_expressions(expressions_elem)
    
    if keep_count >= len(all_expressions):
        return False  # No fix needed
    
    # Remove excess expressions
    expressions_to_remove = all_expressions[keep_count:]
    
    if dry_run:
        logging.info(f"WILL KEEP (first {keep_count} expressions):")
        for i, expr in enumerate(all_expressions[:keep_count]):
            expr_type = expr.get('type', 'original')
            expr_id = expr.get('id', 'N/A')
            expr_text = expr.text[:80] + "..." if len(expr.text) > 80 else expr.text
            logging.info(f"  {i+1}. [{expr_type}] {expr_id}: {expr_text}")
        
        logging.info(f"WILL REMOVE ({len(expressions_to_remove)} excess expressions):")
        for i, expr in enumerate(expressions_to_remove):
            expr_type = expr.get('type', 'original')
            expr_id = expr.get('id', 'N/A')
            expr_text = expr.text[:80] + "..." if len(expr.text) > 80 else expr.text
            logging.info(f"  {i+1}. [{expr_type}] {expr_id}: {expr_text}")
    else:
        logging.info(f"Removing {len(expressions_to_remove)} excess expressions (keeping first {keep_count})")
    
    if not dry_run:
        for expr in expressions_to_remove:
            expressions_elem.remove(expr)
    
    return True

def process_xml_file(file_path, dry_run=False):
    """Process a single XML file to fix expressions."""
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
        
        fixed_objects = 0
        total_objects = 0
        
        # Process each object
        for obj in root.findall('object'):
            total_objects += 1
            expressions_elem = obj.find('expressions')
            
            if expressions_elem is not None:
                obj_name = obj.find('name')
                obj_id = obj.find('id')
                obj_desc = f"Object {obj_id.text if obj_id is not None else 'N/A'} ({obj_name.text if obj_name is not None else 'unknown'})"
                
                if dry_run:
                    logging.info(f"\n--- Processing {obj_desc} ---")
                
                if fix_expressions(expressions_elem, dry_run):
                    fixed_objects += 1
        
        # Process groups if they exist
        groups = root.find('groups')
        if groups is not None:
            for group in groups.findall('group'):
                total_objects += 1
                expressions_elem = group.find('expressions')
                if expressions_elem is not None:
                    group_id = group.find('id')
                    group_category = group.find('category')
                    group_desc = f"Group {group_id.text if group_id is not None else 'N/A'} ({group_category.text if group_category is not None else 'unknown'})"
                    
                    if dry_run:
                        logging.info(f"\n--- Processing {group_desc} ---")
                    
                    if fix_expressions(expressions_elem, dry_run):
                        fixed_objects += 1
        
        if fixed_objects > 0:
            if not dry_run:
                tree.write(file_path, encoding='utf-8', xml_declaration=True)
            logging.info(f"{'[DRY RUN] ' if dry_run else ''}Fixed {fixed_objects}/{total_objects} objects in {file_path}")
            return True
        else:
            logging.debug(f"No fixes needed for {file_path}")
            return False
            
    except ET.ParseError as e:
        logging.error(f"XML parse error in {file_path}: {e}")
        return False
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return False

def find_loveda_files(base_path):
    """Find all LoveDA XML files (starting with 'L')."""
    base_path = Path(base_path)
    pattern = "**/L*.xml"
    
    files = list(base_path.glob(pattern))
    logging.info(f"Found {len(files)} LoveDA files")
    return files

def main():
    parser = argparse.ArgumentParser(description='Fix duplicated expressions in LoveDA annotation files')
    parser.add_argument('--base-path', 
                       default='/cfs/home/u035679/aerialseg/datagen/dataset/patches_rules_expressions_unique',
                       help='Base path to search for XML files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be changed without making changes')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable debug logging')
    parser.add_argument('--file', type=str,
                       help='Process a specific file instead of all LoveDA files')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    setup_logging()
    
    if args.file:
        files = [Path(args.file)]
    else:
        files = find_loveda_files(args.base_path)
        if args.dry_run and len(files) > 0:
            # Pick a random file for dry run demonstration
            selected_file = random.choice(files)
            logging.info(f"DRY RUN: Randomly selected {selected_file} for demonstration")
            files = [selected_file]
    
    fixed_files = 0
    total_files = len(files)
    
    for file_path in files:
        if process_xml_file(file_path, args.dry_run):
            fixed_files += 1
    
    logging.info(f"{'[DRY RUN] ' if args.dry_run else ''}Summary: Fixed {fixed_files}/{total_files} files")

if __name__ == '__main__':
    main()