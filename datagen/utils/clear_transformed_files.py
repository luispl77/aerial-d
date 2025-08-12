#!/usr/bin/env python3
"""
Script to clear all transformed image files from the dataset directory.

Transformed files are identified by filenames ending with "_toD", "_toL", or "_toP" 
(e.g., I1234_patch_0_toD.png, L5678_patch_1_toP.png).
These are image files only (.png) created from domain transfer transformations.

Usage:
    python clear_transformed_files.py [--dry-run] [--verbose]
    
    --dry-run: Show what would be deleted without actually deleting
    --verbose: Show detailed output for each file operation
"""

import os
import argparse
import logging
from pathlib import Path

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def find_transformed_files(base_path):
    """Find all transformed files (ending with 'to_D', 'to_L', or 'to_P') in the dataset directory."""
    base_path = Path(base_path)
    
    # Search for PNG files ending with the transform suffixes
    patterns = [
        "**/*_toD.png",
        "**/*_toL.png", 
        "**/*_toP.png"
    ]
    
    found_files = []
    for pattern in patterns:
        files = list(base_path.glob(pattern))
        found_files.extend(files)
    
    # Sort files for consistent output
    found_files.sort()
    
    logging.info(f"Found {len(found_files)} transformed files")
    return found_files

def delete_files(files, dry_run=False):
    """Delete the specified files."""
    deleted_count = 0
    failed_count = 0
    
    for file_path in files:
        try:
            if dry_run:
                logging.info(f"[DRY RUN] Would delete: {file_path}")
            else:
                file_path.unlink()
                logging.debug(f"Deleted: {file_path}")
                deleted_count += 1
        except Exception as e:
            logging.error(f"Failed to delete {file_path}: {e}")
            failed_count += 1
    
    if dry_run:
        logging.info(f"[DRY RUN] Would delete {len(files)} files")
    else:
        logging.info(f"Successfully deleted {deleted_count} files")
        if failed_count > 0:
            logging.warning(f"Failed to delete {failed_count} files")
    
    return deleted_count, failed_count

def main():
    parser = argparse.ArgumentParser(description='Clear all transformed image files (_toD, _toL, _toP)')
    parser.add_argument('--base-path', 
                       default='/cfs/home/u035679/aerialseg/datagen/dataset',
                       help='Base path to search for transformed files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logging.info("Starting transformed files cleanup...")
    logging.info(f"Base path: {args.base_path}")
    
    # Check if base path exists
    base_path = Path(args.base_path)
    if not base_path.exists():
        logging.error(f"Base path does not exist: {base_path}")
        return 1
    
    # Find all transformed files
    transformed_files = find_transformed_files(base_path)
    
    if not transformed_files:
        logging.info("No transformed files found to delete")
        return 0
    
    # Show breakdown by transform type and directory
    if args.verbose or args.dry_run:
        to_d_files = [f for f in transformed_files if f.name.endswith('_toD.png')]
        to_l_files = [f for f in transformed_files if f.name.endswith('_toL.png')]
        to_p_files = [f for f in transformed_files if f.name.endswith('_toP.png')]
        
        logging.info(f"File breakdown by transform type:")
        logging.info(f"  to_D (DeepGlobe style): {len(to_d_files)}")
        logging.info(f"  to_L (LoveDA style): {len(to_l_files)}")
        logging.info(f"  to_P (Potsdam style): {len(to_p_files)}")
        
        # Group by directory
        by_dir = {}
        for f in transformed_files:
            parent = f.parent
            if parent not in by_dir:
                by_dir[parent] = []
            by_dir[parent].append(f)
        
        logging.info(f"Files by directory:")
        for directory, files in sorted(by_dir.items()):
            logging.info(f"  {directory}: {len(files)} files")
    
    # Delete files
    deleted, failed = delete_files(transformed_files, args.dry_run)
    
    if failed > 0:
        return 1
    
    logging.info("Transformed files cleanup completed successfully")
    return 0

if __name__ == '__main__':
    exit(main())