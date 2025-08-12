#!/usr/bin/env python3
"""
Script to clear all DeepGlobe dataset files from the dataset directory.

DeepGlobe files are identified by filenames starting with "D" (e.g., D1234_patch_0.png, D1234_patch_0.xml).
This script will find and remove both image files (.png) and annotation files (.xml) across all splits.

Usage:
    python clear_deepglobe_files.py [--dry-run] [--verbose]
    
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

def find_deepglobe_files(base_path):
    """Find all DeepGlobe files (starting with 'D') in the dataset directory."""
    base_path = Path(base_path)
    
    # Search in all subdirectories for files starting with 'D'
    patterns = ["**/D*.png", "**/D*.xml"]
    
    found_files = []
    for pattern in patterns:
        files = list(base_path.glob(pattern))
        found_files.extend(files)
    
    # Sort files for consistent output
    found_files.sort()
    
    logging.info(f"Found {len(found_files)} DeepGlobe files")
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
    parser = argparse.ArgumentParser(description='Clear all DeepGlobe dataset files')
    parser.add_argument('--base-path', 
                       default='/cfs/home/u035679/aerialseg/datagen/dataset',
                       help='Base path to search for DeepGlobe files')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be deleted without actually deleting')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    logging.info("Starting DeepGlobe file cleanup...")
    logging.info(f"Base path: {args.base_path}")
    
    # Check if base path exists
    base_path = Path(args.base_path)
    if not base_path.exists():
        logging.error(f"Base path does not exist: {base_path}")
        return 1
    
    # Find all DeepGlobe files
    deepglobe_files = find_deepglobe_files(base_path)
    
    if not deepglobe_files:
        logging.info("No DeepGlobe files found to delete")
        return 0
    
    # Show breakdown by file type and directory
    if args.verbose or args.dry_run:
        png_files = [f for f in deepglobe_files if f.suffix == '.png']
        xml_files = [f for f in deepglobe_files if f.suffix == '.xml']
        
        logging.info(f"File breakdown:")
        logging.info(f"  PNG files (images): {len(png_files)}")
        logging.info(f"  XML files (annotations): {len(xml_files)}")
        
        # Group by directory
        by_dir = {}
        for f in deepglobe_files:
            parent = f.parent
            if parent not in by_dir:
                by_dir[parent] = []
            by_dir[parent].append(f)
        
        logging.info(f"Files by directory:")
        for directory, files in sorted(by_dir.items()):
            logging.info(f"  {directory}: {len(files)} files")
    
    # Delete files
    deleted, failed = delete_files(deepglobe_files, args.dry_run)
    
    if failed > 0:
        return 1
    
    logging.info("DeepGlobe file cleanup completed successfully")
    return 0

if __name__ == '__main__':
    exit(main())