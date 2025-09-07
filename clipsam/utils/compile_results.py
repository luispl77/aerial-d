#!/usr/bin/env python3
"""
Compile Results Script
======================

This script compiles evaluation metrics from all validation result directories
into a single consolidated text file.

Usage:
    python compile_results.py
"""

import os
import glob

def main():
    # Path to the results directory
    results_dir = "/cfs/home/u035679/aerialseg/clipsam/results"
    
    # Output file
    output_file = os.path.join(results_dir, "compiled_results.txt")
    
    # Find all validation_results.txt files
    result_files = glob.glob(os.path.join(results_dir, "*", "validation_results.txt"))
    
    if not result_files:
        print("No validation_results.txt files found!")
        return
    
    # Sort the files by directory name for consistent ordering
    result_files.sort()
    
    compiled_results = []
    
    for file_path in result_files:
        # Extract the source folder name (parent directory)
        source_folder = os.path.basename(os.path.dirname(file_path))
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Extract the first 5 lines (the metrics we want)
            metrics = []
            for line in lines[:5]:
                line = line.strip()
                if line:
                    metrics.append(line)
            
            if metrics:
                # Add source folder name and the metrics
                compiled_results.append(f"Source: {source_folder}")
                compiled_results.extend(metrics)
                compiled_results.append("")  # Empty line for separation
                
                print(f"Processed: {source_folder}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Write to output file
    try:
        with open(output_file, 'w') as f:
            f.write('\n'.join(compiled_results))
        
        print(f"\nCompiled results saved to: {output_file}")
        print(f"Total directories processed: {len(result_files)}")
        
    except Exception as e:
        print(f"Error writing output file: {e}")

if __name__ == "__main__":
    main()