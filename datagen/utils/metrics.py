import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from tqdm import tqdm
import argparse
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except ImportError:
    WORDCLOUD_AVAILABLE = False
    print("Warning: wordcloud package not found. Word cloud generation will be skipped.")
    print("Install with: pip install wordcloud")

import multiprocessing
from multiprocessing import Pool, Value
import time

# Global counter for progress tracking
file_counter = None

def init_globals(counter):
    global file_counter
    file_counter = counter

def process_single_xml_file(xml_file_path):
    """Process a single XML file and return its metrics - worker function for multiprocessing"""
    try:
        result = analyze_single_xml(xml_file_path)
        
        # Update progress counter
        with file_counter.get_lock():
            file_counter.value += 1
            
        return result
    except Exception as e:
        print(f"Error processing {xml_file_path}: {str(e)}")
        return None

def analyze_single_xml(xml_file_path):
    """Analyze a single XML file and return metrics dictionary"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        # Initialize metrics for this file
        file_metrics = {
            "total_instances": 0,
            "total_groups": 0,
            "total_special_pairs": 0,
            "total_expressions": 0,
            "category_stats": defaultdict(int),
            "group_category_stats": defaultdict(int),
            "expressions_per_instance": [],
            "expressions_per_group": [],
            "expression_types": defaultdict(int),
            "expression_texts": [],
            "missing_masks": [],
            "skipped_objects": 0,
            "expression_count_mismatches": []  # Track files with mismatched original/enhanced counts
        }
        
        patch_expressions = 0
        objects_with_expressions = 0
        
        # Analyze individual objects
        for obj in root.findall(".//object"):
            # Check if object has expressions
            expressions = obj.findall(".//expression")
            if not expressions:
                file_metrics["skipped_objects"] += 1
                continue  # Skip objects without expressions
            
            # Check if object has segmentation
            segmentation = obj.find("segmentation")
            if segmentation is None:
                file_metrics["missing_masks"].append(str(xml_file_path))
                continue
            
            # Count instances per category (only those with expressions)
            category = obj.find("name").text
            file_metrics["category_stats"][category] += 1
            file_metrics["total_instances"] += 1
            objects_with_expressions += 1
            
            # Analyze expressions
            num_expressions = len(expressions)
            file_metrics["expressions_per_instance"].append(num_expressions)
            patch_expressions += num_expressions
            
            # Analyze expression types and collect texts
            for expr in expressions:
                file_metrics["total_expressions"] += 1
                expr_type = expr.get("type", "original")
                file_metrics["expression_types"][expr_type] += 1
                # Collect expression text for word cloud
                expr_text = expr.text.strip() if expr.text else ""
                if expr_text:
                    file_metrics["expression_texts"].append(expr_text)
        
        # Analyze groups
        groups_elem = root.find('groups')
        if groups_elem is not None:
            for group in groups_elem.findall('group'):
                # Get group expressions
                expressions = group.findall('expressions/expression')
                if not expressions:
                    continue
                
                # Count groups
                file_metrics["total_groups"] += 1
                
                # Get group category to identify special combination groups
                category = group.find('category')
                category_text = category.text if category is not None else "unknown"
                
                # Check if this is a special combination group (LoveDA)
                loveda_combination_groups = [
                    "natural_land",       # forest + agriculture + barren (all 3)
                    "vegetation",         # forest + agriculture
                    "undeveloped_land",   # forest + barren
                    "rural_land",         # agriculture + barren
                    "built_environment"   # road + building
                ]
                
                if category_text in loveda_combination_groups:
                    file_metrics["total_special_pairs"] += 1
                
                # Legacy check for iSAID special pairs (ID starts with 2000000)
                group_id = int(group.find('id').text)
                if group_id >= 2000000:
                    file_metrics["total_special_pairs"] += 1
                
                # Count group category
                file_metrics["group_category_stats"][category_text] += 1
                
                # Analyze group expressions
                num_expressions = len(expressions)
                file_metrics["expressions_per_group"].append(num_expressions)
                patch_expressions += num_expressions
                
                # Collect group expression texts for word cloud and count types
                for expr in expressions:
                    # Count expression type
                    expr_type = expr.get("type", "original")
                    file_metrics["expression_types"][expr_type] += 1
                    
                    # Collect expression text for word cloud
                    expr_text = expr.text.strip() if expr.text else ""
                    if expr_text:
                        file_metrics["expression_texts"].append(expr_text)
                
                # Add group expressions to total
                file_metrics["total_expressions"] += num_expressions
            

        
        # Only count patches that have objects or groups with expressions
        if objects_with_expressions > 0 or file_metrics["total_groups"] > 0:
            file_metrics["expressions_per_patch"] = [patch_expressions]
        else:
            file_metrics["expressions_per_patch"] = []
        
        # Check for expression count mismatches (original vs enhanced)
        original_count = file_metrics["expression_types"].get("original", 0)
        enhanced_count = file_metrics["expression_types"].get("enhanced", 0)
        
        if original_count != enhanced_count and (original_count > 0 or enhanced_count > 0):
            file_metrics["expression_count_mismatches"].append({
                "file": str(xml_file_path),
                "original": original_count,
                "enhanced": enhanced_count,
                "unique": file_metrics["expression_types"].get("unique", 0)
            })
        
        return file_metrics
        
    except Exception as e:
        print(f"Error analyzing {xml_file_path}: {str(e)}")
        return None

class DatasetMetrics:
    def __init__(self, dataset_path: str):
        """Initialize with path to the final dataset directory."""
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / "train"
        self.val_path = self.dataset_path / "val"
        self.metrics = {
            "train": self._init_metrics_dict(),
            "val": self._init_metrics_dict(),
            "total": self._init_metrics_dict()
        }
        self.missing_masks = []  # Store warnings about missing masks
        self.skipped_objects = {
            "train": 0,
            "val": 0,
            "total": 0
        }  # Count objects skipped due to missing expressions
        self.num_workers = multiprocessing.cpu_count()
    
    def _init_metrics_dict(self):
        """Initialize metrics dictionary for a split."""
        return {
            "total_patches": 0,
            "total_instances": 0,  # Only instances with expressions
            "total_groups": 0,     # Total number of groups
            "total_special_pairs": 0,  # Count of special combination groups (iSAID vehicle pairs + LoveDA land-use combinations)
            "total_expressions": 0,
            "category_stats": defaultdict(int),  # Only categories with expressions
            "group_category_stats": defaultdict(int),  # Categories for groups
            "expressions_per_instance": [],
            "expressions_per_group": [],  # Expressions per group
            "expressions_per_patch": [],
            "expression_types": defaultdict(int),
            "expression_texts": [],  # All expression texts for word cloud
            "expression_count_mismatches": []  # Files with mismatched original/enhanced counts


        }
        
    def analyze_dataset(self) -> Dict:
        """Analyze the entire dataset and return metrics."""
        print("Starting dataset analysis...")
        
        # Analyze train and val splits
        print("\nAnalyzing training split...")
        self._analyze_split(self.train_path, "train")
        print("\nAnalyzing validation split...")
        self._analyze_split(self.val_path, "val")
        
        print("\nCalculating total metrics...")
        # Calculate total metrics
        for key in ["total_patches", "total_instances", "total_expressions", "total_groups", "total_special_pairs"]:
            self.metrics["total"][key] = self.metrics["train"][key] + self.metrics["val"][key]
        
        # Merge category stats
        for category in set(list(self.metrics["train"]["category_stats"].keys()) + 
                          list(self.metrics["val"]["category_stats"].keys())):
            self.metrics["total"]["category_stats"][category] = (
                self.metrics["train"]["category_stats"][category] + 
                self.metrics["val"]["category_stats"][category]
            )
        
        # Merge group category stats
        for category in set(list(self.metrics["train"]["group_category_stats"].keys()) + 
                          list(self.metrics["val"]["group_category_stats"].keys())):
            self.metrics["total"]["group_category_stats"][category] = (
                self.metrics["train"]["group_category_stats"][category] + 
                self.metrics["val"]["group_category_stats"][category]
            )
        
        # Merge expression types
        for expr_type in set(list(self.metrics["train"]["expression_types"].keys()) + 
                          list(self.metrics["val"]["expression_types"].keys())):
            self.metrics["total"]["expression_types"][expr_type] = (
                self.metrics["train"]["expression_types"][expr_type] + 
                self.metrics["val"]["expression_types"][expr_type]
            )
        
        # Merge lists
        self.metrics["total"]["expressions_per_instance"] = (
            self.metrics["train"]["expressions_per_instance"] + 
            self.metrics["val"]["expressions_per_instance"]
        )
        self.metrics["total"]["expressions_per_group"] = (
            self.metrics["train"]["expressions_per_group"] + 
            self.metrics["val"]["expressions_per_group"]
        )
        self.metrics["total"]["expressions_per_patch"] = (
            self.metrics["train"]["expressions_per_patch"] + 
            self.metrics["val"]["expressions_per_patch"]
        )
        self.metrics["total"]["expression_texts"] = (
            self.metrics["train"]["expression_texts"] + 
            self.metrics["val"]["expression_texts"]
        )
        self.metrics["total"]["expression_count_mismatches"] = (
            self.metrics["train"]["expression_count_mismatches"] + 
            self.metrics["val"]["expression_count_mismatches"]
        )
        
        # Calculate averages
        for split in ["train", "val", "total"]:
            if self.metrics[split]["expressions_per_instance"]:
                self.metrics[split]["avg_expressions_per_instance"] = np.mean(self.metrics[split]["expressions_per_instance"])
            if self.metrics[split]["expressions_per_group"]:
                self.metrics[split]["avg_expressions_per_group"] = np.mean(self.metrics[split]["expressions_per_group"])
            if self.metrics[split]["expressions_per_patch"]:
                self.metrics[split]["avg_expressions_per_patch"] = np.mean(self.metrics[split]["expressions_per_patch"])
        
        print("Analysis complete!")
        
        # Print expression count mismatches to terminal immediately
        self._print_expression_mismatches()
        
        return self.metrics
    
    def _print_expression_mismatches(self):
        """Print expression count mismatches to terminal."""
        print("\n" + "="*80)
        print("EXPRESSION COUNT MISMATCH REPORT")
        print("="*80)
        
        total_mismatches = len(self.metrics["total"]["expression_count_mismatches"])
        
        if total_mismatches == 0:
            print("✅ No expression count mismatches found!")
            print("All files have matching original and enhanced expression counts.")
        else:
            print(f"⚠️  Found {total_mismatches} files with mismatched original/enhanced expression counts:")
            print("\nFiles with mismatches:")
            print("-" * 80)
            
            for split in ["train", "val"]:
                split_mismatches = self.metrics[split]["expression_count_mismatches"]
                if split_mismatches:
                    print(f"\n{split.upper()} SPLIT ({len(split_mismatches)} files):")
                    for mismatch in split_mismatches:
                        filename = mismatch["file"].split("/")[-1]  # Just the filename
                        print(f"  {filename}")
                        print(f"    Original: {mismatch['original']}, Enhanced: {mismatch['enhanced']}, Unique: {mismatch['unique']}")
        
        print("="*80)
    
    def _analyze_split(self, split_path: Path, split_name: str):
        """Analyze a specific split (train/val) using multiprocessing."""
        # Get all XML files
        xml_files = list(split_path.glob("**/*.xml"))
        self.metrics[split_name]["total_patches"] = len(xml_files)
        print(f"Found {len(xml_files)} XML files to process using {self.num_workers} workers")
        
        if not xml_files:
            print(f"No XML files found in {split_name} split")
            return
        
        # Initialize global counter for progress tracking
        global file_counter
        file_counter = Value('i', 0)
        
        # Start timing
        start_time = time.time()
        
        # Process files in parallel
        with Pool(processes=self.num_workers, initializer=init_globals, initargs=(file_counter,)) as pool:
            # Use tqdm to show progress
            with tqdm(total=len(xml_files), desc=f"Processing {split_name} files", unit="files") as pbar:
                results = []
                for result in pool.imap_unordered(process_single_xml_file, xml_files):
                    if result is not None:
                        results.append(result)
                    pbar.update(1)
        
        # End timing
        end_time = time.time()
        print(f"{split_name} split processed in {end_time - start_time:.2f} seconds")
        
        # Merge results from all workers
        self._merge_results(results, split_name)
    
    def _merge_results(self, results: List[Dict], split_name: str):
        """Merge results from all worker processes."""
        print(f"Merging results from {len(results)} processed files...")
        
        for file_result in results:
            if file_result is None:
                continue
                
            # Merge scalar values
            self.metrics[split_name]["total_instances"] += file_result["total_instances"]
            self.metrics[split_name]["total_groups"] += file_result["total_groups"]
            self.metrics[split_name]["total_special_pairs"] += file_result["total_special_pairs"]
            self.metrics[split_name]["total_expressions"] += file_result["total_expressions"]
            self.skipped_objects[split_name] += file_result["skipped_objects"]
            
            # Merge category stats
            for category, count in file_result["category_stats"].items():
                self.metrics[split_name]["category_stats"][category] += count
            
            # Merge group category stats
            for category, count in file_result["group_category_stats"].items():
                self.metrics[split_name]["group_category_stats"][category] += count
            
            # Merge expression types
            for expr_type, count in file_result["expression_types"].items():
                self.metrics[split_name]["expression_types"][expr_type] += count
            

            
            # Extend lists
            self.metrics[split_name]["expressions_per_instance"].extend(file_result["expressions_per_instance"])
            self.metrics[split_name]["expressions_per_group"].extend(file_result["expressions_per_group"])
            self.metrics[split_name]["expressions_per_patch"].extend(file_result["expressions_per_patch"])
            self.metrics[split_name]["expression_texts"].extend(file_result["expression_texts"])
            self.metrics[split_name]["expression_count_mismatches"].extend(file_result["expression_count_mismatches"])
            
            # Merge missing masks
            self.missing_masks.extend(file_result["missing_masks"])
    
    def generate_report(self, output_path: str = None):
        """Generate a comprehensive report of the dataset metrics."""
        print("\nGenerating report...")
        report = []
        report.append("=== Dataset Metrics Report ===")
        
        # Add missing masks report if any
        if self.missing_masks:
            report.append("\n=== Objects Missing Masks ===")
            report.append(f"Total files with missing masks: {len(set(self.missing_masks))}")
            report.append("Files affected:")
            for file in sorted(set(self.missing_masks)):
                report.append(f"  {file}")
        
        # Add skipped objects report
        report.append("\n=== Objects Without Expressions ===")
        for split in ["train", "val"]:
            report.append(f"{split.upper()}: {self.skipped_objects[split]} objects skipped")
        report.append(f"TOTAL: {self.skipped_objects['total']} objects skipped")
        
        for split in ["train", "val", "total"]:
            report.append(f"\n=== {split.upper()} Split ===")
            report.append(f"Total Patches: {self.metrics[split]['total_patches']}")
            report.append(f"Total Instances: {self.metrics[split]['total_instances']}")
            report.append(f"Total Groups: {self.metrics[split]['total_groups']}")
            report.append(f"Total Special Combination Groups (iSAID vehicle pairs + LoveDA land-use combinations): {self.metrics[split]['total_special_pairs']}")
            report.append(f"Total Expressions: {self.metrics[split]['total_expressions']}")
            
            if "avg_expressions_per_instance" in self.metrics[split]:
                report.append(f"Average Expressions per Instance: {self.metrics[split]['avg_expressions_per_instance']:.2f}")
            if "avg_expressions_per_group" in self.metrics[split]:
                report.append(f"Average Expressions per Group: {self.metrics[split]['avg_expressions_per_group']:.2f}")
            if "avg_expressions_per_patch" in self.metrics[split]:
                report.append(f"Average Expressions per Patch: {self.metrics[split]['avg_expressions_per_patch']:.2f}")
            
            report.append("\nInstance Category Distribution:")
            for category, count in sorted(self.metrics[split]["category_stats"].items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {category}: {count} instances")
            
            report.append("\nGroup Category Distribution:")
            for category, count in sorted(self.metrics[split]["group_category_stats"].items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {category}: {count} groups")
            
            # Add special breakdown for LoveDA combination groups
            loveda_combinations = ["natural_land", "vegetation", "undeveloped_land", "rural_land", "built_environment"]
            loveda_combo_count = sum(self.metrics[split]["group_category_stats"].get(cat, 0) for cat in loveda_combinations)
            if loveda_combo_count > 0:
                report.append(f"\nLoveDA Land-use Combination Groups Breakdown:")
                for cat in loveda_combinations:
                    count = self.metrics[split]["group_category_stats"].get(cat, 0)
                    if count > 0:
                        if cat == "natural_land":
                            description = "forest + agriculture + barren"
                        elif cat == "vegetation":
                            description = "forest + agriculture"
                        elif cat == "undeveloped_land":
                            description = "forest + barren"
                        elif cat == "rural_land":
                            description = "agriculture + barren"
                        elif cat == "built_environment":
                            description = "roads + buildings"
                        report.append(f"  {cat} ({description}): {count} groups")
            

            
            report.append("\nExpression Type Distribution:")
            for expr_type, count in sorted(self.metrics[split]["expression_types"].items(), key=lambda x: x[1], reverse=True):
                report.append(f"  {expr_type}: {count} expressions")
            
            # Add combined category distribution for total split only
            if split == "total":
                report.append("\n" + "="*50)
                report.append("COMBINED DATASET CATEGORY DISTRIBUTIONS")
                report.append("="*50)
                
                report.append("\nInstance Category Distribution (sorted by count):")
                instance_items = sorted(self.metrics[split]["category_stats"].items(), key=lambda x: x[1], reverse=True)
                for category, count in instance_items:
                    percentage = (count / self.metrics[split]["total_instances"]) * 100 if self.metrics[split]["total_instances"] > 0 else 0
                    report.append(f"  {category}: {count:,} instances ({percentage:.1f}%)")
                
                if self.metrics[split]["group_category_stats"]:
                    report.append("\nGroup Category Distribution (sorted by count):")
                    group_items = sorted(self.metrics[split]["group_category_stats"].items(), key=lambda x: x[1], reverse=True)
                    total_groups = sum(self.metrics[split]["group_category_stats"].values())
                    for category, count in group_items:
                        percentage = (count / total_groups) * 100 if total_groups > 0 else 0
                        report.append(f"  {category}: {count:,} groups ({percentage:.1f}%)")
                
                report.append("="*50)
            

            

        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report_text)
            print(f"Report saved to {output_path}")
        
        return report_text
    
    def plot_distributions(self, output_dir: str, skip_wordcloud: bool = False):
        """Generate visualization plots for key metrics."""
        print("\nGenerating visualizations...")
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for split in ["train", "val"]:
            print(f"Creating plots for {split} split...")
            split_dir = output_dir / split
            split_dir.mkdir(exist_ok=True)
            
            # Category distribution
            plt.figure(figsize=(12, 6))
            categories = list(self.metrics[split]["category_stats"].keys())
            counts = list(self.metrics[split]["category_stats"].values())
            sns.barplot(x=categories, y=counts)
            plt.xticks(rotation=45, ha='right', fontsize=22, fontweight='bold')
            plt.yticks(fontsize=22, fontweight='bold')
            plt.xlabel("Category", fontsize=24, fontweight='bold')
            plt.ylabel("Number of Instances", fontsize=24, fontweight='bold')
            plt.tight_layout()
            plt.savefig(split_dir / "category_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Group category distribution
            if self.metrics[split]["group_category_stats"]:
                plt.figure(figsize=(12, 6))
                group_categories = list(self.metrics[split]["group_category_stats"].keys())
                group_counts = list(self.metrics[split]["group_category_stats"].values())
                
                # Sort by count for better visualization
                group_data = sorted(zip(group_categories, group_counts), key=lambda x: x[1], reverse=True)
                group_categories_sorted, group_counts_sorted = zip(*group_data) if group_data else ([], [])
                
                sns.barplot(x=group_categories_sorted, y=group_counts_sorted)
                plt.xticks(rotation=45, ha='right', fontsize=22, fontweight='bold')
                plt.yticks(fontsize=22, fontweight='bold')
                plt.xlabel("Group Category", fontsize=24, fontweight='bold')
                plt.ylabel("Number of Groups", fontsize=24, fontweight='bold')
                plt.tight_layout()
                plt.savefig(split_dir / "group_category_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # Expression type distribution (improved)
            plt.figure(figsize=(12, 8))
            expr_types = list(self.metrics[split]["expression_types"].keys())
            expr_counts = list(self.metrics[split]["expression_types"].values())
            
            # Sort by count for better visualization
            expr_data = sorted(zip(expr_types, expr_counts), key=lambda x: x[1], reverse=True)
            expr_types_sorted, expr_counts_sorted = zip(*expr_data) if expr_data else ([], [])
            
            # Use different colors for each type
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'][:len(expr_types_sorted)]
            bars = plt.bar(expr_types_sorted, expr_counts_sorted, color=colors)
            
            # Add count labels on bars
            for bar, count in zip(bars, expr_counts_sorted):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'{count:,}', ha='center', va='bottom', fontweight='bold')
            
            plt.xlabel("Expression Type", fontsize=20)
            plt.ylabel("Number of Expressions", fontsize=20)
            plt.xticks(rotation=0, fontsize=18)
            plt.yticks(fontsize=18)
            
            # Add grid for better readability
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(split_dir / "expression_type_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Expressions per instance histogram
            if self.metrics[split]["expressions_per_instance"]:
                plt.figure(figsize=(10, 6))
                sns.histplot(self.metrics[split]["expressions_per_instance"], bins=20)
                plt.title(f"Distribution of Expressions per Instance - {split.upper()}")
                plt.xlabel("Number of Expressions")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(split_dir / "expressions_per_instance.png")
                plt.close()
            
            # Word cloud generation skipped for individual splits (only generated for total)
        
        # Generate combined dataset category distribution plot (instances only)
        print("Creating combined dataset instance category distribution plot...")
        combined_dir = output_dir / "combined"
        combined_dir.mkdir(exist_ok=True)
        
        # Combined Instance Category Distribution (both train and val)
        instance_categories = list(self.metrics["total"]["category_stats"].keys())
        instance_counts = list(self.metrics["total"]["category_stats"].values())
        
        if instance_categories and instance_counts:
            # Sort by count for better visualization
            instance_data = sorted(zip(instance_categories, instance_counts), key=lambda x: x[1], reverse=True)
            instance_categories_sorted, instance_counts_sorted = zip(*instance_data)
            
            # Create a modern, clean chart
            plt.figure(figsize=(18, 12))
            
            # Generate distinct colors for each category using a high-quality colormap
            colors = cm.tab20(np.linspace(0, 1, len(instance_categories_sorted)))
            
            # Create bars with enhanced styling
            bars = plt.bar(instance_categories_sorted, instance_counts_sorted, 
                          color=colors, edgecolor='white', linewidth=1.5,
                          alpha=0.9)
            
            # Add count labels on bars with better formatting
            for bar, count in zip(bars, instance_counts_sorted):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + height*0.005,
                        f'{count:,}', ha='center', va='bottom', 
                        fontweight='bold', fontsize=16, color='black')
            
            # Enhanced styling with larger fonts
            plt.xlabel("Category", fontsize=24, fontweight='bold', labelpad=25)
            plt.ylabel("Number of Instances", fontsize=24, fontweight='bold', labelpad=25)
            plt.xticks(rotation=45, ha='right', fontsize=22, fontweight='bold')
            plt.yticks(fontsize=22, fontweight='bold')
            
            # Add subtle grid for readability
            plt.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
            
            # Remove top and right spines for cleaner look
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(1.2)
            ax.spines['bottom'].set_linewidth(1.2)
            
            # Improve layout and margins
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.15)  # Extra space for rotated labels
            
            plt.savefig(combined_dir / "instance_category_distribution.png", 
                       dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close()
        
        # Generate combined dataset group category distribution plot
        if self.metrics["total"]["group_category_stats"]:
            print("Creating combined dataset group category distribution plot...")
            
            # Combined Group Category Distribution (both train and val)
            group_categories = list(self.metrics["total"]["group_category_stats"].keys())
            group_counts = list(self.metrics["total"]["group_category_stats"].values())
            
            if group_categories and group_counts:
                # Sort by count for better visualization
                group_data = sorted(zip(group_categories, group_counts), key=lambda x: x[1], reverse=True)
                group_categories_sorted, group_counts_sorted = zip(*group_data)
                
                # Create a modern, clean chart
                plt.figure(figsize=(18, 12))
                
                # Generate distinct colors for each category using a high-quality colormap
                colors = cm.tab20(np.linspace(0, 1, len(group_categories_sorted)))
                
                # Create bars with enhanced styling
                bars = plt.bar(group_categories_sorted, group_counts_sorted, 
                              color=colors, edgecolor='white', linewidth=1.5,
                              alpha=0.9)
                
                # Add count labels on bars with better formatting
                for bar, count in zip(bars, group_counts_sorted):
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + height*0.005,
                            f'{count:,}', ha='center', va='bottom', 
                            fontweight='bold', fontsize=15, color='black')
                
                # Enhanced styling with larger fonts
                plt.xlabel("Group Category", fontsize=24, fontweight='bold', labelpad=25)
                plt.ylabel("Number of Groups", fontsize=24, fontweight='bold', labelpad=25)
                plt.xticks(rotation=45, ha='right', fontsize=22, fontweight='bold')
                plt.yticks(fontsize=22, fontweight='bold')
                
                # Add subtle grid for readability
                plt.grid(axis='y', alpha=0.2, linestyle='--', linewidth=0.8)
                
                # Remove top and right spines for cleaner look
                ax = plt.gca()
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_linewidth(1.2)
                ax.spines['bottom'].set_linewidth(1.2)
                
                # Improve layout and margins
                plt.tight_layout()
                plt.subplots_adjust(bottom=0.15)  # Extra space for rotated labels
                
                plt.savefig(combined_dir / "group_category_distribution.png", 
                           dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close()
        
        # Generate total dataset word cloud (if not skipped)
        if skip_wordcloud:
            print("Skipping word cloud generation (--no_word flag)")
        elif self.metrics["total"]["expression_texts"] and WORDCLOUD_AVAILABLE:
            print("Creating total dataset expression word cloud...")
            # Combine all expression texts into a single string
            all_text = " ".join(self.metrics["total"]["expression_texts"])
            
            # Create word cloud
            wordcloud = WordCloud(
                width=1600, 
                height=800, 
                background_color='white',
                max_words=200,
                colormap='viridis',
                relative_scaling=0.5,
                random_state=42
            ).generate(all_text)
            
            # Plot word cloud
            plt.figure(figsize=(20, 10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(combined_dir / "expression_wordcloud.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    Total word cloud saved ({len(self.metrics['total']['expression_texts']):,} expressions)")
        elif self.metrics["total"]["expression_texts"] and not WORDCLOUD_AVAILABLE:
            print("    Skipping total word cloud (wordcloud package not available)")
        
        print(f"Visualizations saved to {output_dir}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate dataset metrics and visualizations")
    parser.add_argument("--no_word", action="store_true", 
                        help="Skip word cloud generation for faster processing")
    parser.add_argument("--dataset_path", type=str, 
                        default="/cfs/home/u035679/datasets/aeriald",
                        help="Path to the dataset directory")
    
    args = parser.parse_args()
    
    # Path to the final dataset directory (after all processing steps)
    # The directory structure is:
    # dataset/
    #   ├── patches/                    # Step 1: Initial patches
    #   ├── patches_rules/             # Step 2: Added rules
    #   ├── patches_rules_expressions/  # Step 3: Generated expressions
    #   └── patches_rules_expressions_unique/  # Step 4: Filtered unique expressions
    #       ├── train/
    #       │   ├── images/
    #       │   └── annotations/
    #       └── val/
    #           ├── images/
    #           └── annotations/
    
    dataset_path = args.dataset_path
    
    print(f"Analyzing dataset at: {dataset_path}")
    if args.no_word:
        print("Word cloud generation will be skipped")
    
    # Initialize and run analysis
    metrics = DatasetMetrics(dataset_path)
    metrics.analyze_dataset()
    
    # Generate report (save to same directory as this script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, "dataset_metrics_report.txt")
    report = metrics.generate_report(report_path)
    print("\nReport Preview:")
    print("=" * 50)
    print(report[:500] + "...\n")  # Show first 500 characters of report
    
    # Generate visualizations (save to same directory as this script)
    plots_dir = os.path.join(script_dir, "dataset_metrics_plots")
    metrics.plot_distributions(plots_dir, skip_wordcloud=args.no_word)

if __name__ == "__main__":
    main() 