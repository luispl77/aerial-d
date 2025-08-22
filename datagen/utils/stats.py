import os
import json
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
import numpy as np
import multiprocessing
from multiprocessing import Pool, Value
from tqdm import tqdm
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

def get_source_dataset(filename):
    """Determine source dataset from filename"""
    if filename.startswith('P'):
        return 'iSAID'
    elif filename.startswith('L'):
        return 'LoveDA'
    else:
        return 'Unknown'

def analyze_expression_features(expression_text):
    """Analyze expression to determine feature types"""
    features = {
        'category': True,  # All expressions refer to categorized objects, so always True
        'position': False,
        'extreme': False,
        'size': False,
        'color': False,
        'relationship': False
    }
    
    expr_lower = expression_text.lower()
    
    # Position keywords
    position_words = ['top', 'bottom', 'left', 'right', 'center', 'middle', 'upper', 'lower', 
                     'corner', 'side', 'north', 'south', 'east', 'west']
    if any(word in expr_lower for word in position_words):
        features['position'] = True
    
    # Extreme keywords
    extreme_words = ['largest', 'smallest', 'biggest', 'tiniest', 'most', 'least', 'first', 'last',
                    'leftmost', 'rightmost', 'topmost', 'bottommost', 'northernmost', 'southernmost']
    if any(word in expr_lower for word in extreme_words):
        features['extreme'] = True
    
    # Size keywords
    size_words = ['large', 'small', 'big', 'tiny', 'huge', 'massive', 'little']
    if any(word in expr_lower for word in size_words):
        features['size'] = True
    
    # Color keywords
    color_words = ['white', 'black', 'red', 'blue', 'green', 'yellow', 'gray', 'grey', 'brown',
                   'light', 'dark', 'bright', 'pale']
    if any(word in expr_lower for word in color_words):
        features['color'] = True
    
    # Relationship keywords
    relationship_words = ['to the', 'next to', 'near', 'close to', 'adjacent', 'beside', 'between',
                         'above', 'below', 'left of', 'right of', 'in front of', 'behind']
    if any(phrase in expr_lower for phrase in relationship_words):
        features['relationship'] = True
    
    return features

def analyze_single_xml(xml_file_path):
    """Analyze a single XML file and return metrics dictionary"""
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        
        filename = root.find('filename').text
        source_dataset = get_source_dataset(filename)
        
        # Initialize metrics for this file
        file_metrics = {
            'filename': filename,
            'source_dataset': source_dataset,
            'individual_objects': 0,
            'groups': 0,
            'individual_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
            'group_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
            'category_stats': {},
            'expression_taxonomy': {}
        }
        
        # Analyze individual objects
        for obj in root.findall('.//object'):
            expressions = obj.findall('.//expression')
            if not expressions:
                continue  # Skip objects without expressions
            
            # Count individual objects with expressions
            file_metrics['individual_objects'] += 1
            
            # Get category
            category = obj.find('name').text
            if category not in file_metrics['category_stats']:
                file_metrics['category_stats'][category] = {
                    'individual_instances': 0,
                    'groups': 0,
                    'instance_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                    'group_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                    'source_dataset': source_dataset
                }
            file_metrics['category_stats'][category]['individual_instances'] += 1
            
            # Analyze expressions for this object
            for expr in expressions:
                # Determine expression type
                if expr.get('type') == 'enhanced':
                    expr_type = 'enhanced'
                elif expr.get('type') == 'unique':
                    expr_type = 'unique'
                elif expr.get('id') is not None:
                    expr_type = 'original'  # Has id attribute (original rule-based)
                else:
                    expr_type = 'original'  # Default fallback
                
                file_metrics['individual_expressions'][expr_type] += 1
                file_metrics['individual_expressions']['total'] += 1
                file_metrics['category_stats'][category]['instance_expressions'][expr_type] += 1
                file_metrics['category_stats'][category]['instance_expressions']['total'] += 1
                
                # Analyze expression features (only for original expressions)
                if expr_type == 'original':
                    features = analyze_expression_features(expr.text)
                    feature_combo = '_'.join([k for k, v in features.items() if v])
                    if feature_combo:
                        if feature_combo not in file_metrics['expression_taxonomy']:
                            file_metrics['expression_taxonomy'][feature_combo] = {'individual': 0, 'group': 0, 'total': 0}
                        file_metrics['expression_taxonomy'][feature_combo]['individual'] += 1
                        file_metrics['expression_taxonomy'][feature_combo]['total'] += 1
        
        # Analyze groups
        groups_elem = root.find('groups')
        if groups_elem is not None:
            for group in groups_elem.findall('group'):
                expressions = group.findall('expressions/expression')
                if not expressions:
                    continue
                
                # Count groups with expressions
                file_metrics['groups'] += 1
                
                # Get group category
                category_elem = group.find('category')
                if category_elem is not None:
                    category = category_elem.text.lower().replace('_', ' ')
                    if category not in file_metrics['category_stats']:
                        file_metrics['category_stats'][category] = {
                            'individual_instances': 0,
                            'groups': 0,
                            'instance_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                            'group_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                            'source_dataset': source_dataset
                        }
                    file_metrics['category_stats'][category]['groups'] += 1
                    
                    # Analyze group expressions
                    for expr in expressions:
                        # Determine expression type
                        if expr.get('type') == 'enhanced':
                            expr_type = 'enhanced'
                        elif expr.get('type') == 'unique':
                            expr_type = 'unique'
                        elif expr.get('id') is not None:
                            expr_type = 'original'  # Has id attribute (original rule-based)
                        else:
                            expr_type = 'original'  # Default fallback
                        
                        file_metrics['group_expressions'][expr_type] += 1
                        file_metrics['group_expressions']['total'] += 1
                        file_metrics['category_stats'][category]['group_expressions'][expr_type] += 1
                        file_metrics['category_stats'][category]['group_expressions']['total'] += 1
                        
                        # Analyze expression features (only for original expressions)
                        if expr_type == 'original':
                            features = analyze_expression_features(expr.text)
                            feature_combo = '_'.join([k for k, v in features.items() if v])
                            if feature_combo:
                                if feature_combo not in file_metrics['expression_taxonomy']:
                                    file_metrics['expression_taxonomy'][feature_combo] = {'individual': 0, 'group': 0, 'total': 0}
                                file_metrics['expression_taxonomy'][feature_combo]['group'] += 1
                                file_metrics['expression_taxonomy'][feature_combo]['total'] += 1
        
        return file_metrics
        
    except Exception as e:
        print(f"Error analyzing {xml_file_path}: {str(e)}")
        return None

class AerialDStats:
    def __init__(self, dataset_path: str):
        """Initialize with path to the aeriald dataset directory."""
        self.dataset_path = Path(dataset_path)
        self.train_path = self.dataset_path / "train"
        self.val_path = self.dataset_path / "val"
        
        # Initialize metrics
        self.stats = {
            'train': self._init_stats_dict(),
            'val': self._init_stats_dict(),
            'total': self._init_stats_dict()
        }
        
        self.num_workers = multiprocessing.cpu_count()
    
    def _init_stats_dict(self):
        """Initialize stats dictionary for a split."""
        return {
            'total_patches': 0,
            'individual_objects': 0,
            'groups': 0,
            'individual_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
            'group_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
            'total_samples': 0,  # individual_objects + groups
            'category_stats': {},
            'expression_taxonomy': {}
        }
    
    def analyze_dataset(self):
        """Analyze the entire dataset and return stats."""
        print("Starting AerialD dataset analysis...")
        
        # Analyze train and val splits
        print("\nAnalyzing training split...")
        self._analyze_split(self.train_path, "train")
        print("\nAnalyzing validation split...")
        self._analyze_split(self.val_path, "val")
        
        print("\nCalculating total stats...")
        # Calculate total stats
        self._calculate_totals()
        
        print("Analysis complete!")
        return self.stats
    
    def _analyze_split(self, split_path: Path, split_name: str):
        """Analyze a specific split (train/val) using multiprocessing."""
        # Get all XML files
        xml_files = list(split_path.glob("**/*.xml"))
        self.stats[split_name]['total_patches'] = len(xml_files)
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
    
    def _merge_results(self, results, split_name: str):
        """Merge results from all worker processes."""
        print(f"Merging results from {len(results)} processed files...")
        
        for file_result in results:
            if file_result is None:
                continue
            
            # Merge scalar values
            self.stats[split_name]['individual_objects'] += file_result['individual_objects']
            self.stats[split_name]['groups'] += file_result['groups']
            
            # Merge expression counts
            for expr_type in ['original', 'enhanced', 'unique', 'total']:
                self.stats[split_name]['individual_expressions'][expr_type] += file_result['individual_expressions'][expr_type]
                self.stats[split_name]['group_expressions'][expr_type] += file_result['group_expressions'][expr_type]
            
            # Merge category stats
            for category, cat_stats in file_result['category_stats'].items():
                if category not in self.stats[split_name]['category_stats']:
                    self.stats[split_name]['category_stats'][category] = {
                        'individual_instances': 0,
                        'groups': 0,
                        'instance_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                        'group_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                        'source_dataset': cat_stats['source_dataset']
                    }
                
                for key in ['individual_instances', 'groups']:
                    self.stats[split_name]['category_stats'][category][key] += cat_stats[key]
                
                for expr_type in ['original', 'enhanced', 'unique', 'total']:
                    self.stats[split_name]['category_stats'][category]['instance_expressions'][expr_type] += cat_stats['instance_expressions'][expr_type]
                    self.stats[split_name]['category_stats'][category]['group_expressions'][expr_type] += cat_stats['group_expressions'][expr_type]
            
            # Merge expression taxonomy
            for feature_combo, combo_stats in file_result['expression_taxonomy'].items():
                if feature_combo not in self.stats[split_name]['expression_taxonomy']:
                    self.stats[split_name]['expression_taxonomy'][feature_combo] = {'individual': 0, 'group': 0, 'total': 0}
                for key in ['individual', 'group', 'total']:
                    self.stats[split_name]['expression_taxonomy'][feature_combo][key] += combo_stats[key]
        
        # Calculate total samples
        self.stats[split_name]['total_samples'] = (
            self.stats[split_name]['individual_objects'] + 
            self.stats[split_name]['groups']
        )
    
    def _calculate_totals(self):
        """Calculate total stats across train and val splits."""
        # Basic counts
        self.stats['total']['total_patches'] = (
            self.stats['train']['total_patches'] + 
            self.stats['val']['total_patches']
        )
        
        self.stats['total']['individual_objects'] = (
            self.stats['train']['individual_objects'] + 
            self.stats['val']['individual_objects']
        )
        
        self.stats['total']['groups'] = (
            self.stats['train']['groups'] + 
            self.stats['val']['groups']
        )
        
        self.stats['total']['total_samples'] = (
            self.stats['train']['total_samples'] + 
            self.stats['val']['total_samples']
        )
        
        # Expression counts
        for expr_type in ['original', 'enhanced', 'unique', 'total']:
            self.stats['total']['individual_expressions'][expr_type] = (
                self.stats['train']['individual_expressions'][expr_type] + 
                self.stats['val']['individual_expressions'][expr_type]
            )
            
            self.stats['total']['group_expressions'][expr_type] = (
                self.stats['train']['group_expressions'][expr_type] + 
                self.stats['val']['group_expressions'][expr_type]
            )
        
        # Category stats
        all_categories = set(list(self.stats['train']['category_stats'].keys()) + 
                           list(self.stats['val']['category_stats'].keys()))
        
        for category in all_categories:
            train_stats = self.stats['train']['category_stats'].get(category, {
                'individual_instances': 0, 'groups': 0,
                'instance_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                'group_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                'source_dataset': 'Unknown'
            })
            val_stats = self.stats['val']['category_stats'].get(category, {
                'individual_instances': 0, 'groups': 0,
                'instance_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                'group_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                'source_dataset': 'Unknown'
            })
            
            if category not in self.stats['total']['category_stats']:
                self.stats['total']['category_stats'][category] = {
                    'individual_instances': 0, 'groups': 0,
                    'instance_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                    'group_expressions': {'original': 0, 'enhanced': 0, 'unique': 0, 'total': 0},
                    'source_dataset': 'Unknown'
                }
            
            self.stats['total']['category_stats'][category]['individual_instances'] = (
                train_stats['individual_instances'] + val_stats['individual_instances']
            )
            
            self.stats['total']['category_stats'][category]['groups'] = (
                train_stats['groups'] + val_stats['groups']
            )
            
            for expr_type in ['original', 'enhanced', 'unique', 'total']:
                self.stats['total']['category_stats'][category]['instance_expressions'][expr_type] = (
                    train_stats['instance_expressions'][expr_type] + 
                    val_stats['instance_expressions'][expr_type]
                )
                
                self.stats['total']['category_stats'][category]['group_expressions'][expr_type] = (
                    train_stats['group_expressions'][expr_type] + 
                    val_stats['group_expressions'][expr_type]
                )
            
            # Determine source dataset for total
            if train_stats['source_dataset'] == val_stats['source_dataset']:
                self.stats['total']['category_stats'][category]['source_dataset'] = train_stats['source_dataset']
            else:
                self.stats['total']['category_stats'][category]['source_dataset'] = 'Mixed'
        
        # Expression taxonomy
        all_features = set(list(self.stats['train']['expression_taxonomy'].keys()) + 
                          list(self.stats['val']['expression_taxonomy'].keys()))
        
        for feature_combo in all_features:
            train_taxonomy = self.stats['train']['expression_taxonomy'].get(feature_combo, {'individual': 0, 'group': 0, 'total': 0})
            val_taxonomy = self.stats['val']['expression_taxonomy'].get(feature_combo, {'individual': 0, 'group': 0, 'total': 0})
            
            if feature_combo not in self.stats['total']['expression_taxonomy']:
                self.stats['total']['expression_taxonomy'][feature_combo] = {'individual': 0, 'group': 0, 'total': 0}
            
            for key in ['individual', 'group', 'total']:
                self.stats['total']['expression_taxonomy'][feature_combo][key] = (
                    train_taxonomy[key] + val_taxonomy[key]
                )
    
    def generate_tables(self, output_path: str = None):
        """Generate the 4 required tables."""
        print("\nGenerating tables...")
        tables = []
        
        # Table 1: Dataset Statistics Summary
        tables.append("=== Table: Dataset Statistics Summary (tab:dataset_stats) ===")
        tables.append("")
        tables.append("| Metric | Train | Val | Total |")
        tables.append("|--------|-------|-----|-------|")
        
        # Calculate averages
        def safe_avg(total, count):
            return total / count if count > 0 else 0
        
        for split in ['train', 'val', 'total']:
            avg_expr_per_individual = safe_avg(
                self.stats[split]['individual_expressions']['total'],
                self.stats[split]['individual_objects']
            )
            avg_expr_per_group = safe_avg(
                self.stats[split]['group_expressions']['total'],
                self.stats[split]['groups']
            )
            self.stats[split]['avg_expressions_per_individual'] = avg_expr_per_individual
            self.stats[split]['avg_expressions_per_group'] = avg_expr_per_group
        
        metrics = [
            ("Total Patches", "total_patches"),
            ("Individual Objects with Expressions", "individual_objects"),
            ("Individual Expressions", "individual_expressions.total"),
            ("Groups with Expressions", "groups"),
            ("Group Expressions", "group_expressions.total"),
            ("Total Samples", "total_samples"),
            ("Avg. Expressions per Individual Object", "avg_expressions_per_individual"),
            ("Avg. Expressions per Group", "avg_expressions_per_group")
        ]
        
        for metric_name, metric_key in metrics:
            row = [metric_name]
            for split in ['train', 'val', 'total']:
                if '.' in metric_key:
                    keys = metric_key.split('.')
                    value = self.stats[split][keys[0]][keys[1]]
                else:
                    value = self.stats[split][metric_key]
                
                if isinstance(value, float):
                    row.append(f"{value:.2f}")
                else:
                    row.append(str(value))
            
            tables.append("| " + " | ".join(row) + " |")
        
        # Table 2: Object Category Distribution
        tables.append("\n=== Table: Object Category Distribution (tab:category_dist) ===")
        tables.append("")
        tables.append("| Category | Individual Instances | Groups | Instance Expressions | Group Expressions | Source Dataset |")
        tables.append("|----------|---------------------|--------|---------------------|-------------------|----------------|")
        
        # Get all categories and sort them
        all_categories = set()
        for split in ['train', 'val']:
            all_categories.update(self.stats[split]['category_stats'].keys())
        
        # Define category order (iSAID first, then LoveDA)
        isaid_categories = ['ship', 'large vehicle', 'small vehicle', 'building', 'storage tank', 
                           'harbor', 'swimming pool', 'tennis court', 'soccer ball field', 
                           'roundabout', 'basketball court', 'bridge', 'ground track field', 
                           'plane', 'helicopter']
        
        loveda_categories = ['building', 'water', 'barren land', 'agricultural area', 'forest area', 'road']
        
        # Sort categories by putting iSAID first, then LoveDA
        sorted_categories = []
        for cat in isaid_categories:
            if cat in all_categories:
                sorted_categories.append(cat)
        for cat in loveda_categories:
            if cat in all_categories and cat not in sorted_categories:
                sorted_categories.append(cat)
        # Add any remaining categories
        for cat in sorted(all_categories):
            if cat not in sorted_categories:
                sorted_categories.append(cat)
        
        for category in sorted_categories:
            total_stats = self.stats['total']['category_stats'][category]
            row = [
                category.title(),
                str(total_stats['individual_instances']),
                str(total_stats['groups']),
                str(total_stats['instance_expressions']['total']),
                str(total_stats['group_expressions']['total']),
                total_stats['source_dataset']
            ]
            tables.append("| " + " | ".join(row) + " |")
        
        # Table 3: Expression Taxonomy
        tables.append("\n=== Table: Expression Taxonomy (tab:expression_types) ===")
        tables.append("")
        tables.append("| Feature Combination | Individual Instance Expressions | Group Expressions | Total Count |")
        tables.append("|-------------------|----------------------------------|-------------------|-------------|")
        
        # Sort feature combinations
        total_taxonomy = self.stats['total']['expression_taxonomy']
        sorted_features = sorted(total_taxonomy.keys(), key=lambda x: total_taxonomy[x]['total'], reverse=True)
        
        for feature_combo in sorted_features:
            combo_stats = total_taxonomy[feature_combo]
            # Format feature combination nicely
            formatted_combo = feature_combo.replace('_', ' + ').title()
            if not formatted_combo:
                formatted_combo = "None"
            
            row = [
                formatted_combo,
                str(combo_stats['individual']),
                str(combo_stats['group']),
                str(combo_stats['total'])
            ]
            tables.append("| " + " | ".join(row) + " |")
        
        # Table 4: LLM Enhancement Stats
        tables.append("\n=== Table: LLM Enhancement Stats (tab:llm_enhancement_stats) ===")
        tables.append("")
        tables.append("| Expression Type | Train | Val | Total |")
        tables.append("|-----------------|-------|-----|-------|")
        
        enhancement_types = [
            ("Rule-Based Expressions", "original"),
            ("LLM Enhanced (Language Variations)", "enhanced"),
            ("LLM Unique (Visual Details)", "unique"),
            ("Total Expressions", "total")
        ]
        
        for type_name, type_key in enhancement_types:
            row = [type_name]
            for split in ['train', 'val', 'total']:
                total_expressions = (
                    self.stats[split]['individual_expressions'][type_key] + 
                    self.stats[split]['group_expressions'][type_key]
                )
                row.append(str(total_expressions))
            
            tables.append("| " + " | ".join(row) + " |")
        
        tables_text = "\n".join(tables)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(tables_text)
            print(f"Tables saved to {output_path}")
        
        return tables_text

def main():
    dataset_path = "/cfs/home/u035679/datasets/aeriald"
    
    print(f"Analyzing AerialD dataset at: {dataset_path}")
    
    # Initialize and run analysis
    stats = AerialDStats(dataset_path)
    stats.analyze_dataset()
    
    # Generate tables
    script_dir = os.path.dirname(os.path.abspath(__file__))
    tables_path = os.path.join(script_dir, "aeriald_dataset_tables.txt")
    tables = stats.generate_tables(tables_path)
    
    print("\nTables Preview:")
    print("=" * 50)
    print(tables[:1000] + "..." if len(tables) > 1000 else tables)

if __name__ == "__main__":
    main()