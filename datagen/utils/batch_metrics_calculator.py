#!/usr/bin/env python3
import os
import sys
import json
import argparse
import datetime
from tqdm import tqdm

# Constants
RESULTS_FILE = "dataset/batch_inference_results.jsonl"  # Fixed results file path

def log_with_timestamp(message):
    """Add timestamp to log messages"""
    timestamp = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {message}")

def calculate_metrics(input_file):
    """Calculate metrics from batch prediction output file"""
    log_with_timestamp(f"Processing batch prediction results from: {input_file}")
    
    # Initialize counters
    stats = {
        'total_requests': 0,
        'successful_requests': 0,
        'total_instances': 0,
        'total_original_expressions': 0,
        'total_enhanced_expressions': 0,
        'total_input_tokens': 0,
        'total_output_tokens': 0,
        'total_text_tokens': 0,
        'total_image_tokens': 0,
        'average_input_tokens': 0,
        'average_output_tokens': 0,
        'total_tokens': 0,
        'input_cost': 0,
        'output_cost': 0,
        'total_cost': 0,
        'unique_patches': set()  # Track unique patch filenames
    }
    
    # Process the file line by line (JSONL format)
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # Count lines to set up progress bar
            line_count = sum(1 for _ in open(input_file, 'r'))
            f.seek(0)  # Reset file pointer
            
            for line in tqdm(f, total=line_count, desc="Processing responses"):
                try:
                    # Parse JSON
                    response_data = json.loads(line)
                    stats['total_requests'] += 1
                    
                    # Try to extract patch filename from fileUri in the request
                    try:
                        if 'request' in response_data and 'contents' in response_data['request']:
                            for content in response_data['request']['contents']:
                                if 'parts' in content:
                                    for part in content['parts']:
                                        if part.get('fileData') and part['fileData'].get('fileUri'):
                                            file_uri = part['fileData']['fileUri']
                                            # Extract filename from gs:// URI
                                            file_name = file_uri.split('/')[-1]
                                            # Extract patch ID (typically pattern like P0010_patch_000141)
                                            patch_id = '_'.join(file_name.split('_')[:3])
                                            stats['unique_patches'].add(patch_id)
                    except Exception as e:
                        log_with_timestamp(f"Error extracting patch ID: {e}")
                    
                    # Check if we have a valid response
                    if 'response' in response_data and 'candidates' in response_data['response']:
                        candidate = response_data['response']['candidates'][0]
                        
                        if 'content' in candidate and 'parts' in candidate['content']:
                            stats['successful_requests'] += 1
                            
                            # Extract token counts from usageMetadata
                            if 'usageMetadata' in response_data['response']:
                                usage = response_data['response']['usageMetadata']
                                
                                # Input tokens
                                if 'promptTokenCount' in usage:
                                    input_tokens = int(usage['promptTokenCount'])
                                    stats['total_input_tokens'] += input_tokens
                                
                                # Output tokens
                                if 'candidatesTokenCount' in usage:
                                    output_tokens = int(usage['candidatesTokenCount'])
                                    stats['total_output_tokens'] += output_tokens
                                
                                # Total tokens
                                if 'totalTokenCount' in usage:
                                    total_tokens = int(usage['totalTokenCount'])
                                else:
                                    total_tokens = input_tokens + output_tokens
                                
                                # Token details by modality
                                if 'promptTokensDetails' in usage:
                                    for detail in usage['promptTokensDetails']:
                                        if detail['modality'] == 'TEXT':
                                            stats['total_text_tokens'] += int(detail['tokenCount'])
                                        elif detail['modality'] == 'IMAGE':
                                            stats['total_image_tokens'] += int(detail['tokenCount'])
                                
                                if 'candidatesTokensDetails' in usage:
                                    for detail in usage['candidatesTokensDetails']:
                                        if detail['modality'] == 'TEXT':
                                            stats['total_text_tokens'] += int(detail['tokenCount'])
                            
                            # Parse prediction content
                            try:
                                # Extract the JSON content from the text field
                                content_text = candidate['content']['parts'][0]['text'].strip()
                                output_data = json.loads(content_text)
                                
                                # Count instance
                                stats['total_instances'] += 1
                                
                                # Count expressions
                                if 'enhanced_expressions' in output_data:
                                    for expr_group in output_data['enhanced_expressions']:
                                        stats['total_original_expressions'] += 1
                                        if 'enhanced_versions' in expr_group:
                                            stats['total_enhanced_expressions'] += len(expr_group['enhanced_versions'])
                            except (json.JSONDecodeError, KeyError, TypeError) as e:
                                log_with_timestamp(f"Error parsing prediction content: {e}")
                                continue
                
                except json.JSONDecodeError as e:
                    log_with_timestamp(f"Error parsing JSON line: {e}")
                    continue
                except Exception as e:
                    log_with_timestamp(f"Error processing line: {e}")
                    continue
    
    except FileNotFoundError:
        log_with_timestamp(f"Error: File not found: {input_file}")
        return None
    
    # Add count of unique patches
    stats['unique_patch_count'] = len(stats['unique_patches'])
    # Remove the set from stats as it's not needed anymore and can't be easily displayed
    del stats['unique_patches']
    
    # Calculate derived statistics
    if stats['successful_requests'] > 0:
        stats['average_input_tokens'] = stats['total_input_tokens'] / stats['successful_requests']
        stats['average_output_tokens'] = stats['total_output_tokens'] / stats['successful_requests']
    
    stats['total_tokens'] = stats['total_input_tokens'] + stats['total_output_tokens']
    
    # Calculate cost estimates (using Gemini 2.0 Flash pricing)
    # Flash: $0.05/1M input tokens, $0.20/1M output tokens
    stats['input_cost'] = (stats['total_input_tokens'] / 1_000_000) * 0.05
    stats['output_cost'] = (stats['total_output_tokens'] / 1_000_000) * 0.20
    stats['total_cost'] = stats['input_cost'] + stats['output_cost']
    
    # Calculate extrapolation factors
    TOTAL_DATASET_PATCHES = 71721
    if stats['unique_patch_count'] > 0:
        stats['extrapolation_factor'] = TOTAL_DATASET_PATCHES / stats['unique_patch_count']
        
        # Calculate extrapolated metrics
        stats['extrapolated_requests'] = int(stats['total_requests'] * stats['extrapolation_factor'])
        stats['extrapolated_instances'] = int(stats['total_instances'] * stats['extrapolation_factor'])
        stats['extrapolated_tokens'] = int(stats['total_tokens'] * stats['extrapolation_factor'])
        stats['extrapolated_cost'] = stats['total_cost'] * stats['extrapolation_factor']
        
        # Calculate extrapolated expression counts
        stats['extrapolated_original_expressions'] = int(stats['total_original_expressions'] * stats['extrapolation_factor'])
        stats['extrapolated_enhanced_expressions'] = int(stats['total_enhanced_expressions'] * stats['extrapolation_factor'])
        stats['extrapolated_total_expressions'] = stats['extrapolated_original_expressions'] + stats['extrapolated_enhanced_expressions']
    
    return stats

def print_metrics_report(stats):
    """Print metrics report to console"""
    if not stats:
        return
    
    report = f"""
BATCH PREDICTION METRICS
=======================
Date: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

REQUESTS
--------
Total API Requests: {stats['total_requests']}
Successful Requests: {stats['successful_requests']}
Success Rate: {(stats['successful_requests'] / stats['total_requests'] * 100) if stats['total_requests'] > 0 else 0:.2f}%

COUNTS
------
Total Instances Processed: {stats['total_instances']}
Total Original Expressions: {stats['total_original_expressions']}
Total Enhanced Expressions: {stats['total_enhanced_expressions']}
Average Enhanced Expressions per Original: {(stats['total_enhanced_expressions'] / stats['total_original_expressions']) if stats['total_original_expressions'] > 0 else 0:.2f}
Unique Patches Processed: {stats['unique_patch_count']}

TOKEN USAGE
-----------
Average Input Tokens per Request: {stats['average_input_tokens']:.2f}
Average Output Tokens per Request: {stats['average_output_tokens']:.2f}
Average Total Tokens per Request: {(stats['average_input_tokens'] + stats['average_output_tokens']):.2f}
Total Input Tokens: {stats['total_input_tokens']}
Total Output Tokens: {stats['total_output_tokens']}
Total Tokens: {stats['total_tokens']}
Text Tokens: {stats['total_text_tokens']}
Image Tokens: {stats['total_image_tokens']}

COST ESTIMATE (Gemini 2.0 Flash)
------------
Input Cost (at $0.05 per million tokens): ${stats['input_cost']:.4f}
Output Cost (at $0.20 per million tokens): ${stats['output_cost']:.4f}
Total Estimated Cost: ${stats['total_cost']:.4f}

EXTRAPOLATION TO FULL DATASET (71,721 patches)
----------------------------------
Extrapolation Factor: {stats.get('extrapolation_factor', 0):.2f}x
Estimated Total Requests: {stats.get('extrapolated_requests', 0):,}
Estimated Total Instances: {stats.get('extrapolated_instances', 0):,}
Estimated Total Tokens: {stats.get('extrapolated_tokens', 0):,}
Estimated Total Cost: ${stats.get('extrapolated_cost', 0):.2f}
Estimated Original (Rule-Based) Expressions: {stats.get('extrapolated_original_expressions', 0):,}
Estimated Enhanced (LLM) Expressions: {stats.get('extrapolated_enhanced_expressions', 0):,}
Estimated Total Expressions (Combined): {stats.get('extrapolated_total_expressions', 0):,}
"""

    # Print to console
    print(report)

def main():
    parser = argparse.ArgumentParser(description='Calculate metrics from batch prediction results')
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(RESULTS_FILE):
        log_with_timestamp(f"Error: Input file {RESULTS_FILE} not found")
        sys.exit(1)
    
    # Calculate metrics
    stats = calculate_metrics(RESULTS_FILE)
    
    if stats:
        # Print report to console
        print_metrics_report(stats)
    else:
        log_with_timestamp("Failed to calculate metrics")
        sys.exit(1)

if __name__ == "__main__":
    main() 