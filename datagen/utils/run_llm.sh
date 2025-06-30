#!/bin/bash

# Print header
echo "============================================="
echo "Running LLM Dataset Generation Pipeline"
echo "============================================="

# Start timing the full pipeline
START_TIME=$(date +%s)

# Default values for arguments
SPLIT="both"
DATASET_FOLDER="dataset"
INPUT_JSONL_TRAIN="$DATASET_FOLDER/batch_prediction/batch_prediction_requests_train.jsonl"
INPUT_JSONL_VAL="$DATASET_FOLDER/batch_prediction/batch_prediction_requests_val.jsonl"
OUTPUT_GCS="gs://aerial-bucket/batch_inference/results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --split)
            SPLIT="$2"
            shift 2
            ;;
        --dataset_folder)
            DATASET_FOLDER="$2"
            INPUT_JSONL_TRAIN="$DATASET_FOLDER/batch_prediction/batch_prediction_requests_train.jsonl"
            INPUT_JSONL_VAL="$DATASET_FOLDER/batch_prediction/batch_prediction_requests_val.jsonl"
            shift 2
            ;;
        --input_jsonl_train)
            INPUT_JSONL_TRAIN="$2"
            shift 2
            ;;
        --input_jsonl_val)
            INPUT_JSONL_VAL="$2"
            shift 2
            ;;
        --output_gcs)
            OUTPUT_GCS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--split {train,val,both}] [--dataset_folder FOLDER] [--input_jsonl_train PATH] [--input_jsonl_val PATH] [--output_gcs PATH]"
            exit 1
            ;;
    esac
done

# Function to run a script and check its exit status
run_script() {
    local script_name=$1
    local script_args=$2
    
    echo -e "\nRunning $script_name..."
    if [ -n "$script_args" ]; then
        python "llm/$script_name" $script_args
    else
        python "llm/$script_name"
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: $script_name failed to execute properly"
        exit 1
    fi
    echo "Completed $script_name successfully"
}

# Step 1: Prepare batch JSONL files
echo -e "\nStep 1: Preparing batch JSONL files..."
run_script "1_prepare_batch_jsonl.py" "--dataset_folder $DATASET_FOLDER"

# Step 2: Run batch prediction
echo -e "\nStep 2: Running batch prediction..."
BATCH_ARGS="--split $SPLIT --dataset_folder $DATASET_FOLDER --input_jsonl_train $INPUT_JSONL_TRAIN --input_jsonl_val $INPUT_JSONL_VAL --output_gcs $OUTPUT_GCS"
run_script "2_run_batch_prediction.py" "$BATCH_ARGS"

# Step 3: Parse batch results
echo -e "\nStep 3: Parsing batch results..."
run_script "3_parse_batch_results.py" "--dataset_folder $DATASET_FOLDER"

# Calculate total execution time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Format duration into hours, minutes, seconds
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo -e "\n============================================="
echo "LLM Pipeline completed successfully!"
echo "Outputs are in the $DATASET_FOLDER/batch_prediction directory"
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=============================================" 