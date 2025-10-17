#!/bin/bash

# Simple script to run test combinations
# Usage: bash run_test_combinations.sh --models "model1,model2" --datasets "dataset1,dataset2" [--historic]

MODELS=""
DATASETS=""
HISTORIC_FLAG=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --models)
            MODELS="$2"
            shift 2
            ;;
        --datasets)
            DATASETS="$2"
            shift 2
            ;;
        --historic)
            HISTORIC_FLAG="--historic"
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# Convert to arrays
IFS=',' read -ra MODEL_ARRAY <<< "$MODELS"
IFS=',' read -ra DATASET_ARRAY <<< "$DATASETS"

echo "Running combinations..."

# Run all combinations
for model in "${MODEL_ARRAY[@]}"; do
    for dataset in "${DATASET_ARRAY[@]}"; do
        echo "Running: python test.py --model_name $model --dataset_type $dataset $HISTORIC_FLAG"
        python test.py --model_name "$model" --dataset_type "$dataset" $HISTORIC_FLAG
        echo "Completed: $model on $dataset"
        echo "---"
    done
done

echo "All combinations completed"