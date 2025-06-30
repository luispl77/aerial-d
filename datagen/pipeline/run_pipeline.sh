#!/bin/bash

# Print header
echo "============================================="
echo "Running RRSIS Dataset Generation Pipeline"
echo "============================================="

# Start timing the full pipeline
START_TIME=$(date +%s)

# Default values for step 1 arguments
NUM_IMAGES=""
START_IMAGE_ID=""
END_IMAGE_ID=""
NUM_WORKERS=""
RANDOM_SEED=""
CLEAN=false
ZIP=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_images)
            NUM_IMAGES="--num_images $2"
            shift 2
            ;;
        --start_image_id)
            START_IMAGE_ID="--start_image_id $2"
            shift 2
            ;;
        --end_image_id)
            END_IMAGE_ID="--end_image_id $2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="--num_workers $2"
            shift 2
            ;;
        --random_seed)
            RANDOM_SEED="--random_seed $2"
            shift 2
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        --zip)
            ZIP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--num_images N] [--start_image_id X] [--end_image_id Y] [--num_workers W] [--random_seed S] [--clean] [--zip]"
            echo "Note: --num_images N will select N images from each split for iSAID, LoveDA, and DeepGlobe datasets"
            echo "      --clean will delete the dataset directory before starting"
            echo "      --zip will create a zip archive of the final dataset"
            exit 1
            ;;
    esac
done

# Clean dataset directory if requested
if [ "$CLEAN" = true ]; then
    echo -e "\nCleaning dataset directory..."
    rm -rf dataset/
    echo "Dataset directory cleaned"
fi

# Function to run a script and check its exit status
run_script() {
    local script_name=$1
    local script_args=$2
    
    echo -e "\nRunning $script_name..."
    if [ -n "$script_args" ]; then
        python "pipeline/$script_name" $script_args
    else
        python "pipeline/$script_name"
    fi
    
    if [ $? -ne 0 ]; then
        echo "Error: $script_name failed to execute properly"
        exit 1
    fi
    echo "Completed $script_name successfully"
}

# Combine step 1 arguments
STEP1_ARGS="$NUM_IMAGES $START_IMAGE_ID $END_IMAGE_ID $NUM_WORKERS $RANDOM_SEED"

# Run each pipeline script in sequence
run_script "1_isaid_patches.py" "$STEP1_ARGS"
run_script "2_loveda_patches.py" "$STEP1_ARGS"
run_script "3_deepglobe_patches.py" "$STEP1_ARGS"
run_script "4_add_rules.py"
run_script "5_generate_all_expressions.py"
run_script "6_filter_unique.py"
run_script "7_historic_filter.py"

# Run zip script if requested
if [ "$ZIP" = true ]; then
    echo -e "\nCreating zip archive of dataset..."
    python "utils/zip_dataset.py"
    
    if [ $? -ne 0 ]; then
        echo "Warning: zip_dataset.py failed to execute properly"
    else
        echo "Dataset zip archive created successfully"
    fi
fi

# Calculate total execution time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

# Format duration into hours, minutes, seconds
HOURS=$((DURATION / 3600))
MINUTES=$(( (DURATION % 3600) / 60 ))
SECONDS=$((DURATION % 60))

echo -e "\n============================================="
echo "Pipeline completed successfully!"
echo "Outputs are in the dataset/ directory"
if [ "$ZIP" = true ]; then
    echo "Zip archive available in the output directory"
fi
echo "Total execution time: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "=============================================" 