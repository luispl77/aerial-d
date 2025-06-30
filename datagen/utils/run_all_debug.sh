#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Print header
echo "============================================="
echo "Running All Debug Visualization Scripts"
echo "============================================="

# Function to run a script and check its exit status
run_script() {
    echo -e "\nRunning $1..."
    python "${SCRIPT_DIR}/$1"
    if [ $? -ne 0 ]; then
        echo "Error: $1 failed to execute properly"
        exit 1
    fi
    echo "Completed $1 successfully"
}

# Run each debug script in sequence
run_script "1_debug_create_patches.py"
run_script "2_debug_add_rules.py"
run_script "3_debug_generate_all_expressions.py"
run_script "4_debug_filter_unique.py"

echo -e "\n============================================="
echo "All debug visualizations completed successfully!"
echo "Outputs are in the debug/ directory"
echo "=============================================" 