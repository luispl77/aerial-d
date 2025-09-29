#!/bin/bash

# Rsync script to sync aerialseg project from remote server
# Excludes models/ directory and .git folder to avoid large file transfers

rsync -avzh \
    --exclude 'models/' \
    --exclude '.git/' \
    --exclude '*.pyc' \
    --exclude '__pycache__/' \
    --exclude '.DS_Store' \
    --exclude 'datagen/isaid/' \
    --exclude 'datagen/rrsisd/' \
    --exclude 'datagen/refsegrs' \
    --exclude 'datagen/NWPU-Refer' \
    --exclude 'datagen/Urban1960SatBench' \
    --exclude 'datagen/saved_visualizations' \
    --exclude 'datagen/*.zip' \
    --exclude 'datagen/*.tar' \
    --exclude 'datagen/*.gz' \
    --exclude 'datagen/patches' \
    --exclude 'datagen/patches_modified' \
    --exclude 'datagen/patches*' \
    --exclude 'datagen/.env' \
    --exclude 'datagen/gemma_pytorch' \
    --exclude 'datagen/llm_crops' \
    --exclude 'datagen/llm_crops_descriptions' \
    --exclude 'datagen/llm_crops_enhanced_expressions' \
    --exclude 'datagen/llm_crops_combined_output' \
    --exclude 'datagen/sota' \
    --exclude 'datagen/SOTA' \
    --exclude 'datagen/dataset*' \
    --exclude 'datagen/temp*' \
    --exclude 'datagen/debug*' \
    --exclude 'datagen/gemini_finetuning_data' \
    --exclude 'datagen/gemini_triplet_finetuning_data' \
    --exclude 'datagen/gen-lang*' \
    --exclude 'datagen/google-cloud-sdk' \
    --exclude 'datagen/aeriald.zip' \
    --exclude 'datagen/patch_visualization' \
    --exclude 'datagen/.vscode' \
    --exclude 'datagen/LoveDA*' \
    --exclude 'datagen/melbourne_1945_tiles/' \
    --exclude 'datagen/deepglobe/' \
    --exclude 'llm/gemma-3-4b-it' \
    --exclude 'llm/gemma-3-1b-it' \
    --exclude 'llm/gemma_pytorch' \
    --exclude 'llm/gemma_model/' \
    --exclude 'llm/debug_output/' \
    --exclude 'llm/enhanced_annotations*/' \
    --exclude 'llm/enhanced_gemma_annotations*/' \
    --exclude 'llm/enhanced*/' \
    --exclude 'llm/.env' \
    --exclude 'llm/gemma-product-simple' \
    --exclude 'llm/gemma-aerial-referring' \
    --exclude 'llm/enhanced_gemma_annotations/' \
    --exclude 'llm/gemma-aerial-referring-12b-lora' \
    --exclude 'llm/gemma-aerial-referring-4b-lora' \
    --exclude 'llm/gemma-env' \
    --exclude 'llm/merged_model*/' \
    --exclude 'llm/gemma-aerial-12b' \
    --exclude 'llm/gemma-aerial-12b_old' \
    --exclude 'llm/*/__pycache__/' \
    --exclude 'llm/debug*' \
    --exclude 'llm/temp*' \
    --exclude 'clipsam/utils/static' \
    --exclude 'clipsam/models' \
    --exclude 'clipsam/aeriald.zip' \
    --exclude 'clipsam/aeriald' \
    --exclude 'clipsam/__pycache__' \
    --exclude 'clipsam/.env' \
    --exclude 'clipsam/style_transfer_results' \
    --exclude 'clipsam/utils/filter_examples' \
    x02:~/aerialseg/ ~/aerialseg/

echo "Sync completed!"
