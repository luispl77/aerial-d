# Aerial-D: Generalized Referring Expression Segmentation on Aerial Photos

## Project Overview

Aerial-D is a research project focused on **generalized referring expression segmentation of aerial photographs**. The project introduces the Aerial-D dataset (37,288 images with 1.52M referring expressions) and implements RSRefSeg (SigLIP2 + SAM architecture) for unified training across multiple datasets. The main goal is to segment aerial images using natural language descriptions covering instances, groups, and semantic regions.

### Key Components:

- **Dataset Generation Pipeline**: Rule-based and LLM-enhanced annotation generation
- **RSRefSeg Model**: SigLIP2 + SAM architecture with LoRA adapters for referring segmentation
- **LLM Enhancement**: Gemma3 and OpenAI o3 models for expression enhancement
- **Web Applications**: Flask-based visualization and annotation tools

## Project Structure

```
/Users/luispl/aerial-d/
├── datagen/                    # Dataset generation pipeline
│   ├── pipeline/              # Rule-based annotation pipeline (Steps 1-7)
│   └── utils/                 # Utilities, debug scripts, web apps
├── rsrefseg/                  # SigLIP2+SAM model implementation
│   ├── model.py              # Main model architecture
│   ├── train.py              # Training script
│   ├── test.py               # Evaluation script
│   └── utils/                # Style transfer, web apps
├── llm/                       # LLM enhancement pipeline
│   ├── gemma3_enhance.py     # Gemma3-based enhancement
│   ├── o3_enhance.py         # OpenAI O3 enhancement
│   └── gemma3_lora_finetune.py # LoRA fine-tuning for Gemma3
├── docs/                      # Project webpage files
└── .git/                     # Git repository
```

## Key Directories and Their Roles

### 1. **datagen/pipeline/** - Dataset Generation Pipeline

Sequential pipeline for creating the AerialD dataset:

1. **1_isaid_patches.py** - Extract 480x480 patches from iSAID dataset
2. **2_loveda_patches.py** - Extract patches from LoveDA dataset
3. **4_add_rules.py** - Add spatial/relational rules (3x3 grid, size, color)
4. **5_generate_all_expressions.py** - Generate referring expressions
5. **6_filter_unique.py** - Remove ambiguous/duplicate expressions
6. **7_historic_filter.py** - Apply historic imagery simulation
7. **8_vllm_enhance.py** - Optional VLLM enhancement

**Entry Point**: `./datagen/pipeline/run_pipeline.sh`

### 2. **datagen/utils/** - Utilities and Web Apps

- `app.py` - Main Flask app for viewing LLM-enhanced annotations
- `rule_viewer.py` - Viewer for rule-based annotations
- `manual_classifier.py` - Manual annotation classification tool
- `gemini_labeler.py` - Gemini-based labeling interface
- `batch_metrics_calculator.py` - Dataset statistics calculator
- Debug scripts: `1_debug_*.py` through `4_debug_*.py`

### 3. **rsrefseg/** - Model Implementation

- `model.py` - SigLipSamSegmentator architecture with domain adaptation
- `train.py` - Training script with gradient reversal layer
- `test.py` - Evaluation and visualization
- `utils/` - Style transfer, inference apps

### 4. **llm/** - LLM Enhancement

- `gemma3_enhance.py` - Gemma3-based expression enhancement
- `o3_enhance.py` - OpenAI O3 enhancement
- `gemma3_lora_finetune.py` - LoRA fine-tuning for Gemma3
- Enhanced annotation directories with generated content

## Configuration Files

### Requirements Files

- `requirements.txt` - Main project dependencies
- `datagen/requirements.txt` - Dataset generation dependencies (if separate)
- `rsrefseg/requirements.txt` - Model training dependencies (if separate)
- `llm/requirements.txt` - LLM enhancement dependencies (if separate)

### Key Dependencies

- **PyTorch** - Core ML framework
- **Transformers** - Hugging Face models (SigLIP, SAM, Gemma3)
- **OpenCV** - Image processing
- **Flask** - Web applications
- **pycocotools** - COCO format handling
- **Vertex AI** - Google Cloud LLM APIs

## Build/Test/Run Commands

### Dataset Generation

```bash
# Full pipeline (all datasets)
cd ~/aerial-d/datagen
./pipeline/run_pipeline.sh

# Partial dataset (N images per split)
./pipeline/run_pipeline.sh --num_images 100

# Skip LLM enhancement (Steps 1-6 only)
./pipeline/run_pipeline.sh --skip_step7 --clean

# With cleaning and zipping
./pipeline/run_pipeline.sh --clean --zip
```

### Model Training

```bash
cd ~/aerial-d/rsrefseg

# Basic training
python train.py --dataset_root ../datagen/dataset --custom_name aeriald_run

# Resume training
python train.py --resume
```

### Model Testing/Evaluation

```bash
cd ~/aerial-d/rsrefseg

# Test model
python test.py --model_name aeriald_run --dataset_type aeriald

# Test with published checkpoints
python test.py --model_name rsrefseg_aerial-d --dataset_type aeriald

# Visualization only
python test.py --vis_only --num_vis 50
```

### Web Applications

```bash
# Dataset browser
cd ~/aerial-d/datagen/utils  
python rule_viewer.py --port 5004

# RSRefSeg inference app
cd ~/aerial-d/rsrefseg/utils
python rsrefseg_inference_app.py --model_name aeriald_run --port 5002
```

### LLM Enhancement

```bash
cd ~/aerial-d/llm

# OpenAI O3 enhancement
python o3_enhance.py --dataset_dir ../datagen/dataset

# Gemma3 LoRA fine-tuning
python gemma3_lora_finetune.py \
  --enhanced_data_dir enhanced_annotations_o3_dual \
  --model_name gemma-aerial-12b \
  --output_dir ./gemma-aerial-12b

# Run Step 7 (vLLM enhancement)
cd ~/aerial-d/datagen
python pipeline/7_vllm_enhance.py
```

## Architecture Patterns

### 1. **Model Architecture (RSRefSeg)**

- **SigLIP2 Encoder**: Text and image feature extraction
- **SAM Decoder**: Segmentation mask generation
- **LoRA Adapters**: Parameter-efficient fine-tuning

### 2. **Dataset Pipeline Pattern**

- **Patch Extraction**: Sliding window with overlap handling
- **Rule-based Generation**: Spatial rules, size relationships, color analysis
- **Expression Generation**: Combinatorial referring expression creation
- **LLM Enhancement**: Natural language diversification

### 3. **Web Application Pattern**

- **Flask + Template Rendering**: Server-side HTML generation
- **Image Serving**: Direct file serving with caching
- **Interactive Navigation**: Patch-by-patch browsing
- **Real-time Visualization**: Matplotlib + OpenCV integration

## Existing Documentation

### Primary Documentation

- `README.md` - Main project documentation with dataset and model information
- `datagen/utils/docs.md` - Detailed pipeline documentation (if exists)
- `rsrefseg/README.md` - RSRefSeg implementation overview (if exists)
- `llm/README.md` - Gemma3 setup and usage (if exists)

### Development Notes

- `TODO.md` - Active development tasks
- `archive TODO.md` - Completed tasks archive

## Git Repository

- **Remote**: https://github.com/luisml77/aerial-d.git
- **Main Branch**: `main`
- **Current Status**: Active development with regular commits

## Hugging Face Resources

- **Dataset**: [luisml77/aerial-d](https://huggingface.co/datasets/luisml77/aerial-d)
- **Model Checkpoints**: [luisml77/rsrefseg](https://huggingface.co/luisml77/rsrefseg)
- **Gemma3 Fine-tuned**: [luisml77/gemma-aerial-12b](https://huggingface.co/luisml77/gemma-aerial-12b)
- **o3 Distilled Dataset**: [luisml77/aeriald_o3_500](https://huggingface.co/datasets/luisml77/aeriald_o3_500)

## Important Notes for Future Claude Instances

### 0. **Conda Environment Management**

- **ALWAYS activate the `aerial` conda environment before running any commands**:
  - `conda activate aerial`
- Legacy environments (`aerial-seg-datagen`, `aerial-seg`, `gemma3`) remain available for reproduction but the unified `aerial` env supersedes them for day-to-day work.
- **IMPORTANT**: Never run Python scripts without first activating the appropriate environment

### 0.1. **Script Execution Policy**

- **IMPORTANT**: The user runs all scripts themselves unless explicitly told otherwise
- Do NOT execute scripts with Bash tool - only create/modify them
- Always inform user when scripts are ready to be run

### 0.2. **TODO.md Development Strategy**

- **Task Management**: All active development tasks are tracked in the root-level `TODO.md` file within the "Tasks That Need To Be Completed" section.
- **Archival Process**: When a task is fully addressed and ready for review, move its bullet entry to `archive TODO.md` under the appropriate heading. Do not keep a "Completed Tasks" section inside `TODO.md`.
- **Workflow Process**:
  1. User records new tasks in the "Tasks That Need To Be Completed" section.
  2. Claude and the user collaborate to resolve the tasks.
  3. After finishing a task, relocate the exact bullet (with any relevant notes) to `archive TODO.md` to preserve history.
- **Task Format**: Tasks in `TODO.md` should remain clear, actionable items that can be checked off once moved to the archive.
- **Completion Criteria**: A task is archived when:
  - The implementation is complete.
  - Changes are ready for user inspection.
  - The requirement has been fully addressed.

### 1. **File Paths**

- Use relative paths from the repository root (`~/aerial-d/`) when possible
- The working directory varies depending on the component being used

### 2. **Model Checkpoints**

- Models are stored in `rsrefseg/models/`
- Use `--custom_name` flag to control checkpoint naming
- Published checkpoints: `rsrefseg_aerial-d.pt`, `rsrefseg_combined.pt`

### 3. **Dataset Locations**

- Generated datasets are in `datagen/dataset/`
- Raw datasets (iSAID, LoveDA) should be downloaded using provided scripts
- Public release available on Hugging Face: `luisml77/aerial-d`

### 4. **GPU Usage**

- Default GPU ID is 0, can be changed with `--gpu_id` parameter
- CUDA compilation is disabled in LLM scripts for compatibility

### 5. **Port Management**

- Flask apps use different ports (5001, 5002, etc.) to avoid conflicts
- Check for running processes before starting new web apps

### 6. **Environment Setup**

- Each component (datagen, rsrefseg, llm) has its own requirements.txt
- Consider using virtual environments for different components

This documentation provides a comprehensive overview for working effectively with the Aerial-D codebase. The project combines computer vision, natural language processing, and web development in a cohesive research framework for generalized referring expression segmentation in aerial imagery.
