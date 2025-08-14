# AerialSeg: Open-Vocabulary Aerial Image Segmentation

## Project Overview

AerialSeg is a research project focused on **open-vocabulary segmentation of aerial photographs**. The project implements RSRefSeg (Referring Segmentation with Rule-based annotation System) and includes both dataset generation pipelines and model training/evaluation code. The main goal is to segment aerial images using natural language descriptions.

### Key Components:
- **Dataset Generation Pipeline**: Rule-based and LLM-enhanced annotation generation
- **ClipSAM Model**: SigLIP + SAM architecture for referring segmentation  
- **LLM Enhancement**: Gemma3 and GPT models for expression enhancement
- **Web Applications**: Flask-based visualization and annotation tools

## Project Structure

```
/cfs/home/u035679/aerialseg/
├── datagen/                    # Dataset generation pipeline
│   ├── pipeline/              # Rule-based annotation pipeline (Steps 1-8)
│   └── utils/                 # Utilities, debug scripts, web apps
├── clipsam/                   # SigLIP+SAM model implementation
│   ├── model.py              # Main model architecture
│   ├── train.py              # Training script
│   ├── test.py               # Evaluation script
│   └── utils/                # Style transfer, web apps
├── llm/                       # LLM enhancement pipeline
│   ├── gemma3_enhance.py     # Gemma3-based enhancement
│   ├── o3_enhance.py         # OpenAI O3 enhancement
│   └── utils/                # LLM utility scripts
├── thesis_tex/               # LaTeX thesis document
└── .git/                     # Git repository
```

## Key Directories and Their Roles

### 1. **datagen/pipeline/** - Dataset Generation Pipeline
Sequential pipeline for creating the AerialD dataset:

1. **1_isaid_patches.py** - Extract 480x480 patches from iSAID dataset
2. **2_loveda_patches.py** - Extract patches from LoveDA dataset  
3. **3_deepglobe_patches.py** - Extract patches from DeepGlobe dataset
4. **4_add_rules.py** - Add spatial/relational rules (3x3 grid, size, color)
5. **5_generate_all_expressions.py** - Generate referring expressions
6. **6_filter_unique.py** - Remove ambiguous/duplicate expressions
7. **7_historic_filter.py** - Apply historic imagery simulation
8. **8_vllm_enhance.py** - Optional VLLM enhancement

**Entry Point**: `./datagen/pipeline/run_pipeline.sh`

### 2. **datagen/utils/** - Utilities and Web Apps
- `app.py` - Main Flask app for viewing LLM-enhanced annotations
- `rule_viewer.py` - Viewer for rule-based annotations
- `manual_classifier.py` - Manual annotation classification tool
- `gemini_labeler.py` - Gemini-based labeling interface
- `batch_metrics_calculator.py` - Dataset statistics calculator
- Debug scripts: `1_debug_*.py` through `4_debug_*.py`

### 3. **clipsam/** - Model Implementation
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
- `/cfs/home/u035679/aerialseg/datagen/requirements.txt` - Dataset generation dependencies
- `/cfs/home/u035679/aerialseg/clipsam/requirements.txt` - Model training dependencies  
- `/cfs/home/u035679/aerialseg/llm/requirements.txt` - LLM enhancement dependencies

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
cd /cfs/home/u035679/aerialseg/datagen
./pipeline/run_pipeline.sh

# Partial dataset (N images per split)
./pipeline/run_pipeline.sh --num_images 100

# With cleaning and zipping
./pipeline/run_pipeline.sh --clean --zip
```

### Model Training
```bash
cd /cfs/home/u035679/aerialseg/clipsam

# Basic training
python train.py --epochs 5 --batch_size 4 --lr 1e-4

# Resume training
python train.py --resume

# With domain adaptation
python train.py --enable_grl --grl_lambda_schedule exponential
```

### Model Testing/Evaluation
```bash
cd /cfs/home/u035679/aerialseg/clipsam

# Test model
python test.py --model_name clip_sam_20250731_105510_epochs1_bs4x2_lr0.0001

# Visualization only
python test.py --vis_only --num_vis 50
```

### Web Applications
```bash
# Main annotation viewer
cd /cfs/home/u035679/aerialseg/datagen/utils  
python app.py --split train --port 5001

# Rule-based viewer
python rule_viewer.py --split val --port 5002

# ClipSAM inference app
cd /cfs/home/u035679/aerialseg/clipsam/utils
python clip_sam_app.py
```

### LLM Enhancement
```bash
cd /cfs/home/u035679/aerialseg/llm

# Gemma3 enhancement
python gemma3_enhance.py --input_dir ../datagen/dataset --output_dir enhanced_output

# OpenAI O3 enhancement  
python o3_enhance.py --dataset_dir ../datagen/dataset
```

## Architecture Patterns

### 1. **Model Architecture (ClipSAM)**
- **SigLIP Encoder**: Text and image feature extraction
- **SAM Decoder**: Segmentation mask generation
- **Domain Adaptation**: Gradient Reversal Layer for multi-domain training
- **LoRA Fine-tuning**: Parameter-efficient adaptation

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
- `/cfs/home/u035679/aerialseg/datagen/utils/docs.md` - Detailed pipeline documentation
- `/cfs/home/u035679/aerialseg/clipsam/README.md` - RSRefSeg implementation overview
- `/cfs/home/u035679/aerialseg/llm/README.md` - Gemma3 setup and usage
- `/cfs/home/u035679/aerialseg/clipsam/utils/README_style_transfer.md` - Style transfer methods

### Development Notes
- `/cfs/home/u035679/aerialseg/datagen/utils/notes.md` - TODOs and development progress
- Various README files in model directories

### Thesis Document
- `/cfs/home/u035679/aerialseg/thesis_tex/main.tex` - Academic thesis in LaTeX

## Git Repository
- **Remote**: https://github.com/luisml77/aerialseg.git
- **Main Branch**: `main`
- **Current Status**: Active development with regular commits

## Important Notes for Future Claude Instances

### 0. **Conda Environment Management**
- **ALWAYS activate the correct conda environment before running any commands**:
  - `conda activate aerial-seg-datagen` for datagen/ folder code
  - `conda activate aerial-seg` for clipsam/ folder code  
  - `conda activate gemma3` for llm/ folder code
- **IMPORTANT**: Never run Python scripts without first activating the appropriate environment

### 0.1. **Script Execution Policy**
- **IMPORTANT**: The user runs all scripts themselves unless explicitly told otherwise
- Do NOT execute scripts with Bash tool - only create/modify them
- Always inform user when scripts are ready to be run

### 1. **File Paths**
- Always use absolute paths starting with `/cfs/home/u035679/aerialseg/`
- The working directory varies depending on the component being used

### 2. **Model Checkpoints**
- Models are stored in `/cfs/home/u035679/aerialseg/clipsam/models/`
- Follow naming convention: `clip_sam_YYYYMMDD_HHMMSS_epochs{N}_bs{batch_size}_lr{lr}`

### 3. **Dataset Locations**
- Generated datasets are in `/cfs/home/u035679/aerialseg/datagen/dataset/`
- Raw datasets (iSAID, LoveDA, DeepGlobe) should be downloaded using provided scripts

### 4. **GPU Usage**
- Default GPU ID is 0, can be changed with `--gpu_id` parameter
- CUDA compilation is disabled in LLM scripts for compatibility

### 5. **Port Management**
- Flask apps use different ports (5001, 5002, etc.) to avoid conflicts
- Check for running processes before starting new web apps

### 6. **Environment Setup**
- Each component (datagen, clipsam, llm) has its own requirements.txt
- Consider using virtual environments for different components

This documentation provides a comprehensive overview for working effectively with the AerialSeg codebase. The project combines computer vision, natural language processing, and web development in a cohesive research framework for aerial image segmentation.