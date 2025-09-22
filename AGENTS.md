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
├── tex/                       # LaTeX documents
│   ├── dissertation/          # Main thesis document
│   └── article/              # Article/extended abstract document
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

### LaTeX Documents

- `/Users/luispl/Documents/aerialseg/tex/dissertation/main.tex` - Main academic thesis in LaTeX (using IST thesis template)
- `/Users/luispl/Documents/aerialseg/tex/article/ExtendedAbstract.tex` - Extended abstract article for thesis publication

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

### 0.2. **TODO.md Development Strategy**

- **Task Management**: All development tasks are tracked in the root-level `TODO.md` file
- **File Structure**: The `TODO.md` file is organized into two distinct sections:
  1. **"Tasks That Need To Be Completed"** - Top section containing all pending work
  2. **"Completed Tasks"** - Bottom section containing finished work
- **Workflow Process**:
  1. User writes tasks/issues in the "Tasks That Need To Be Completed" section
  2. Claude and user collaboratively solve the tasks
  3. After implementing changes and preparing them for user inspection, **move the completed task** from the "Tasks That Need To Be Completed" section to the "Completed Tasks" section
  4. Keep `TODO.md` as the single source of truth for both pending and completed work
- **Task Format**: Tasks in `TODO.md` should be clear, actionable items that can be tracked and completed
- **Completion Criteria**: A task is moved to "Completed Tasks" when:
  - The implementation is complete
  - Changes are ready for user review/inspection
  - The specific issue or requirement has been fully addressed
- **Status Tracking**:
  - Pending tasks: Listed in "Tasks That Need To Be Completed" section
  - Completed tasks: Moved to "Completed Tasks" section
  - This provides a complete history of work while clearly separating current from finished tasks

### 1. **File Paths**

- Always use absolute paths starting with `/cfs/home/u035679/aerialseg/`
- The working directory varies depending on the component being used

### 2. **Model Checkpoints**

- Models are stored in `/cfs/home/u035679/aerialseg/clipsam/models/`
- Follow naming convention: `clip_sam_YYYYMMDD_HHMMSS_epochs{N}_bs{batch_size}_lr{lr}`

### 3. **Dataset Locations**

- Generated datasets are in `/cfs/home/u035679/aerialseg/datagen/dataset/`
- Raw datasets (iSAID, LoveDA) should be downloaded using provided scripts

### 4. **GPU Usage**

- Default GPU ID is 0, can be changed with `--gpu_id` parameter
- CUDA compilation is disabled in LLM scripts for compatibility

### 5. **Port Management**

- Flask apps use different ports (5001, 5002, etc.) to avoid conflicts
- Check for running processes before starting new web apps

### 6. **Environment Setup**

- Each component (datagen, clipsam, llm) has its own requirements.txt
- Consider using virtual environments for different components

### 8. **LaTeX Document Structure**

```
tex/
├── dissertation/               # Main thesis document
│   ├── main.tex               # Main LaTeX document (entry point)
│   ├── main.pdf               # Generated PDF output
│   ├── istulthesis.cls        # IST thesis class file
│   ├── Thesis-MSc-*.tex       # IST template components
│   ├── Images/                # All figures and images
│   │   ├── RSRefSeg.png       # Architecture diagrams
│   │   ├── dataset.png        # Dataset visualizations
│   │   ├── clipsam.png        # Model architecture
│   │   └── *.png, *.pdf, *.jpg # Other figures
│   ├── Chapters/              # Individual chapter files
│   │   ├── Thesis-MSc-Chapter_1.tex # Introduction
│   │   ├── Thesis-MSc-Chapter_2.tex # Fundamental concepts
│   │   ├── Thesis-MSc-Chapter_3.tex # Related work
│   │   ├── Thesis-MSc-Chapter_4.tex # Dataset construction approach
│   │   ├── Thesis-MSc-Chapter_5.tex # Experiments
│   │   ├── Thesis-MSc-Chapter_6.tex # Conclusion and future work
│   │   ├── Thesis-MSc-Abstract-*.tex # Abstracts (EN/PT)
│   │   ├── Thesis-MSc-Acknowledgments.tex # Acknowledgments
│   │   └── Thesis-MSc-Appendix*.tex # Appendices
│   ├── tables_and_code/       # Code listings and tables
│   └── Thesis-MSc-Bibliography.bib # References
└── article/                    # Extended abstract article
    ├── ExtendedAbstract.tex    # Main article document (entry point)
    ├── ExtendedAbstract.pdf    # Generated PDF output
    ├── ExtendedAbstract_*.tex  # Article section files
    ├── images/                 # Article figures
    └── ExtendedAbstract_ref_db.bib # References
```

### 9. **Thesis Writing Guidelines**

- **Source Code Reference**: When writing about technical implementation details, ALWAYS check the corresponding source code in the repository to confirm accuracy. The thesis and source code are co-located for this purpose.
- **Writing Style**: Use descriptive prose instead of bullet points. Provide detailed, flowing descriptions that explain concepts thoroughly in paragraph form.
- **Technical Accuracy**: Verify pipeline steps, model architectures, and implementation details by examining the actual code files before writing about them.
- **IST Template**: The thesis uses the official IST (Instituto Superior Técnico) LaTeX template with proper document class and formatting requirements.
- **Compilation**: 
  - Dissertation: Use `pdflatex main.tex` in the `/Users/luispl/Documents/aerialseg/tex/dissertation/` directory
  - Article: Use `pdflatex ExtendedAbstract.tex` in the `/Users/luispl/Documents/aerialseg/tex/article/` directory

### 10. **Critical Writing Pattern - ALWAYS Follow This Structure**

**NEVER start a paragraph with technical implementation details. ALWAYS follow this pattern:**

1. **Motivation First**: Explain WHY this step is needed
   - "In order to provide additional spatial context..."
   - "To enable relationships between different instances..."
   - "To better localize objects within the scene..."

2. **Problem Statement**: Explain what challenge/limitation this addresses
   - "However, conventional grid positioning may be insufficient..."
   - "Objects positioned near boundaries present a challenge..."

3. **Solution Introduction**: Introduce what you're going to do
   - "...we introduce extreme position detection"
   - "...the system implements spatial relationship calculation"

4. **Technical Details**: Only AFTER motivation, problem, and solution intro
   - Specific algorithms, parameters, thresholds, etc.

### 11. **Natural Story Writing - CRITICAL GUIDELINES**

**NEVER be robotic or use unexplained technical terms. ALWAYS write as a natural story:**

1. **Tell the transformation story**: Explain how we take existing datasets (instance segmentation, semantic segmentation) and transform their objects into natural language targets
2. **Avoid technical jargon**: NEVER use terms like "spatial context", "uniqueness filtering", "dynamic thresholds" without first explaining what you mean in simple terms
3. **Explain the core challenge**: Focus on the interesting problem - how to describe objects using only what we know from annotations (bounding boxes, masks, categories)
4. **Natural flow**: Write like you're explaining to someone what you're actually doing, not listing technical specifications
5. **Explain before naming**: When you need to introduce a process, first explain what it does, then you can give it a name. Don't throw around unexplained terms.

**Example of WRONG approach**: "To provide spatial context for object localization, we implement uniqueness filtering"

**Example of CORRECT approach**: "The core challenge is figuring out how to describe these objects using only what we know from their bounding boxes and masks. When multiple objects end up with identical characteristics and generate the exact same expressions, we solve this by taking all expressions and matching them against each other - when we find duplicates, we cancel both out as ambiguous."

### 12. **CRITICAL: Polish Rough Language Into Academic Writing**

**When the user provides rough, informal explanations, DO NOT copy verbatim. Your job is to:**

1. **Fix typos and grammar**: Clean up all spelling mistakes, punctuation, and grammatical errors
2. **Convert informal to formal**: Transform casual language into proper academic prose
3. **Maintain the user's ideas**: Keep all the technical concepts and flow exactly as intended
4. **Polish the presentation**: Make it publication-ready while preserving the user's voice and structure

**Example of WRONG approach**: Copying "we let the model see and pay attention to the surrounding features" directly

**Example of CORRECT approach**: Transform into "we enable the model to analyze and incorporate surrounding contextual features"

**NEVER just copy rough language - always polish it into proper academic writing while maintaining the exact technical meaning and flow.**

**Example of WRONG approach**: "Spatial relationship calculation employs an angle-based directional system..."

**Example of CORRECT approach**: "In order to enable complex referring expressions that describe objects relative to other instances, the system implements spatial relationship calculation. This addresses cases where grid positioning alone cannot uniquely identify objects in crowded scenes. The system employs an angle-based directional system..."

**Key reminders:**
- Spatial directions are: above, below, to the left, to the right (NOT north, northeast, etc.)
- Always explain thresholds, distances, and filtering criteria in detail
- Never mention function names - keep it professional

This documentation provides a comprehensive overview for working effectively with the AerialSeg codebase. The project combines computer vision, natural language processing, and web development in a cohesive research framework for aerial image segmentation.
