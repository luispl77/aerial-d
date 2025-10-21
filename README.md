# The Aerial-D Dataset for Generalized Referring Expression Segmentation on Aerial Photos

<div align="center">

### 🔗 Quick Links

**[🌐 Project Page](https://luispl77.github.io/aerial-d)** | **[📊 Dataset (HuggingFace)](https://huggingface.co/datasets/luisml77/aerial-d)** | **[📄 Paper](https://luispl77.github.io/aerial-d)** | **[🤖 Models](https://huggingface.co/collections/luisml77/aerial-d-68a17e2431daebb96218edce)**

[![Project Page](https://img.shields.io/badge/Project%20Page-visit-blue)](https://luispl77.github.io/aerial-d)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange)](https://huggingface.co/datasets/luisml77/aerial-d)
[![Paper](https://img.shields.io/badge/Paper-Preprint-lightgrey)](https://luispl77.github.io/aerial-d)

</div>

![Aerial-D dataset examples](docs/6samples.png)

## Overview
This repository provides end-to-end tooling for *The Aerial-D Dataset for Generalized Referring Expression Segmentation on Aerial Photos* (submitted to IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, J-STARS). The project introduces:
- **Aerial-D**, a 37,288-image dataset with 1.52M referring expressions covering instances, groups, and semantic regions across 21 categories.
- **Automatic data generation**, combining rule-based templates with LLM rewriting to produce grounded language at scale while filtering ambiguous references.
- **Unified RSRefSeg training**, pairing SigLIP2 and SAM with LoRA adapters to learn from Aerial-D alongside RefSegRS, RRSIS-D, NWPU-Refer, and Urban1960SatSeg.

### Hugging Face Collection
All public artifacts live in the [Aerial-D collection on Hugging Face](https://huggingface.co/collections/luisml77/aerial-d-68a17e2431daebb96218edce):
1. [luisml77/gemma-aerial-12b](https://huggingface.co/luisml77/gemma-aerial-12b) — Gemma3 finetuned weights for Step 7
2. [luisml77/aeriald_o3_500](https://huggingface.co/datasets/luisml77/aeriald_o3_500) — distilled 500-sample o3 dataset for Gemma3 distillation
3. [luisml77/aerial-d](https://huggingface.co/datasets/luisml77/aerial-d) — full dataset release
4. [luisml77/rsrefseg](https://huggingface.co/luisml77/rsrefseg) — RSRefSeg checkpoints (`rsrefseg_aerial-d.pt`, `rsrefseg_combined.pt`)

## Repository Structure
- `datagen/`: dataset extraction, rule-driven expression generation, historic filtering, and enhancement utilities.
- `rsrefseg/`: SigLIP+SAM training/testing, visualizations, and style-transfer experiments.
- `llm/`: Gemma3 enhancement pipeline, QLoRA fine-tuning, and OpenAI o3 reference scripts.
- `docs/`: project webpage files.
- `tex/`: LaTeX source for article and dissertation.

## Getting Started

### Environment Setup
**Option 1: Using Conda (recommended, requires Python 3.12)**
```bash
conda create -n aerial python=3.12
conda activate aerial
pip install -r requirements.txt
```

**Option 2: Using venv**
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Dataset Download/Generation
You can reproduce Aerial-D locally or download the public release.

**Download from Hugging Face**
```bash
huggingface-cli download luisml77/aerial-d --repo-type dataset --local-dir datagen/dataset
```

**Optional: rebuild locally**
```bash
cd /cfs/home/u035679/aerialseg/datagen
./pipeline/run_pipeline.sh --clean
# Package the result into aeriald.zip if needed
python pipeline/zip_dataset.py --base_dir dataset --zip_path aeriald.zip
```
The pipeline extracts iSAID/LoveDA patches, assigns rules (3×3 grid, relations, extremes, size cues), generates expressions, filters for uniqueness, and applies optional historic filters. Step 7 (`7_vllm_enhance.py`) expects the Gemma3 checkpoint produced in the **LLM Expression Enhancement** section; either complete those steps first or download [luisml77/gemma-aerial-12b](https://huggingface.co/luisml77/gemma-aerial-12b) and the distilled dataset [luisml77/aeriald_o3_500](https://huggingface.co/datasets/luisml77/aeriald_o3_500) from the collection before enabling Step 7. Utilities for viewing and metrics live under `datagen/utils/`.

### Model Training and Evaluation (Aerial-D)
`model.py` defines the SigLIP2 + SAM architecture (RSRefSeg) with LoRA adapters. Training and testing use the dataset downloaded above.

```bash
# Train (writes checkpoint under rsrefseg/models/ by default)
cd /cfs/home/u035679/aerialseg/rsrefseg
python train.py --dataset_root ../datagen/dataset --custom_name aeriald_run

# Test the produced checkpoint
python test.py --model_name aeriald_run --dataset_type aeriald

# Skip training: download the published Aerial-D checkpoint
huggingface-cli download luisml77/aerial-seg --repo-type model --local-dir models/rsrefseg_aeriald
python test.py --model_name rsrefseg_aeriald --dataset_type aeriald

# Evaluate the combined multi-dataset checkpoint (requires SAM ViT-Large)
huggingface-cli download luisml77/rsrefseg --repo-type model --local-dir models/rsrefseg_combined
python test.py --model_name rsrefseg_combined --dataset_type aeriald --sam_model facebook/sam-vit-large
```
The training script fine-tunes SigLIP2-SO400M and SAM-ViT (Base or Large) on Aerial-D only. The optional `--custom_name` flag controls the run folder name under `rsrefseg/models/`, which you pass to `test.py` for evaluation. Visualization-only and Flask inference utilities remain available under `rsrefseg/utils/`.

### LLM Expression Enhancement
```bash
cd /cfs/home/u035679/aerialseg/llm
python gemma3_enhance.py --input_dir ../datagen/dataset --output_dir enhanced_output
python o3_enhance.py --dataset_dir ../datagen/dataset

# QLoRA fine-tuning
python gemma3_lora_finetune.py \
  --enhanced_data_dir enhanced_annotations_o3_dual \
  --model_name gemma-aerial-12b \
  --output_dir ./gemma-aerial-12b \
  --lora_r 64 --lora_alpha 16
```
Gemma3-12B is distilled from 500 high-quality OpenAI o3 samples using QLoRA (~238× cheaper than direct OpenAI o3 usage). The `llm/` directory also contains inference helpers, dataset cards, and scripts for managing Hugging Face artifacts.

## Historic Image Filters
Training-time augmentations approximate monochrome, grainy, and sepia degradations through luminance conversion, gamma/contrast adjustments, and additive noise. Combined with Urban1960SatSeg, these filters preserve segmentation quality under archival conditions.

## Cite Aerial-D
Please cite the article when using this repository:

```bibtex
@article{lopes2025aeriald,
  title={The Aerial-D Dataset for Generalized Referring Expression Segmentation on Aerial Photos},
  author={Lopes, Luis Pedro Soares Marnoto Gaspar},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (J-STARS)},
  year={2025},
  note={Submitted}
}
```

**Dataset Citation:**
```bibtex
@dataset{aerial-d-2024,
  title={AERIAL-D: Referring Expression Segmentation for Aerial Imagery},
  author={Lopes, Luis Pedro Soares Marnoto Gaspar},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/luisml77/aerial-d}
}
```

## Contributing
Issues and pull requests are welcome. Please open an issue before submitting substantial changes.

## Contact
For inquiries, email [maarnotto@gmail.com](mailto:maarnotto@gmail.com) or open a GitHub issue.
