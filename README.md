# Generalized Referring Expression Segmentation on Aerial Photos

[![Project Page](https://img.shields.io/badge/Project%20Page-visit-blue)](https://luispl77.github.io/aerial-d)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-orange)](https://huggingface.co/datasets/luisml77/aerial-d)
[![Paper](https://img.shields.io/badge/Paper-Preprint-lightgrey)](https://luispl77.github.io/aerial-d)

![Aerial-D dataset example](docs/dataset.png)

## Overview
AerialSeg delivers end-to-end tooling for the article *Generalized Referring Expression Segmentation on Aerial Photos*. The project introduces:
- **Aerial-D**, a 37,288-image dataset with 1.52M referring expressions covering instances, groups, and semantic regions across 21 categories.
- **Automatic data generation**, combining rule-based templates with LLM rewriting to produce grounded language at scale while filtering ambiguous references.
- **Unified RSRefSeg training**, pairing SigLIP2 and SAM with LoRA adapters to learn from Aerial-D alongside RefSegRS, RRSIS-D, NWPU-Refer, and Urban1960SatSeg.
- **Historic robustness**, using stochastic grayscale, sepia, and grain filters plus real historic imagery to maintain accuracy on archival photographs.

## Repository Structure
- `datagen/`: dataset extraction, rule-driven expression generation, historic filtering, and enhancement utilities.
- `rsrefseg/`: SigLIP+SAM training/testing, visualizations, and style-transfer experiments.
- `llm/`: Gemma3 enhancement pipeline, QLoRA fine-tuning, and O3 reference scripts.
- `docs/`, `tex/`: documentation figures and manuscript sources.

## Getting Started

### Environment Setup
The project relies on dedicated conda environments per component:
- `conda activate aerial-seg-datagen` for `datagen/`
- `conda activate aerial-seg` for `rsrefseg/`
- `conda activate gemma3` for `llm/`

Install Python dependencies with the environment-specific `requirements.txt` files or use the root list for a monolithic setup.

### Dataset Generation
```bash
cd /cfs/home/u035679/aerialseg/datagen
./pipeline/run_pipeline.sh            # full automated pipeline
./pipeline/run_pipeline.sh --clean    # regenerate intermediates
./pipeline/run_pipeline.sh --num_images 100  # subset for debugging
```
Outputs are written to `datagen/dataset/` with optional historic-filter augmentations available at training time. Key steps include iSAID/LoveDA patch extraction, rule attribution (3×3 grid, relations, extremes, size cues), expression generation, uniqueness filtering, and optional LLM enhancement. The `datagen/utils/` folder houses web viewers (`app.py`, `rule_viewer.py`), metrics scripts, Hugging Face dataset utilities, and historic-effect debugging helpers.

### Model Training and Evaluation
```bash
cd /cfs/home/u035679/aerialseg/rsrefseg
python train.py --epochs 5 --batch_size 4 --lr 1e-4
python train.py --enable_grl --grl_lambda_schedule exponential
python test.py --model_name <checkpoint_name>
```
The RSRefSeg checkpoints fine-tune SigLIP2-SO400M and SAM-ViT (Base or Large) using LoRA ranks 16/32 while mixing Aerial-D with RefSegRS, RRSIS-D, NWPU-Refer, and Urban1960SatBench. Scripts cover resuming, dataset-specific evaluation, visualization-only passes, and style-transfer experiments (`utils/test_style_transfer.py`). Flask inference apps live under `rsrefseg/utils/`.

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
Gemma3-12B is distilled from 500 high-quality OpenAI o3 samples using QLoRA (~238× cheaper than direct o3 usage). The `llm/` directory also contains inference helpers, dataset cards, and scripts for managing Hugging Face artifacts.

## Historic Image Filters
Training-time augmentations approximate monochrome, grainy, and sepia degradations through luminance conversion, gamma/contrast adjustments, and additive noise. Combined with Urban1960SatSeg, these filters preserve segmentation quality under archival conditions.

## Cite Aerial-D
Please cite the dataset when using this repository:

```bibtex
@dataset{aerial-d-2024,
  title={Aerial-D: Referring Expression Segmentation for Aerial Imagery},
  author={Luis M. Lopes and contributors},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/luisml77/aerial-d}
}
```

## Contributing
Issues and pull requests are welcome. Please open an issue before submitting substantial changes.

## Contact
For inquiries, email [maarnotto@gmail.com](mailto:maarnotto@gmail.com) or open a GitHub issue.
