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
- `datagen/`: dataset extraction, rule-driven expression generation, and enhancement utilities.
- `clipsam/`: RSRefSeg implementation, training, evaluation, and visualization tools.
- `llm/`: Gemma3 fine-tuning (QLoRA), OpenAI o3 reference scripts, and evaluation assets.
- `docs/`, `tex/`: documentation figures and manuscript sources.

## Getting Started

### Environment Setup
The project relies on dedicated conda environments per component:
- `conda activate aerial-seg-datagen` for `datagen/`
- `conda activate aerial-seg` for `clipsam/`
- `conda activate gemma3` for `llm/`

Install Python dependencies with the environment-specific `requirements.txt` files or use the root list for a monolithic setup.

### Dataset Generation
```bash
cd /cfs/home/u035679/aerialseg/datagen
./pipeline/run_pipeline.sh            # full automated pipeline
./pipeline/run_pipeline.sh --clean    # regenerate intermediates
./pipeline/run_pipeline.sh --num_images 100  # subset for debugging
```
Outputs are written to `datagen/dataset/` with optional historic-filter augmentations available at training time.

### Model Training and Evaluation
```bash
cd /cfs/home/u035679/aerialseg/clipsam
python train.py --epochs 5 --batch_size 4 --lr 1e-4
python train.py --enable_grl --grl_lambda_schedule exponential
python test.py --model_name <checkpoint_name>
```
The RSRefSeg checkpoints in the article fine-tune SigLIP2-SO400M and SAM-ViT (Base or Large) using LoRA ranks 16 and 32 respectively while mixing Aerial-D with external datasets.

### LLM Expression Enhancement
```bash
cd /cfs/home/u035679/aerialseg/llm
python gemma3_enhance.py --input_dir ../datagen/dataset --output_dir enhanced_output
python o3_enhance.py --dataset_dir ../datagen/dataset
```
Gemma3-12B is distilled from 500 high-quality OpenAI o3 samples using QLoRA, driving the large-scale enhancement pass at roughly 238× lower cost than direct o3 usage.

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
