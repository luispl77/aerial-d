## AerialSeg: Open‑Vocabulary Aerial Image Segmentation with Referring Expressions

![AerialSeg dataset example](docs/dataset.png)

### Dataset and paper
- **Paper**: Coming soon
- **Download**: [AERIAL‑D on Hugging Face](https://huggingface.co/datasets/luisml77/aerial-d)

AerialSeg is an open‑source framework for segmenting aerial images from natural‑language prompts. It includes an automatic dataset pipeline (Aerial‑D), a SigLIP+SAM model (ClipSAM), and LLM tooling (Gemma3/O3) for expression enhancement and evaluation in remote sensing.

### Highlights
- **Dataset generation (AerialD)**: Create referring expression annotations from iSAID and LoveDA via rule‑based synthesis, uniqueness filtering, and optional LLM enhancement
- **ClipSAM model**: SigLIP text/image encoders + SAM decoder with optional domain adaptation and LoRA fine‑tuning
- **LLM enhancement**: Gemma3/OpenAI pipelines to diversify expressions and fine‑tune language models for aerial understanding
- **Web apps**: Flask viewers for browsing annotations and running interactive segmentation demos

### Quick links
- `datagen/` – dataset pipeline and tools ([README](datagen/README.md))
- `clipsam/` – SigLIP+SAM model, train/test utilities ([README](clipsam/README.md))
- `llm/` – LLM enhancement and LoRA fine‑tuning ([README](llm/README.md))
- `tex/` – LaTeX thesis and article sources

---

## Getting started

### Requirements
- Python 3.10+
- CUDA‑enabled GPU recommended (for training/inference)
- Conda or venv for isolated environments

### Environments and install
Create one environment per component (recommended):

```bash
# Dataset generation
conda create -n aerial-seg-datagen python=3.10 -y
conda activate aerial-seg-datagen
pip install -r datagen/requirements.txt

# Model (ClipSAM)
conda create -n aerial-seg python=3.10 -y
conda activate aerial-seg
pip install -r clipsam/requirements.txt

# LLM enhancement / Gemma3
conda create -n gemma3 python=3.10 -y
conda activate gemma3
pip install -r llm/requirements.txt
```

---

## Quickstart

### 1) Build the dataset (AerialD)
```bash
cd datagen
./pipeline/run_pipeline.sh                # full pipeline
./pipeline/run_pipeline.sh --num_images 100  # small sample
./pipeline/run_pipeline.sh --clean --zip  # optional cleanup+zip
```
Outputs are written under `datagen/dataset/` (see `datagen/README.md`).

### 2) Train ClipSAM
```bash
cd clipsam
python train.py --epochs 5 --batch_size 4 --lr 1e-4

# Optional: domain adaptation
python train.py --enable_grl --grl_lambda_schedule exponential

# Resume
python train.py --resume
```
Model checkpoints are saved in `clipsam/models/` following
`clip_sam_YYYYMMDD_HHMMSS_epochs{N}_bs{B}_lr{LR}`.

### 3) Evaluate and visualize
```bash
cd clipsam
python test.py --model_name <checkpoint_dir_name>

# Visualizations only
python test.py --vis_only --num_vis 50
```

### 4) Enhance expressions with LLMs
```bash
cd llm
# Gemma3 enhancement
python gemma3_enhance.py --input_dir ../datagen/dataset --output_dir enhanced_output

# OpenAI O3 enhancement
python o3_enhance.py --dataset_dir ../datagen/dataset
```

### 5) Web applications
```bash
# Annotation viewer (LLM‑enhanced)
cd datagen/utils
python app.py --split train --port 5001

# Rule‑based viewer
python rule_viewer.py --split val --port 5002

# ClipSAM demo app
cd ../../clipsam/utils
python clip_sam_app.py
```

---

## Repository structure
```text
/aerialseg/
├── datagen/       # Dataset generation pipeline and utilities
├── clipsam/       # SigLIP+SAM model, training and testing
├── llm/           # LLM enhancement and LoRA fine‑tuning
├── docs/          # Project images and static docs
└── tex/           # Thesis and extended abstract (LaTeX)
```

Key entry points:
- `datagen/pipeline/run_pipeline.sh` – end‑to‑end dataset build
- `clipsam/train.py` and `clipsam/test.py` – model train/eval
- `llm/gemma3_enhance.py` and `llm/o3_enhance.py` – LLM pipelines

---

## Data and models
- **Datasets**: generated under `datagen/dataset/`
- **Models**: stored under `clipsam/models/`
- **Visualizations**: see `clipsam/visualizations/` and `datagen/saved_visualizations/`

---

## Citations
If you use this repository, please cite the dataset and (when available) the thesis/paper.

```bibtex
@dataset{aerial-d-2024,
  title={AERIAL-D: Referring Expression Instance Segmentation in Aerial Imagery},
  author={Luis M. Lopes and contributors},
  year={2024},
  publisher={Hugging Face},
  url={https://huggingface.co/datasets/luisml77/aerial-d}
}
```

---

## Acknowledgments
- iSAID and LoveDA datasets
- Hugging Face Transformers, PyTorch, OpenCV, Flask
- SAM and SigLIP authors
- Google Gemma and OpenAI models used for language enhancement

---

## Contributing
Issues and pull requests are welcome. Please open an issue to discuss substantial changes.

## Contact
For questions, please open an issue on the repository.
