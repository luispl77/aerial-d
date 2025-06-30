# Gemma 3 with Hugging Face Transformers

This repository shows how to run and fine‑tune Google’s Gemma 3 models using the [transformers](https://github.com/huggingface/transformers) library.

## Setup

Create a Python environment and install dependencies:

```bash
python -m venv gemma-env
source gemma-env/bin/activate
pip install -r requirements.txt
```

Log in to the Hugging Face Hub if required:

```bash
huggingface-cli login
```

## Documentation

* **Running models:** see [docs/gemma3_transformers.md](docs/gemma3_transformers.md) for examples covering text and multimodal checkpoints with chat templates.
* **Enforcing JSON output:** see [docs/enforcing_json_output.md](docs/enforcing_json_output.md).
* **Fine-tuning (SFT / LoRA):** see [docs/fine_tuning_sft_lora.md](docs/fine_tuning_sft_lora.md).


## Setup

Create a Python environment and install dependencies:

```bash
python -m venv gemma-env
source gemma-env/bin/activate
pip install -r requirements.txt

