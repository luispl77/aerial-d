## LLM Enhancement and Fine‑Tuning (Gemma 3)

Enhance Aerial‑D expressions and fine‑tune a Gemma 3 model via LoRA for aerial domain language understanding.

### Setup
- Install dependencies from repo root: `pip install -r ../requirements.txt`
- Optional: `huggingface-cli login` for model access/push

### Enhance expressions
Generate enhanced/unique expressions from Aerial‑D annotations.
```bash
python gemma3_enhance.py --input_dir ../datagen/dataset --output_dir enhanced_output
python o3_enhance.py --dataset_dir ../datagen/dataset
```

Outputs are written under `./enhanced_*` directories with per‑object/group folders and JSON files.

### Fine‑tune Gemma 3 with LoRA
```bash
python gemma3_lora_finetune.py \
  --enhanced_data_dir enhanced_annotations_o3_dual \
  --model_name gemma-aerial-12b \
  --output_dir ./gemma-aerial-12b \
  --lora_r 64 --lora_alpha 16
```

Result: a domain‑adapted Gemma 3 checkpoint usable to diversify expressions or generate natural language prompts for training/evaluation.

### Notes
- CUDA not required for enhancement, recommended for fine‑tuning
- Keep outputs versioned to trace which expressions/models were used
