# Running Gemma 3 with Hugging Face Transformers

This document describes how to load and run the Gemma 3 family of models using the [transformers](https://github.com/huggingface/transformers) library. Examples cover the base and instruction-tuned checkpoints as well as multimodal inference using chat templates.

## Installation

Create a fresh Python environment and install the required packages.

```bash
python -m venv gemma-env
source gemma-env/bin/activate
pip install --upgrade pip
pip install transformers accelerate huggingface_hub
```

(Optional) Install `torch` compiled for your GPU if it is not installed automatically by `accelerate`.

## Loading a Text Model

Gemma 3 checkpoints are available on the Hugging Face Hub. The instruction-tuned 1B model can be loaded as follows:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "google/gemma-3-1b-it"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
```

Generate text with the model:

```python
prompt = "Explain the water cycle in one paragraph."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Multimodal Inference

For models that support image inputs, use the `AutoProcessor` and the multimodal model variant:

```python
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq

model_id = "google/gemma-3-9b-it-mm"  # example multimodal checkpoint
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id, device_map="auto")

image = Image.open("example.jpg")
prompt = "Describe this image"
inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=128)
print(processor.decode(outputs[0], skip_special_tokens=True))
```

## Using the Chat Template

Gemma models include a chat template to handle system, user and assistant turns. Retrieve it via the tokenizer:

```python
chat = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "How do I plant tomatoes?"}
]
encoded = tokenizer.apply_chat_template(chat, return_tensors="pt").to(model.device)
outputs = model.generate(encoded, max_new_tokens=128)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

For additional checkpoints such as `2b`, `7b`, `12b` and `27b`, replace `model_id` with the desired variant, e.g. `google/gemma-3-7b-it`.

