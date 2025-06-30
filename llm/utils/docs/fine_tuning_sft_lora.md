# Fine-tuning Gemma 3

This page describes two common approaches for supervised fine-tuning (SFT) with Gemma 3 models using the `transformers` library:

1. **Full-parameter fine-tuning** – update all model weights.
2. **LoRA** – a parameter-efficient technique that injects low-rank adapters.

Both methods rely on the `Trainer` API from `transformers` and the `peft` library for LoRA.

## Data Format

Training data should be a JSONL or CSV file where each row contains an instruction and the expected response:

```json
{"instruction": "Translate 'Hello' to French", "response": "Bonjour"}
```

The dataset can be loaded with `datasets.load_dataset` and tokenized with the model's tokenizer.

## Full-Parameter Fine-tuning

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments

dataset = load_dataset("path/to/data", split="train")
model_id = "google/gemma-3-1b-it"

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

def tokenize(batch):
    return tokenizer(batch["instruction"], text_target=batch["response"], truncation=True)

tokenized = dataset.map(tokenize, batched=True)

args = TrainingArguments(
    output_dir="ft-gemma3",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=10,
    fp16=True,
)

trainer = Trainer(model=model, args=args, train_dataset=tokenized)
trainer.train()
```

## LoRA Fine-tuning

Install the [peft](https://github.com/huggingface/peft) package and wrap the model with LoRA layers:

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

Training proceeds as in the full fine-tuning example, but only the LoRA parameters are updated, making the process faster and requiring less memory.

