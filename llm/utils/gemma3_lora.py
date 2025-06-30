#!/usr/bin/env python3
"""
Fine-tune Gemma 3 for vision tasks using QLoRA
Based on the Hugging Face documentation for Gemma vision fine-tuning
"""

import torch
import argparse
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import SFTConfig, SFTTrainer
import requests
import os
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser(description='Fine-tune Gemma 3 for vision tasks')
    parser.add_argument('--model_id', type=str, default="google/gemma-3-4b-it",
                       help='Hugging Face model ID')
    parser.add_argument('--output_dir', type=str, default="gemma-product-description",
                       help='Output directory for fine-tuned model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Per device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--push_to_hub', action='store_true', help='Push model to HF Hub')
    parser.add_argument('--hf_token', type=str, help='Hugging Face token')
    parser.add_argument('--test_only', action='store_true', help='Only run inference test')
    parser.add_argument('--merge_adapter', action='store_true', 
                       help='Merge adapter with base model after training')
    return parser.parse_args()

def setup_device_and_login(args):
    """Setup device and login to Hugging Face"""
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Check GPU capability for bfloat16
    if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8:
        print("Warning: GPU does not support bfloat16 optimally, but proceeding anyway.")
    
    # Login to Hugging Face if token provided
    if args.hf_token:
        login(args.hf_token)
        print("Logged in to Hugging Face Hub")
    
    return device

def create_dataset():
    """Create and prepare the fine-tuning dataset"""
    print("Loading and preparing dataset...")
    
    # System message for the assistant
    system_message = "You are an expert product description writer for Amazon."
    
    # User prompt that combines the user query and the schema
    user_prompt = """Create a Short Product description based on the provided <PRODUCT> and <CATEGORY> and image.
Only return description. The description should be SEO optimized and for a better mobile search experience.

<PRODUCT>
{product}
</PRODUCT>

<CATEGORY>
{category}
</CATEGORY>
"""
    
    # Convert dataset to OAI messages
    def format_data(sample):
        return {
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": system_message}],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt.format(
                                product=sample["Product Name"],
                                category=sample["Category"],
                            ),
                        },
                        {
                            "type": "image",
                            "image": sample["image"],
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": sample["description"]}],
                },
            ],
        }
    
    # Load dataset from the hub
    dataset = load_dataset("philschmid/amazon-product-descriptions-vlm", split="train")
    
    # Convert dataset to OAI messages
    # Use list comprehension to keep Pil.Image type, .map() would convert image to bytes
    dataset = [format_data(sample) for sample in dataset]
    
    print(f"Dataset prepared with {len(dataset)} samples")
    print("Sample data structure:")
    print(dataset[0]["messages"])
    
    return dataset, system_message, user_prompt

def process_vision_info(messages):
    """Process vision information from messages"""
    image_inputs = []
    # Iterate through each conversation
    for msg in messages:
        # Get content (ensure it's a list)
        content = msg.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Check each content element for images
        for element in content:
            if isinstance(element, dict) and (
                "image" in element or element.get("type") == "image"
            ):
                # Get the image and convert to RGB
                if "image" in element:
                    image = element["image"]
                else:
                    image = element
                image_inputs.append(image.convert("RGB"))
    return image_inputs

def setup_model_and_processor(args, device):
    """Setup model and processor with quantization"""
    print(f"Loading model: {args.model_id}")
    
    # Define model init arguments
    model_kwargs = dict(
        attn_implementation="eager",  # Use "flash_attention_2" when running on Ampere or newer GPU
        torch_dtype=torch.bfloat16,  # What torch dtype to use, defaults to auto
        device_map={0: args.gpu} if torch.cuda.is_available() else "cpu",  # Use specified GPU
    )

    # BitsAndBytesConfig int-4 config
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=model_kwargs["torch_dtype"],
        bnb_4bit_quant_storage=model_kwargs["torch_dtype"],
    )

    # Load model and tokenizer
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, **model_kwargs)
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    print("Model and processor loaded successfully")
    return model, processor

def setup_peft_config():
    """Setup PEFT configuration for QLoRA"""
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=[
            "lm_head",
            "embed_tokens",
        ],
    )
    return peft_config

def create_collate_fn(processor):
    """Create collate function for training"""
    def collate_fn(examples):
        texts = []
        images = []
        for example in examples:
            image_inputs = process_vision_info(example["messages"])
            text = processor.apply_chat_template(
                example["messages"], add_generation_prompt=False, tokenize=False
            )
            texts.append(text.strip())
            images.append(image_inputs)

        # Tokenize the texts and process the images
        batch = processor(text=texts, images=images, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens and image tokens in the loss computation
        labels = batch["input_ids"].clone()

        # Mask image tokens
        image_token_id = [
            processor.tokenizer.convert_tokens_to_ids(
                processor.tokenizer.special_tokens_map["boi_token"]
            )
        ]
        # Mask tokens for not being used in the loss computation
        labels[labels == processor.tokenizer.pad_token_id] = -100
        labels[labels == image_token_id] = -100
        labels[labels == 262144] = -100

        batch["labels"] = labels
        return batch
    
    return collate_fn

def setup_training_config(args):
    """Setup training configuration"""
    training_args = SFTConfig(
        output_dir=args.output_dir,                     # directory to save and repository id
        num_train_epochs=args.epochs,                   # number of training epochs
        per_device_train_batch_size=args.batch_size,    # batch size per device during training
        gradient_accumulation_steps=args.gradient_accumulation_steps,  # number of steps before performing a backward/update pass
        gradient_checkpointing=True,                    # use gradient checkpointing to save memory
        optim="adamw_torch_fused",                      # use fused adamw optimizer
        logging_steps=5,                                # log every 5 steps
        save_strategy="epoch",                          # save checkpoint every epoch
        learning_rate=args.learning_rate,               # learning rate, based on QLoRA paper
        bf16=True,                                      # use bfloat16 precision
        max_grad_norm=0.3,                              # max gradient norm based on QLoRA paper
        warmup_ratio=0.03,                              # warmup ratio based on QLoRA paper
        lr_scheduler_type="constant",                   # use constant learning rate scheduler
        push_to_hub=args.push_to_hub,                   # push model to hub
        report_to="tensorboard",                        # report metrics to tensorboard
        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },  # use reentrant checkpointing
        dataset_text_field="",                          # need a dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
    )
    training_args.remove_unused_columns = False  # important for collator
    
    return training_args

def train_model(model, processor, dataset, peft_config, training_args):
    """Train the model using SFTTrainer"""
    print("Starting training...")
    
    # Create collate function
    collate_fn = create_collate_fn(processor)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=processor,
        data_collator=collate_fn,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    print("Training completed!")
    return trainer

def merge_and_save_model(args):
    """Merge adapter with base model and save"""
    print("Merging adapter with base model...")
    
    # Load base model
    model = AutoModelForImageTextToText.from_pretrained(args.model_id, low_cpu_mem_usage=True)
    
    # Merge LoRA and base model and save
    peft_model = PeftModel.from_pretrained(model, args.output_dir)
    merged_model = peft_model.merge_and_unload()
    merged_model.save_pretrained("merged_model", safe_serialization=True, max_shard_size="2GB")
    
    processor = AutoProcessor.from_pretrained(args.output_dir)
    processor.save_pretrained("merged_model")
    
    print("Model merged and saved to 'merged_model' directory")

def generate_description(sample, model, processor, system_message, user_prompt):
    """Generate product description for a sample"""
    # Convert sample into messages and then apply the chat template
    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_message}]},
        {"role": "user", "content": [
            {"type": "image", "image": sample["image"]},
            {"type": "text", "text": user_prompt.format(product=sample["product_name"], category=sample["category"])},
        ]},
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Process the image and text
    image_inputs = process_vision_info(messages)
    # Tokenize the text and process the images
    inputs = processor(
        text=[text],
        images=image_inputs,
        padding=True,
        return_tensors="pt",
    )
    # Move the inputs to the device
    inputs = inputs.to(model.device)

    # Generate the output
    stop_token_ids = [processor.tokenizer.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=256, 
        top_p=1.0, 
        do_sample=True, 
        temperature=0.8, 
        eos_token_id=stop_token_ids, 
        disable_compile=True
    )
    # Trim the generation and decode the output to text
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    return output_text[0]

def test_inference(args, system_message, user_prompt):
    """Test model inference with a sample"""
    print("Testing model inference...")
    
    # Load Model with PEFT adapter
    model = AutoModelForImageTextToText.from_pretrained(
        args.output_dir,
        device_map={0: args.gpu} if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    processor = AutoProcessor.from_pretrained(args.output_dir)
    
    # Test sample with Product Name, Category and Image
    sample = {
        "product_name": "Hasbro Marvel Avengers-Serie Marvel Assemble Titan-Held, Iron Man, 30,5 cm Actionfigur",
        "category": "Toys & Games | Toy Figures & Playsets | Action Figures",
        "image": Image.open(requests.get("https://m.media-amazon.com/images/I/81+7Up7IWyL._AC_SY300_SX300_.jpg", stream=True).raw).convert("RGB")
    }
    
    # Generate the description
    description = generate_description(sample, model, processor, system_message, user_prompt)
    
    print("\n" + "="*50)
    print("INFERENCE TEST RESULTS")
    print("="*50)
    print(f"Product: {sample['product_name']}")
    print(f"Category: {sample['category']}")
    print(f"Generated Description:\n{description}")
    print("="*50)

def main():
    args = parse_args()
    
    # Setup device and login
    device = setup_device_and_login(args)
    
    if args.test_only:
        # Only run inference test
        _, system_message, user_prompt = create_dataset()
        test_inference(args, system_message, user_prompt)
        return
    
    # Create dataset
    dataset, system_message, user_prompt = create_dataset()
    
    # Setup model and processor
    model, processor = setup_model_and_processor(args, device)
    
    # Setup PEFT config
    peft_config = setup_peft_config()
    
    # Setup training config
    training_args = setup_training_config(args)
    
    # Train model
    trainer = train_model(model, processor, dataset, peft_config, training_args)
    
    # Free memory
    del model
    del trainer
    torch.cuda.empty_cache()
    
    # Merge adapter if requested
    if args.merge_adapter:
        merge_and_save_model(args)
    
    # Test inference
    test_inference(args, system_message, user_prompt)
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main()