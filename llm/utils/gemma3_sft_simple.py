#!/usr/bin/env python3
"""
Simple Fine-tune Gemma 3 for vision tasks using SFTTrainer (no QLoRA)
Much simpler approach based on TRL SFTTrainer documentation
"""

import torch
import argparse
from datasets import load_dataset
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from trl import SFTConfig, SFTTrainer
import requests
from huggingface_hub import login

def parse_args():
    parser = argparse.ArgumentParser(description='Simple fine-tune Gemma 3 for vision tasks')
    parser.add_argument('--model_id', type=str, default="google/gemma-3-4b-it",
                       help='Hugging Face model ID')
    parser.add_argument('--output_dir', type=str, default="gemma-product-simple",
                       help='Output directory for fine-tuned model')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Per device batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--learning_rate', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--push_to_hub', action='store_true', help='Push model to HF Hub')
    parser.add_argument('--hf_token', type=str, help='Hugging Face token')
    parser.add_argument('--test_only', action='store_true', help='Only run inference test')
    return parser.parse_args()

def setup_device_and_login(args):
    """Setup device and login to Hugging Face"""
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
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
    
    # Convert dataset to messages format that SFTTrainer understands
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
    
    # Convert dataset to messages format
    formatted_dataset = [format_data(sample) for sample in dataset]
    
    print(f"Dataset prepared with {len(formatted_dataset)} samples")
    print("Sample data structure:")
    print(formatted_dataset[0]["messages"])
    
    return formatted_dataset, system_message, user_prompt

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

def setup_model_and_processor(args, device):
    """Setup model and processor (no quantization for simple approach)"""
    print(f"Loading model: {args.model_id}")
    
    # Load model without quantization for simpler approach
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
        attn_implementation="eager"
    )
    processor = AutoProcessor.from_pretrained(args.model_id)
    
    print("Model and processor loaded successfully")
    return model, processor

def train_model(model, processor, dataset, args):
    """Train the model using SFTTrainer"""
    print("Starting training...")
    
    # Create training configuration
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        bf16=True,
        logging_steps=5,
        save_strategy="no",  # Don't save during training to avoid tied weights issue
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        lr_scheduler_type="constant",
        push_to_hub=args.push_to_hub,
        report_to="tensorboard",
        remove_unused_columns=False,
        dataset_text_field="",  # need a dummy field for collator
        dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
    )
    
    # Create collate function
    collate_fn = create_collate_fn(processor)
    
    # Create trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=processor,
        data_collator=collate_fn,
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model()
    
    print("Training completed!")
    return trainer

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
    
    # Load trained model
    model = AutoModelForImageTextToText.from_pretrained(
        args.output_dir,
        device_map=f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu",
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
    
    # Train model
    trainer = train_model(model, processor, dataset, args)
    
    # Free memory
    del model
    del trainer
    torch.cuda.empty_cache()
    
    # Test inference
    test_inference(args, system_message, user_prompt)
    
    print("Fine-tuning completed successfully!")

if __name__ == "__main__":
    main() 