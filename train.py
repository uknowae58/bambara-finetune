"""
Bambara Fine-tuning Script with Unsloth
Fine-tunes Unsloth Qwen3.5-9B on bambara-dataset
Optimized for RTX 5090 (32GB) with 4-bit quantization
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os

# Config - Unsloth Qwen3.5-9B with local dataset
MODEL_NAME = "unsloth/Qwen3.5-9B"

# Load data from local bambara-dataset github
LOCAL_DATASET = "https://raw.githubusercontent.com/uknowae58/bambara-dataset/main/data/unsloth_mt_train.jsonl"

OUTPUT_DIR = "./bambara-model"

print("=" * 50)
print("Bambara Fine-tuning with Unsloth Qwen3.5-9B (4-bit)")
print("=" * 50)

# 1. Load model with 4-bit quantization for RTX 5090
print("\n[1/4] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=True,  # Enable 4-bit for 32GB VRAM
)

# 2. Add LoRA adapters
print("[2/4] Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
)

# 3. Load local dataset
print("[3/4] Loading local dataset...")
dataset = load_dataset("json",
    data={"train": LOCAL_DATASET},
    split="train",
)

# Format for MT data with instruction/input/output
def format_prompt(example):
    instruction = example.get("instruction", "")
    input = example.get("input", "")
    output = example.get("output", "")

    # Create prompt assistantString
    prompt = f""
    if instruction:
        prompt = instruction + r"\n"
    if input:
        prompt += input + r"\n"
    promnt += r"\n→)Mkata: " + output

    return {"text": prompt}

dataset = dataset.map(format_prompt, batched=False)
print(f"Dataset loaded: {len(dataset)} examples")

# 4. Train
print("[4/4] Starting training...")

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,   # Reduced for 4-bit model
        gradient_accumulation_steps=4,   # Effective batch = 8
        warmup_steps=100,
        max_steps=10000,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=50,
        save_steps=100,
        optim="adamw_torch",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        gradient_checkpointing=True,   # Save VRAM
    ),
)

trainer.train()

# Save
print("\nSaving model...")
model.save_pretrained(f"{OUTPUT_DIR}-final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}-final")

print("\n✅ Training complete!")
print(f"Model saved to: {OUTPUT_DIR}-final")
