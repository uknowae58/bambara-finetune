"""
Bambara Fine-tuning Script with Unsloth
Fine-tunes Qwen3.5 on bambara-lm-qa dataset
"""

from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
import os

# Config
MODEL_NAME = "Qwen/Qwen2.5-0.5B"
MAX_seq_LENGTH = 2048
DATASET_NAME = "oza75/bambara-lm-qa"
OUTPUT_DIR = "./bambara-model"

print("=" * 50)
print("Bambara Fine-tuning with Unsloth")
print("=" * 50)

# 1. Load model
print("\n[1/4] Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_NAME,
    max_seq_length=MAX_seq_LENGTH,
    dtype=torch.float16,
    load_in_4bit=True,
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

# 3. Load dataset
print("[3/4] Loading dataset...")
dataset = load_dataset(DATASET_NAME, split="train")

# Format for Qwen3
def format_prompt(example):
    return {
        "text": f"<|im_start|>user\n{example.get('question', '')}<|im_end|>\n<|im_start|>assistant\n{example.get('answer', '')}<|im_end|>"
    }

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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
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
    ),
)

trainer.train()

# Save
print("\nSaving model...")
model.save_pretrained(f"{OUTPUT_DIR}-final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}-final")

print("\n✅ Training complete!")
print(f"Model saved to: {OUTPUT_DIR}-final")
