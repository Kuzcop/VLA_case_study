import torch
from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported
import wandb  
import random
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# Model config
max_seq_length = 512  # Enough for context + Q&A
model_name = "unsloth/Llama-3.2-1B-Instruct"

# Load model with 4-bit quantization
print("\nLoading LLaMA 3.2 1B...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # QLoRA - fits in 8GB VRAM
)

# Add LoRA adapters
print("Adding LoRA adapters...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # LoRA rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing="unsloth",  # Memory efficient
    random_state=42,
)

# Load datasets
print("Loading datasets...")
train_dataset = load_from_disk("C:/Github_workspace/vlam_sensmore/dataset/train")
val_dataset = load_from_disk("C:/Github_workspace/vlam_sensmore/dataset/val")

print(f"   Train examples: {len(train_dataset)}")
print(f"   Val examples: {len(val_dataset)}")

# LLaMA 3 chat template
chat_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a Vision Language Action Model assisting wheel loader operators. You receive scene descriptions with detected objects and their positions, then provide clear operational guidance.<|eot_id|><|start_header_id|>user<|end_header_id|>

{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{output}<|eot_id|>"""

def formatting_func(examples):
    """Format examples with chat template"""
    texts = []
    for instruction, output in zip(examples['instruction'], examples['output']):
        text = chat_template.format(
            instruction=instruction,
            output=output
        )
        texts.append(text)
    return texts

# Calculate steps (adjust based on your dataset size)
num_train_examples = len(train_dataset)
epochs = 3
batch_size = 2
grad_accum = 4
effective_batch_size = batch_size * grad_accum

max_steps = (num_train_examples * epochs) // effective_batch_size

print(f"\nTraining configuration:")
print(f"   Epochs: {epochs}")
print(f"   Batch size: {batch_size} x {grad_accum} = {effective_batch_size} (effective)")
print(f"   Max steps: {max_steps}")
print(f"   Steps per epoch: ~{num_train_examples // effective_batch_size}")


# Initialize wandb before training
print("\nInitializing Weights & Biases...")
wandb.init(
    project="vlam-wheel-loader",  # Your project name
    name=f"llama-3.2-1b-vlam-run_{random.randint(1, 100)}",  # This run's name
    config={
        "model": "LLaMA-3.2-1B",
        "lora_r": 16,
        "lora_alpha": 16,
        "batch_size": 2,
        "grad_accum": 4,
        "learning_rate": 2e-4,
        "max_steps": max_steps,
        "dataset_size": len(train_dataset),
    }
)

# Training arguments
training_args = TrainingArguments(
    output_dir="C:/Github_workspace/vlam_sensmore/llama-vlam-lora",
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=grad_accum,
    warmup_steps=min(50, max_steps // 10),  # 10% warmup
    max_steps=max_steps,
    learning_rate=2e-4,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    logging_steps=10,
    optim="adamw_8bit",  # Memory efficient optimizer
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    save_strategy="steps",
    save_steps=max(50, max_steps // 10),  # Save 10 checkpoints
    eval_strategy="steps",
    eval_steps=max(50, max_steps // 10),
    save_total_limit=3,
    report_to="wandb",  # Disable wandb/tensorboard
    load_best_model_at_end=False,  # Saves time
)

# Trainer
print("\nInitializing trainer...")
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    dataset_text_field="text",
    formatting_func=formatting_func,
    max_seq_length=max_seq_length,
    args=training_args,
)

# Train!
print("\nStarting training...")

trainer.train()

# Save final model
print("\nSaving model...")
model.save_pretrained("C:/Github_workspace/vlam_sensmore/llama-vlam-final")
tokenizer.save_pretrained("C:/Github_workspace/vlam_sensmore/llama-vlam-final")

print("\nTraining complete!")
print("   Model saved to: C:/Github_workspace/vlam_sensmore/")
wandb.finish()