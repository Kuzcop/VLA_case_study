import json
from pathlib import Path
from datasets import Dataset
import random

# Load VQA pairs from JSONL
vqa_file = Path("C:/Github_workspace/vlam_sensmore/dataset/vqa_pairs.jsonl")
vqa_pairs = []

print("Loading VQA pairs...")
with open(vqa_file, 'r', encoding='utf-8') as f:
    for line in f:
        vqa_pairs.append(json.loads(line))

print(f"Loaded {len(vqa_pairs)} VQA pairs")

# Format for LLaMA 3.2 instruction tuning
def format_prompt(example):
    """Convert to instruction-output format"""
    
    # The instruction includes the grounding context + question
    instruction = f"""{example['grounding_context']}

[COMMAND] {example['question']}"""
    
    return {
        'instruction': instruction,
        'output': example['answer']
    }

# Format all pairs
print("Formatting for training...")
formatted_data = [format_prompt(pair) for pair in vqa_pairs]

# Split train/val (90/10)
random.seed(42)
random.shuffle(formatted_data)
split_idx = int(len(formatted_data) * 0.9)
train_data = formatted_data[:split_idx]
val_data = formatted_data[split_idx:]

print(f"Train: {len(train_data)}, Val: {len(val_data)}")

# Convert to HuggingFace Dataset
train_dataset = Dataset.from_list(train_data)
val_dataset = Dataset.from_list(val_data)

# Save to disk
output_dir_train = Path("C:/Github_workspace/vlam_sensmore/dataset/train")
output_dir_val = Path("C:/Github_workspace/vlam_sensmore/dataset/val")

train_dataset.save_to_disk(str(output_dir_train))
val_dataset.save_to_disk(str(output_dir_val))

print(f"\nTraining data prepared!")
print(f"   Train saved to: {output_dir_train}")
print(f"   Val saved to: {output_dir_val}")

# Show sample
print("\nSample training example:")
print("="*60)
print(f"INSTRUCTION:\n{train_data[0]['instruction']}\n")
print(f"OUTPUT:\n{train_data[0]['output']}")
print("="*60)

print("\nAnother sample:")
print("="*60)
print(f"INSTRUCTION:\n{train_data[1]['instruction']}\n")
print(f"OUTPUT:\n{train_data[1]['output']}")
print("="*60)