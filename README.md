# Case Study Vision Language Action Model

End-to-end pipeline for grounded instruction following in construction site operations.

## Project Structure
```
root/
├── frames/                         # Original extracted frames (all videos)
├── pile_detection_verification/
│   ├──simple_nouns/
│   │   ├── frames_with_piles/      # Frames containing detected piles
│   │   └──frames_without_piles/    # Frames with no detections
│   ├──descriptive_phrases/
│       ├── frames_with_piles/      # Frames containing detected piles
│       └──frames_without_piles/    # Frames with no detections   
├── dataset/
│   ├── vqa_pairs.jsonl/            # GPT-4o detection JSON outputs
│   ├── train/                      # Training split 
│   ├── val/                        # Validation split
├── llama-vlam-lora/                # Training checkpoints
│   ├── checkpoint-XXX/             # Training checkpoints
├── llama-vlam-final/               # Final fine-tuned model
├── outputs/                        # Generated demo visualizations
└── logs/                           # Training logs (if using wandb)
```

## Key Files

### Setup & Environment
- **`vlam_environment.yml`** - Conda environment specification (all dependencies)

### Data Pipeline
- **`eval_yolo.py`** - Run YOLO-World detection on frames in /frames -> generates detection JSON files in /pile_detection_verfication. I use the /simple_nouns/with_piles frames to generate VQA training data
- **`generate_VQA.py`** - Use GPT-4o to generate VQA pairs from detections and placed in /dataset
- **`prepare_training_data.py`** - Convert VQA JSONL -> HuggingFace Dataset format for training /dataset/train and /dataset/val

### Training
- **`finetune.py`** - Fine-tune LLaMA 3.2 1B with LoRA on VQA dataset
  - Requires: `C:/vlam/dataset/train` and `C:/vlam/dataset/val` to exist
  - Outputs: Checkpoints to `C:/vlam/llama-vlam-lora/`
  - Final model saved to: `C:/vlam/llama-vlam-final/`

### Inference & Demo
- **`VLA.py`** ⭐ **[MAIN FILE]** - Complete VLAM pipeline (detection → LLM → visualization)
  - Loads YOLO + fine-tuned LLaMA
  - Processes test images with user commands
  - Saves annotated images + JSON outputs to `outputs/`

### Utilities
- **`test_gpu_simple.py`** - Verify GPU, PyTorch, Unsloth installation
- **`visualize_sample.py`** - Draw bounding boxes on sample detection

---

## Quick Start

### 1. Setup Environment
```bash
# Create environment from yml
conda env create -f vlam_environment.yml
conda activate vlam

# OR install from requirements.txt
conda create -n vlam python=3.11 -y
conda activate vlam
```

### 2. Verify GPU Setup
```bash
python test_unsloth.py
```

### 3. Run Full Pipeline (Already Trained)

If you already have the fine-tuned model at `C:/vlam/llama-vlam-final/`:
```bash
python VLA.py
```

This will:
1. Load YOLO-World detector
2. Load fine-tuned LLaMA 3.2 1B
3. Process 3 random test images with 3 commands each
4. Save 9 annotated images + JSON to `outputs/`


---

## Configuration

### Update Paths in VLA.py
```python
# Line ~186
vlam = VLAMPipeline(
    llm_path="C:/vlam/models/llama-vlam-final",  # Your model path
    yolo_model="yolov8x-worldv2.pt"              # Auto-downloads first run
)

# Line ~191
test_dir = Path("C:/vlam/frames_with_piles")  # Your test images folder
```

### Customize Commands

Edit line ~194 in `VLA.py`:
```python
commands = [
    "Where should I go next?",
    "Go to the nearest pile and fill the bucket!",
    "What do you see ahead?"
]
```

---

## Output Format

### Demo Outputs (from VLA.py)

**Annotated Images**: `outputs/*.jpg`
- Original frame with bounding boxes drawn
- Object labels with confidence scores
- VLAM response text overlay

**JSON Results**: `outputs/*.json`
```json
{
  "image": "frame_0001.jpg",
  "command": "Where should I go next?",
  "detections": [
    {
      "class": "mound",
      "confidence": 0.87,
      "position": {
        "cx": 0.52,
        "cy": 0.38,
        "area": 0.045,
        "direction": "far-center",
        "bbox": [130, 158, 254, 215]
      }
    }
  ],
  "grounding_context": "[SCENE] Construction site...",
  "response": "A large mound is detected at far-center..."
}
```
