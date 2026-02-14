import os
import json
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm
import random
import base64
import cv2
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Paths
DETECTIONS_JSON  = Path("C:/Github_workspace/vlam_sensmore/pile_detection_verification/simple_nouns/metadata.json") 
FRAMES_WITH_PILES_DIR  = Path("C:/Github_workspace/vlam_sensmore/pile_detection_verification/simple_nouns/with_piles")
FRAMES_WITHOUT_PILES_DIR = Path("C:/Github_workspace/vlam_sensmore/pile_detection_verification/simple_nouns/with_piles")
OUTPUT_FILE = Path("C:/Github_workspace/vlam_sensmore/dataset/vqa_pairs.jsonl")

# Load your detections
print("Loading detections...")
with open(DETECTIONS_JSON) as f:
    data = json.load(f)

print(f"Loaded detection data:")
print(f"  Total with piles: {data['with_piles']}")

# Sample size (adjust for cost control)
MAX_SAMPLES = 500  

# Convert bbox to position format
def bbox_to_position(bbox, img_w, img_h):
    """Convert [x1, y1, x2, y2] to normalized position"""
    x1, y1, x2, y2 = bbox
    
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    area = ((x2 - x1) * (y2 - y1)) / (img_w * img_h)
    
    # Semantic direction
    h = 'left' if cx < 0.4 else ('right' if cx > 0.6 else 'center')
    v = 'far' if cy < 0.35 else ('near' if cy > 0.65 else 'mid')
    direction = f'{v}-{h}'
    
    return {
        'cx': round(cx, 3),
        'cy': round(cy, 3),
        'area': round(area, 4),
        'direction': direction
    }

def format_grounding_context(detections, img_w, img_h):
    """Format detections as LLM input text"""
    lines = ["[SCENE] Construction site view from wheel loader.", "[DETECTIONS]"]
    for i, det in enumerate(detections, 1):
        pos = bbox_to_position(det['bbox'], img_w, img_h)
        lines.append(
            f"  {i}. {det['class']} (conf={det['confidence']:.2f}) "
            f"at {pos['direction']} ({pos['cx']}, {pos['cy']}) "
            f"size={pos['area']:.3f}"
        )
    return "\n".join(lines)

def get_image_dimensions(img_path):
    """Get image width and height"""
    img = cv2.imread(str(img_path))
    if img is None:
        return 640, 480  # fallback
    return img.shape[1], img.shape[0]  # width, height

# Prepare data with grounding context
print("\nPreparing image data...")
images_with_piles = []
for item in data['with_piles_details']:
    img_name = item['image']
    img_path = FRAMES_WITH_PILES_DIR / img_name
    
    if not img_path.exists():
        print(f"Image not found: {img_path}")
        continue
    
    img_w, img_h = get_image_dimensions(img_path)
    grounding_context = format_grounding_context(item['detections'], img_w, img_h)
    
    images_with_piles.append({
        'image': img_name,
        'path': img_path,
        'detections': item['detections'],
        'grounding_context': grounding_context,
        'width': img_w,
        'height': img_h
    })

print(f"Found {len(images_with_piles)} valid images with piles")

# Sample if too many
if len(images_with_piles) > MAX_SAMPLES:
    images_with_piles = random.sample(images_with_piles, MAX_SAMPLES)
    print(f"Sampled {MAX_SAMPLES} images for VQA generation")

# System prompt
SYSTEM_PROMPT = """You are generating training data for a Vision Language Action Model (VLAM) that helps wheel loader operators.

The image contains detected piles/materials with position information provided.

Generate 3 VQA pairs where:
- Questions are operational commands or navigation queries
- Answers reference the detected object positions and provide actionable guidance
- Use compass directions (ahead, left, right) and coordinates
- Suggest realistic movements (distances, angles)

**Output format:** JSON array only, no markdown: [{"question": "...", "answer": "..."}, ...]

**Question examples:**
- "Where should I go next?"
- "Go to the nearest pile and fill the bucket!"
- "What material do you see?"
- "Direct me to the closest pile."
- "Is there anything to load ahead?"
- "Which pile should I approach first?"
- "Describe what you see in front of you."
- "Where is the material pile?"

**Answer format:**
- Reference positions: "Pile detected at center (0.52, 0.45)"
- Include action: "Move forward approximately 20 meters, no steering adjustment needed"
- Mention material type if known: "Large gravel pile visible ahead"
- If multiple piles: "Two piles detected: one at far-left (0.25, 0.38) and one ahead-right (0.70, 0.42). Recommend approaching the closer left pile first."
- Be specific about direction and distance
"""

def encode_image(image_path):
    """Encode image to base64"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode('utf-8')

def generate_vqa(img_data):
    """Generate VQA pairs via GPT-4o"""
    
    user_prompt = f"""Image: Construction site from wheel loader POV

{img_data['grounding_context']}

Generate 3 diverse VQA pairs for training. Output JSON array only."""

    base64_image = encode_image(img_data['path'])
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=500,
            temperature=0.7
        )
        
        content = response.choices[0].message.content.strip()
        
        # Clean markdown if present
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        
        vqa_pairs = json.loads(content)
        return vqa_pairs
    
    except Exception as e:
        print(f"\nError on {img_data['image']}: {e}")
        return []

# Generate VQA pairs
print(f"\nStarting VQA generation for {len(images_with_piles)} images...")
print(f"Expected output: ~{len(images_with_piles) * 3} VQA pairs\n")

all_vqa_pairs = []
failed_count = 0

for img_data in tqdm(images_with_piles, desc="Generating VQA"):
    vqa_pairs = generate_vqa(img_data)
    
    if vqa_pairs:
        for pair in vqa_pairs:
            all_vqa_pairs.append({
                'image': img_data['image'],
                'grounding_context': img_data['grounding_context'],
                'question': pair['question'],
                'answer': pair['answer']
            })
    else:
        failed_count += 1

# Save to JSONL
print(f"\nSaving results...")
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    for pair in all_vqa_pairs:
        f.write(json.dumps(pair, ensure_ascii=False) + '\n')

print(f"\nVQA Generation Complete!")
print(f"Total VQA pairs: {len(all_vqa_pairs)}")
print(f"Images processed: {len(images_with_piles) - failed_count}/{len(images_with_piles)}")
print(f"Failed: {failed_count}")
print(f"Saved to: {OUTPUT_FILE}")

# Show sample VQA pairs
print(f"\nSample VQA pairs:")
for i, pair in enumerate(random.sample(all_vqa_pairs, min(5, len(all_vqa_pairs))), 1):
    print(f"\n{i}. IMAGE: {pair['image']}")
    print(f"CONTEXT: {pair['grounding_context'][:100]}...")
    print(f"Q: {pair['question']}")
    print(f"A: {pair['answer'][:120]}...")

# Save a pretty-printed sample for inspection
sample_file = OUTPUT_FILE.parent / "vqa_samples.json"
with open(sample_file, 'w', encoding='utf-8') as f:
    json.dump(random.sample(all_vqa_pairs, min(10, len(all_vqa_pairs))), f, indent=2, ensure_ascii=False)

print(f"\nSample file saved: {sample_file}")