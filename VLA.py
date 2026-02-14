import torch
from unsloth import FastLanguageModel
from ultralytics import YOLO
import cv2
from pathlib import Path
import json
import random

class VLAMPipeline:
    """Vision Language Action Model for Wheel Loader Operations"""
    
    def __init__(self, 
                 llm_path="C:/Github_workspace/vlam_sensmore/llama-vlam-final/",
                 yolo_model="yolov8x-worldv2.pt"):
        
        print("Initializing VLAM Pipeline...")
        
        # Load YOLO detector
        print("Loading YOLO-World detector...")
        self.yolo = YOLO(yolo_model)
        self.yolo.set_classes(['pile', 'heap', 'mound', 'dirt', 'gravel', 'sand', 'soil'])
        print("YOLO loaded")
        
        # Load fine-tuned LLM
        print("📦 Loading fine-tuned LLaMA 3.2 1B...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=llm_path,
            max_seq_length=512,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)  # Enable inference mode
        print("LLM loaded")
            
    def detect_objects(self, image_path):
        """Run YOLO detection on image"""
        # Load image
        img = cv2.imread(str(image_path))
        img_h, img_w = img.shape[:2]
        
        # Run detection
        results = self.yolo.predict(image_path, conf=0.05, verbose=False)
        
        # Extract detections
        detections = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Calculate normalized position (convert to native Python float)
                cx = float(((x1 + x2) / 2) / img_w)
                cy = float(((y1 + y2) / 2) / img_h)
                area = float(((x2-x1) * (y2-y1)) / (img_w * img_h))
                
                # Semantic direction
                h = 'left' if cx < 0.4 else ('right' if cx > 0.6 else 'center')
                v = 'far' if cy < 0.35 else ('near' if cy > 0.65 else 'mid')
                direction = f'{v}-{h}'
                
                detections.append({
                    'class': self.yolo.names[int(box.cls[0])],
                    'confidence': float(box.conf[0]),  # Convert to Python float
                    'position': {
                        'cx': round(cx, 3),
                        'cy': round(cy, 3),
                        'area': round(area, 4),
                        'direction': direction,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]  # Convert to Python int
                    }
                })
        
        return detections, img, (img_w, img_h)
    
    def format_grounding_context(self, detections):
        """Format detections as text for LLM"""
        if not detections:
            return "[SCENE] Construction site view from wheel loader.\n[DETECTIONS] No objects detected in current view."
        
        lines = ["[SCENE] Construction site view from wheel loader.", "[DETECTIONS]"]
        for i, det in enumerate(detections, 1):
            pos = det['position']
            lines.append(
                f"  {i}. {det['class']} (conf={det['confidence']:.2f}) "
                f"at {pos['direction']} ({pos['cx']}, {pos['cy']}) "
                f"size={pos['area']:.3f}"
            )
        return "\n".join(lines)
    
    def generate_response(self, grounding_context, command):
        """Generate LLM response given context and command"""
        
        # Build prompt in LLaMA 3 format
        prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

    You are a Vision Language Action Model assisting wheel loader operators. You receive scene descriptions with detected objects and their positions, then provide clear operational guidance.<|eot_id|><|start_header_id|>user<|end_header_id|>

    {grounding_context}

    [COMMAND] {command}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

    """
        
        # Tokenize
        inputs = self.tokenizer([prompt], return_tensors="pt").to("cuda")
        
        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        # Decode full response
        full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract ONLY the assistant's answer (everything after the last "assistant" tag)
        if "assistant" in full_response:
            # Split and take everything after the last occurrence
            parts = full_response.split("assistant")
            response = parts[-1].strip()
            # Remove any leading special tokens or whitespace
            response = response.lstrip("<|end_header_id|>").strip()
        else:
            response = full_response.strip()
        
        return response
    
    def process(self, image_path, command):
        """Full pipeline: detect → ground → respond"""
        
        print(f"\nProcessing: {Path(image_path).name}")
        print(f"Command: {command}\n")
        
        # Step 1: Detect objects
        detections, img, (img_w, img_h) = self.detect_objects(image_path)
        print(f"Detected {len(detections)} objects")
        
        # Step 2: Format grounding context
        grounding_context = self.format_grounding_context(detections)
        print(f"\n{grounding_context}\n")
        
        # Step 3: Generate response
        response = self.generate_response(grounding_context, command)
        print(f"VLAM Response:\n{response}\n")
        
        return {
            'image': str(image_path),
            'command': command,
            'detections': detections,
            'grounding_context': grounding_context,
            'response': response
        }
    
    def visualize(self, image_path, result, output_path=None):
        """Draw detections on image"""
        img = cv2.imread(str(image_path))
        
        for det in result['detections']:
            bbox = det['position']['bbox']
            label = f"{det['class']} {det['confidence']:.2f}"
            
            # Draw box
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            cv2.putText(img, label, (bbox[0], bbox[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add response text at bottom
        response_lines = result['response'][:200].split('\n')  # First 200 chars
        y_offset = img.shape[0] - 60
        for line in response_lines[:2]:  # Max 2 lines
            cv2.putText(img, line, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            y_offset += 20
        
        if output_path:
            cv2.imwrite(str(output_path), img)
            print(f"💾 Saved visualization: {output_path}")
        
        return img


# ============ DEMO SCRIPT ============
if __name__ == "__main__":
    
    # Initialize pipeline
    vlam = VLAMPipeline(
        llm_path="C:/Github_workspace/vlam_sensmore/llama-vlam-final/",
        yolo_model="yolov8x-worldv2.pt"
    )
    
    # Test on sample images
    test_dir = Path("C:/Github_workspace/vlam_sensmore/pile_detection_verification/simple_nouns/with_piles")
    all_images = list(test_dir.glob("*.jpg"))
    test_images = random.sample(all_images, min(3, len(all_images)))
    
    commands = [
        "Where should I go next?",
        "Go to the nearest pile and fill the bucket!",
        "What do you see ahead?"
    ]
    
    output_dir = Path("C:/Github_workspace/vlam_sensmore/outputs")
    output_dir.mkdir(exist_ok=True)
    
    for img_path in test_images:
        for command in commands:
            # Process
            result = vlam.process(img_path, command)
            
            # Visualize
            vis_path = output_dir / f"{img_path.stem}_{command[:20].replace(' ', '_')}.jpg"
            vlam.visualize(img_path, result, vis_path)
            
            # Save JSON
            json_path = output_dir / f"{img_path.stem}_{command[:20].replace(' ', '_')}.json"
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            print("="*60)
    
    print("\nDemo complete! Check C:/Github_workspace/vlam_sensmore/outputs")