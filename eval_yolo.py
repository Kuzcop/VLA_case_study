from ultralytics import YOLO
from pathlib import Path
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import shutil
import pickle

class PileDetectionVerifier:
    def __init__(self, image_dir, output_dir="verification_output"):
        self.image_dir = Path(image_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def verify_detections(self, model_name, class_names, conf_threshold=0.05, 
                         batch_size=10, samples_per_range=5, save_predictions=True):
        """
        Run detection and create verification visualizations
        Returns detections grouped by confidence ranges
        """
        print(f"\n{'='*60}")
        print(f"Verifying detections for classes: {class_names}")
        print(f"{'='*60}\n")
        
        model = YOLO(model_name)
        model.set_classes(class_names)
        
        jpg_paths = sorted(list(self.image_dir.glob("*.jpg")))
        
        # Process in batches
        all_results = []
        all_detections = []
        
        for i in range(0, len(jpg_paths), batch_size):
            batch_paths = jpg_paths[i:i+batch_size]
            batch_results = model.predict(batch_paths, conf=conf_threshold, verbose=False)
            all_results.extend(batch_results)
            
            for idx, result in enumerate(batch_results):
                img_idx = i + idx
                img_detections = {
                    'image_path': str(batch_paths[idx]),
                    'image_name': batch_paths[idx].name,
                    'detections': []
                }
                
                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls_id = int(box.cls[0])
                        cls_name = class_names[cls_id]
                        
                        img_detections['detections'].append({
                            'class': cls_name,
                            'confidence': round(conf, 3),
                            'bbox': [float(x1), float(y1), float(x2), float(y2)]
                        })
                
                all_detections.append(img_detections)
        
        # Save predictions if requested
        if save_predictions:
            self._save_predictions(all_results, all_detections, jpg_paths, class_names)
        
        # Group by confidence ranges
        confidence_ranges = {
            'high': (0.5, 1.0),
            'medium': (0.2, 0.5),
            'low': (0.05, 0.2)
        }
        
        grouped = {k: [] for k in confidence_ranges.keys()}
        
        for i, det in enumerate(all_detections):
            if det['detections']:
                max_conf = max(d['confidence'] for d in det['detections'])
                for range_name, (min_conf, max_conf_range) in confidence_ranges.items():
                    if min_conf <= max_conf < max_conf_range:
                        grouped[range_name].append((i, det, all_results[i]))
                        break
        
        # Visualize samples from each range
        for range_name, items in grouped.items():
            if items:
                print(f"\n{range_name.upper()} confidence ({confidence_ranges[range_name][0]}-{confidence_ranges[range_name][1]}): {len(items)} images")
                self._visualize_samples(items, range_name, samples_per_range, class_names)
        
        return all_detections, all_results, jpg_paths
    
    def _save_predictions(self, results, detections, jpg_paths, class_names):
        """Save prediction results for later visualization"""
        predictions_dir = self.output_dir / 'predictions'
        predictions_dir.mkdir(exist_ok=True)
        
        # Save as pickle (preserves full Results objects)
        pickle_path = predictions_dir / 'results.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump({
                'results': results,
                'detections': detections,
                'image_paths': [str(p) for p in jpg_paths],
                'class_names': class_names
            }, f)
        print(f"Saved prediction objects to: {pickle_path}")
        
        # Save as JSON (human-readable)
        json_path = predictions_dir / 'detections.json'
        with open(json_path, 'w') as f:
            json.dump({
                'detections': detections,
                'class_names': class_names,
                'total_images': len(jpg_paths),
                'images_with_detections': sum(1 for d in detections if d['detections'])
            }, f, indent=2)
        print(f"Saved detections JSON to: {json_path}")
        
        # Save individual annotated images
        annotated_dir = predictions_dir / 'annotated_images'
        annotated_dir.mkdir(exist_ok=True)
        
        print(f"Saving annotated images...")
        saved_count = 0
        for i, (result, detection) in enumerate(zip(results, detections)):
            if detection['detections']:  # Only save images with detections
                # Use YOLO's built-in plotting
                annotated = result.plot()  # Returns annotated image as numpy array
                annotated_img = Image.fromarray(annotated)
                
                save_path = annotated_dir / jpg_paths[i].name
                annotated_img.save(save_path)
                saved_count += 1
        
        print(f"Saved {saved_count} annotated images to: {annotated_dir}")
    
    def _visualize_samples(self, items, range_name, num_samples, class_names):
        """Visualize sample detections"""
        num_samples = min(num_samples, len(items))
        # Sample evenly across the range
        indices = [int(i * len(items) / num_samples) for i in range(num_samples)]
        
        fig, axes = plt.subplots(1, num_samples, figsize=(5*num_samples, 5))
        if num_samples == 1:
            axes = [axes]
        
        for plot_idx, item_idx in enumerate(indices):
            i, det, result = items[item_idx]
            img = Image.open(det['image_path'])
            
            axes[plot_idx].imshow(img)
            axes[plot_idx].axis('off')
            
            # Title with image name and detection info
            title = f"{det['image_name']}\n"
            for d in det['detections']:
                title += f"{d['class']}: {d['confidence']:.2f}\n"
            axes[plot_idx].set_title(title, fontsize=8)
            
            # Draw boxes
            for d in det['detections']:
                x1, y1, x2, y2 = d['bbox']
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[plot_idx].add_patch(rect)
                
                # Label with class and confidence
                label = f"{d['class']}\n{d['confidence']:.2f}"
                axes[plot_idx].text(x1, y1-5, label,
                                   color='red', fontsize=8, weight='bold',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        save_path = self.output_dir / f'{range_name}_confidence_samples.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved visualization to: {save_path}")
        plt.close()
    
    def organize_images(self, detections, jpg_paths, class_set_name, 
                       min_confidence=0.15):
        """
        Organize images into 'with_piles' and 'without_piles' folders
        """
        print(f"\n{'='*60}")
        print(f"Organizing images for: {class_set_name}")
        print(f"Minimum confidence threshold: {min_confidence}")
        print(f"{'='*60}\n")
        
        # Create output directories
        base_dir = self.output_dir / class_set_name
        with_piles_dir = base_dir / 'with_piles'
        without_piles_dir = base_dir / 'without_piles'
        
        with_piles_dir.mkdir(parents=True, exist_ok=True)
        without_piles_dir.mkdir(parents=True, exist_ok=True)
        
        # Categorize and copy images
        with_piles = []
        without_piles = []
        
        for i, det in enumerate(detections):
            # Check if has valid detections above threshold
            valid_detections = [d for d in det['detections'] 
                              if d['confidence'] >= min_confidence]
            
            src_path = Path(det['image_path'])
            
            if valid_detections:
                # Copy to with_piles
                dst_path = with_piles_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                with_piles.append({
                    'image': src_path.name,
                    'detections': valid_detections
                })
            else:
                # Copy to without_piles
                dst_path = without_piles_dir / src_path.name
                shutil.copy2(src_path, dst_path)
                without_piles.append(src_path.name)
        
        # Save metadata
        metadata = {
            'class_set_name': class_set_name,
            'min_confidence': min_confidence,
            'total_images': len(detections),
            'with_piles': len(with_piles),
            'without_piles': len(without_piles),
            'with_piles_details': with_piles
        }
        
        metadata_path = base_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Images with piles: {len(with_piles)} → {with_piles_dir}")
        print(f"Images without piles: {len(without_piles)} → {without_piles_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        # Print detection statistics
        if with_piles:
            print("\nDetection statistics:")
            class_counts = {}
            for item in with_piles:
                for d in item['detections']:
                    cls = d['class']
                    class_counts[cls] = class_counts.get(cls, 0) + 1
            
            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {cls}: {count} detections")
        
        return with_piles_dir, without_piles_dir, metadata

    @staticmethod
    def load_predictions(predictions_dir):
        """Load saved predictions for visualization"""
        predictions_dir = Path(predictions_dir)
        pickle_path = predictions_dir / 'results.pkl'
        
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        return data['results'], data['detections'], data['image_paths'], data['class_names']
    
    @staticmethod
    def visualize_saved_predictions(predictions_dir, image_indices=None, save_output=True):
        """
        Visualize saved predictions interactively
        
        Args:
            predictions_dir: Path to predictions folder
            image_indices: List of image indices to visualize, or None for all with detections
            save_output: Whether to save visualization
        """
        results, detections, image_paths, class_names = PileDetectionVerifier.load_predictions(predictions_dir)
        
        # If no indices specified, show all images with detections
        if image_indices is None:
            image_indices = [i for i, d in enumerate(detections) if d['detections']]
            print(f"Found {len(image_indices)} images with detections")
        
        # Limit to reasonable number for display
        if len(image_indices) > 20:
            print(f"Showing first 20 out of {len(image_indices)} images")
            image_indices = image_indices[:20]
        
        # Create grid
        n_images = len(image_indices)
        n_cols = min(5, n_images)
        n_rows = (n_images + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_images == 1:
            axes = [axes]
        else:
            axes = axes.flatten() if n_images > 1 else [axes]
        
        for plot_idx, img_idx in enumerate(image_indices):
            result = results[img_idx]
            det = detections[img_idx]
            
            # Load and display image
            img = Image.open(image_paths[img_idx])
            axes[plot_idx].imshow(img)
            axes[plot_idx].axis('off')
            
            # Title
            title = f"{Path(image_paths[img_idx]).name}\n"
            for d in det['detections']:
                title += f"{d['class']}: {d['confidence']:.2f}\n"
            axes[plot_idx].set_title(title, fontsize=8)
            
            # Draw boxes
            for d in det['detections']:
                x1, y1, x2, y2 = d['bbox']
                rect = patches.Rectangle(
                    (x1, y1), x2-x1, y2-y1,
                    linewidth=2, edgecolor='red', facecolor='none'
                )
                axes[plot_idx].add_patch(rect)
                
                label = f"{d['class']}\n{d['confidence']:.2f}"
                axes[plot_idx].text(x1, y1-5, label,
                                   color='red', fontsize=7, weight='bold',
                                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide extra subplots
        for idx in range(n_images, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_output:
            output_path = Path(predictions_dir).parent / 'prediction_review.png'
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to: {output_path}")
        
        plt.show()
        
        return results, detections

# Initialize verifier
verifier = PileDetectionVerifier(
    image_dir=r"C:/Github_workspace/vlam_sensmore/frames/",
    output_dir="pile_detection_verification"
)

# Test the two best performing class sets
test_configs = [
    {
        'name': 'simple_nouns',
        'classes': ['pile', 'heap', 'mound', 'dirt', 'gravel', 'sand', 'soil']
    },
    {
        'name': 'descriptive_phrases',
        'classes': ['pile of dirt', 'pile of gravel', 'pile of sand',
                   'mound of soil', 'heap of rocks', 'loose material',
                   'construction debris', 'excavated material']
    }
]

results_summary = {}

for config in test_configs:
    print(f"\n\n{'#'*60}")
    print(f"# TESTING: {config['name']}")
    print(f"{'#'*60}")
    
    # 1. Verify detections with visualizations (saves predictions automatically)
    detections, results, jpg_paths = verifier.verify_detections(
        'yolov8x-worldv2.pt',
        config['classes'],
        conf_threshold=0.05,
        batch_size=10,
        samples_per_range=5,
        save_predictions=True  # This saves the predictions
    )
    
    # 2. Organize images
    with_piles_dir, without_piles_dir, metadata = verifier.organize_images(
        detections,
        jpg_paths,
        config['name'],
        min_confidence=0.05
    )
    
    results_summary[config['name']] = metadata

# Save overall summary
summary_path = Path("pile_detection_verification") / "overall_summary.json"
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2)

print("\n\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for name, metadata in results_summary.items():
    print(f"\n{name}:")
    print(f"  Total images: {metadata['total_images']}")
    print(f"  With piles: {metadata['with_piles']} ({metadata['with_piles']/metadata['total_images']*100:.1f}%)")
    print(f"  Without piles: {metadata['without_piles']} ({metadata['without_piles']/metadata['total_images']*100:.1f}%)")

print(f"\nAll results saved to: pile_detection_verification/")
print(f"Predictions saved to: pile_detection_verification/predictions/")