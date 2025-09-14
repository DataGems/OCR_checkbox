#!/usr/bin/env python3
"""
Generate balanced classification dataset combining real and synthetic checkbox crops.
Target: 30-40% synthetic for better generalization than detection's 90%.
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import random
from tqdm import tqdm
import yaml
from ultralytics import YOLO

class ClassificationDatasetGenerator:
    """Generate balanced classification dataset for checkbox states."""
    
    def __init__(self, detection_model_path: str = "models/best.pt"):
        self.detection_model = YOLO(detection_model_path)
        self.crop_size = (32, 32)  # Standard classification input
        
    def extract_checkboxes_from_funsd(self, funsd_dir: Path, max_samples: int = 150) -> List[Dict]:
        """Extract checkbox crops from FUNSD dataset."""
        crops = []
        images_dir = funsd_dir / "training_data" / "images"
        annotations_dir = funsd_dir / "training_data" / "annotations"
        
        if not images_dir.exists() or not annotations_dir.exists():
            print(f"⚠️ FUNSD directories not found: {funsd_dir}")
            return crops
        
        image_files = list(images_dir.glob("*.png"))[:max_samples]
        
        for img_file in tqdm(image_files, desc="Extracting FUNSD checkboxes"):
            # Load image
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            # Run detection to find checkboxes
            results = self.detection_model(image, conf=0.3)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        conf = box.conf[0].item()
                        
                        if conf > 0.5:  # Only high-confidence detections
                            # Extract crop
                            crop = image[int(y1):int(y2), int(x1):int(x2)]
                            if crop.size > 0:
                                # Classify checkbox state (heuristic)
                                state = self._classify_checkbox_heuristic(crop)
                                
                                # Resize to standard size
                                crop_resized = cv2.resize(crop, self.crop_size)
                                
                                crops.append({
                                    'image': crop_resized,
                                    'label': state,
                                    'source': 'funsd',
                                    'confidence': conf,
                                    'original_size': (int(x2-x1), int(y2-y1))
                                })
        
        print(f"✅ Extracted {len(crops)} FUNSD checkbox crops")
        return crops
    
    def extract_checkboxes_from_checkboxqa(self, checkboxqa_dir: Path, max_samples: int = 200) -> List[Dict]:
        """Extract checkbox crops from CheckboxQA PDFs (if available)."""
        crops = []
        
        # For now, return empty since we need real CheckboxQA PDFs
        # This would be implemented with PDF processing + detection
        print(f"⚠️ CheckboxQA crop extraction not implemented yet")
        print(f"   Need to: 1) Download PDFs, 2) Convert to images, 3) Run detection")
        
        return crops
    
    def generate_synthetic_checkboxes(self, num_samples: int = 400) -> List[Dict]:
        """Generate synthetic checkbox crops."""
        crops = []
        
        states = ['checked', 'unchecked', 'unclear']
        state_weights = [0.4, 0.5, 0.1]  # More unchecked (common in real forms)
        
        for i in tqdm(range(num_samples), desc="Generating synthetic checkboxes"):
            # Choose state
            state = np.random.choice(states, p=state_weights)
            
            # Generate checkbox
            checkbox_img = self._generate_single_checkbox(state)
            
            crops.append({
                'image': checkbox_img,
                'label': state,
                'source': 'synthetic',
                'confidence': 1.0,
                'original_size': self.crop_size
            })
        
        return crops
    
    def _generate_single_checkbox(self, state: str) -> np.ndarray:
        """Generate a single synthetic checkbox."""
        img = np.ones((*self.crop_size, 3), dtype=np.uint8) * 255  # White background
        
        # Add some background variation
        noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
        img = cv2.add(img, noise)
        img = np.clip(img, 240, 255)  # Keep it light
        
        # Draw checkbox border
        border_thickness = random.randint(1, 2)
        border_color = random.randint(0, 50)  # Dark border
        
        cv2.rectangle(img, (2, 2), (self.crop_size[0]-3, self.crop_size[1]-3), 
                     (border_color, border_color, border_color), border_thickness)
        
        # Add state-specific content
        if state == 'checked':
            # Draw check mark or X
            if random.random() > 0.5:
                # Check mark
                cv2.line(img, (8, 16), (14, 22), (0, 0, 0), 2)
                cv2.line(img, (14, 22), (24, 10), (0, 0, 0), 2)
            else:
                # X mark
                cv2.line(img, (8, 8), (24, 24), (0, 0, 0), 2)
                cv2.line(img, (24, 8), (8, 24), (0, 0, 0), 2)
        
        elif state == 'unclear':
            # Add some ambiguous marking
            center = (self.crop_size[0]//2, self.crop_size[1]//2)
            cv2.circle(img, center, 3, (100, 100, 100), -1)
        
        # Add slight blur and noise
        if random.random() > 0.7:
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        return img
    
    def _classify_checkbox_heuristic(self, crop: np.ndarray) -> str:
        """Heuristic classification of checkbox state from crop."""
        if crop.size == 0:
            return 'unclear'
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Calculate fill ratio (dark pixels / total pixels)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        fill_ratio = np.sum(binary == 255) / binary.size
        
        # Simple thresholding
        if fill_ratio > 0.3:
            return 'checked'
        elif fill_ratio < 0.1:
            return 'unchecked'
        else:
            return 'unclear'
    
    def generate_classification_dataset(self, 
                                      synthetic_ratio: float = 0.4,
                                      real_sources: List[str] = ['funsd'],
                                      total_samples: int = 1000,
                                      output_dir: Path = Path("data/classification")) -> Dict:
        """Generate balanced classification dataset."""
        
        print(f"🎯 Generating classification dataset:")
        print(f"   Target samples: {total_samples}")
        print(f"   Synthetic ratio: {synthetic_ratio:.1%}")
        print(f"   Real sources: {real_sources}")
        
        # Calculate sample distribution
        synthetic_count = int(total_samples * synthetic_ratio)
        real_count = total_samples - synthetic_count
        
        all_crops = []
        
        # Generate synthetic crops
        synthetic_crops = self.generate_synthetic_checkboxes(synthetic_count)
        all_crops.extend(synthetic_crops)
        
        # Extract real crops
        if 'funsd' in real_sources:
            funsd_crops = self.extract_checkboxes_from_funsd(
                Path("data/raw/funsd"), 
                max_samples=real_count // len(real_sources)
            )
            all_crops.extend(funsd_crops)
        
        if 'checkboxqa' in real_sources:
            checkboxqa_crops = self.extract_checkboxes_from_checkboxqa(
                Path("data/raw/checkboxqa"),
                max_samples=real_count // len(real_sources)
            )
            all_crops.extend(checkboxqa_crops)
        
        # Shuffle dataset
        random.shuffle(all_crops)
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into train/val/test
        n_train = int(0.7 * len(all_crops))
        n_val = int(0.2 * len(all_crops))
        
        splits = {
            'train': all_crops[:n_train],
            'val': all_crops[n_train:n_train+n_val],
            'test': all_crops[n_train+n_val:]
        }
        
        # Save splits
        for split_name, split_data in splits.items():
            split_dir = output_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            # Create class directories
            for label in ['checked', 'unchecked', 'unclear']:
                (split_dir / label).mkdir(exist_ok=True)
            
            # Save images
            for i, sample in enumerate(split_data):
                label = sample['label']
                source = sample['source']
                
                filename = f"{source}_{i:04d}.png"
                filepath = split_dir / label / filename
                
                cv2.imwrite(str(filepath), sample['image'])
        
        # Generate dataset statistics
        stats = self._calculate_dataset_stats(all_crops, splits)
        
        # Save metadata
        with open(output_dir / "dataset_info.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\n✅ Classification dataset generated:")
        print(f"   Total samples: {len(all_crops)}")
        print(f"   Synthetic: {len([c for c in all_crops if c['source'] == 'synthetic'])} ({synthetic_ratio:.1%})")
        print(f"   Real: {len([c for c in all_crops if c['source'] != 'synthetic'])}")
        print(f"   Train/Val/Test: {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")
        print(f"   Saved to: {output_dir}")
        
        return stats
    
    def _calculate_dataset_stats(self, all_crops: List[Dict], splits: Dict) -> Dict:
        """Calculate dataset statistics."""
        
        def get_label_counts(crops):
            counts = {'checked': 0, 'unchecked': 0, 'unclear': 0}
            for crop in crops:
                counts[crop['label']] += 1
            return counts
        
        def get_source_counts(crops):
            counts = {}
            for crop in crops:
                source = crop['source']
                counts[source] = counts.get(source, 0) + 1
            return counts
        
        stats = {
            'total_samples': len(all_crops),
            'label_distribution': get_label_counts(all_crops),
            'source_distribution': get_source_counts(all_crops),
            'splits': {
                split_name: {
                    'count': len(split_data),
                    'label_distribution': get_label_counts(split_data),
                    'source_distribution': get_source_counts(split_data)
                }
                for split_name, split_data in splits.items()
            },
            'synthetic_ratio': len([c for c in all_crops if c['source'] == 'synthetic']) / len(all_crops)
        }
        
        return stats

def main():
    """Generate classification dataset."""
    generator = ClassificationDatasetGenerator("models/best.pt")
    
    stats = generator.generate_classification_dataset(
        synthetic_ratio=0.4,  # 40% synthetic
        real_sources=['funsd'],  # Only FUNSD for now
        total_samples=800  # Manageable size
    )
    
    print(f"\n📊 Dataset Statistics:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()