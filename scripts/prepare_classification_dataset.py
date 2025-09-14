#!/usr/bin/env python3
"""
Prepare classification dataset for checkbox state recognition.

Combines:
1. Detected checkbox crops from our detection model
2. CheckboxQA labels for contextual understanding
3. Synthetic checkbox crops with known states
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from tqdm import tqdm
import argparse
from ultralytics import YOLO

class ClassificationDatasetBuilder:
    """Build classification dataset from detection results + CheckboxQA."""
    
    def __init__(self, output_dir: str = "data/classification"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create class directories
        for class_name in ['checked', 'unchecked', 'unclear']:
            (self.output_dir / class_name).mkdir(exist_ok=True)
        
        self.stats = {
            'checked': 0,
            'unchecked': 0, 
            'unclear': 0,
            'total_crops': 0
        }
    
    def extract_checkbox_crops_from_detection(self, 
                                            detection_model_path: str,
                                            image_dir: Path,
                                            min_conf: float = 0.5) -> List[Dict]:
        """
        Use trained detection model to extract checkbox crops from documents.
        
        Returns:
            List of crop info with image path, bbox, confidence
        """
        print(f"🔍 Extracting checkbox crops using {detection_model_path}")
        
        if not Path(detection_model_path).exists():
            print(f"❌ Detection model not found: {detection_model_path}")
            return []
        
        model = YOLO(detection_model_path)
        crops = []
        
        if not image_dir.exists():
            print(f"❌ Image directory not found: {image_dir}")
            return []
        
        for img_path in tqdm(image_dir.glob("*.png"), desc="Processing images"):
            results = model(str(img_path), conf=min_conf)
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for i, box in enumerate(boxes):
                        conf = box.conf[0].item()
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        crops.append({
                            'image_path': str(img_path),
                            'bbox': [int(x1), int(y1), int(x2), int(y2)],
                            'confidence': conf,
                            'crop_id': f"{img_path.stem}_{i}"
                        })
        
        print(f"✅ Extracted {len(crops)} checkbox crops")
        return crops
    
    def load_checkboxqa_labels(self) -> Dict[str, List[Dict]]:
        """Load CheckboxQA annotations."""
        checkboxqa_file = Path("data/raw/checkboxqa/data/gold.jsonl")
        
        if not checkboxqa_file.exists():
            print(f"❌ CheckboxQA file not found: {checkboxqa_file}")
            return {}
        
        print(f"📋 Loading CheckboxQA labels from {checkboxqa_file}")
        
        labels_by_doc = {}
        with open(checkboxqa_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                doc_name = data['name']
                annotations = data['annotations']
                
                labels_by_doc[doc_name] = annotations
        
        print(f"✅ Loaded labels for {len(labels_by_doc)} documents")
        return labels_by_doc
    
    def classify_checkbox_from_context(self, question: str, answer: str) -> str:
        """
        Classify checkbox state based on question and answer context.
        
        Args:
            question: The form question
            answer: The form answer
            
        Returns:
            'checked', 'unchecked', or 'unclear'
        """
        question = question.lower()
        answer = answer.lower().strip()
        
        # Yes/No questions
        if any(word in question for word in ['is ', 'are ', 'do ', 'did ', 'have ', 'was ', 'were ']):
            if answer in ['yes', 'y', 'true', '1', 'checked']:
                return 'checked'
            elif answer in ['no', 'n', 'false', '0', 'unchecked', '']:
                return 'unchecked'
            elif answer in ['na', 'n/a', 'not applicable', 'unclear']:
                return 'unclear'
        
        # Selection questions (what is selected/checked)
        if any(phrase in question for phrase in ['what is selected', 'what is checked', 'what option']):
            if answer and answer != 'none' and answer != '':
                return 'checked'  # Something was selected
            else:
                return 'unchecked'  # Nothing selected
        
        # Multiple choice with specific values
        if answer and len(answer) > 0 and answer not in ['none', 'n/a', 'na']:
            return 'checked'  # Has a value, likely checked
        
        return 'unclear'  # Default for ambiguous cases
    
    def generate_synthetic_crops(self, num_crops: int = 1000) -> List[Dict]:
        """Generate synthetic checkbox crops with known states."""
        print(f"🎨 Generating {num_crops} synthetic checkbox crops")
        
        from generate_synthetic_checkboxes import SyntheticCheckboxGenerator, CheckboxStyle
        
        generator = SyntheticCheckboxGenerator()
        synthetic_crops = []
        
        for i in tqdm(range(num_crops), desc="Generating synthetic crops"):
            # Random style
            style = CheckboxStyle(
                size=random.randint(20, 40),
                border_width=random.randint(1, 3),
                border_color=random.choice([(0,0,0), (50,50,50), (100,100,100)]),
                check_style=random.choice(['checkmark', 'x', 'filled', 'dot']),
                rounded_corners=random.choice([True, False])
            )
            
            # Random state
            state = random.choice(['checked', 'unchecked', 'unclear'])
            
            # Generate checkbox image
            checkbox_img = generator.generate_checkbox(style, state)
            
            # Add some padding and resize to standard size
            padding = 10
            padded_size = checkbox_img.shape[0] + 2 * padding
            padded_img = np.ones((padded_size, padded_size, 3), dtype=np.uint8) * 255
            
            h, w = checkbox_img.shape[:2]
            padded_img[padding:padding+h, padding:padding+w] = checkbox_img
            
            # Resize to standard classification size (64x64)
            resized = cv2.resize(padded_img, (64, 64))
            
            synthetic_crops.append({
                'image': resized,
                'state': state,
                'source': 'synthetic',
                'crop_id': f"synthetic_{i:04d}"
            })
        
        return synthetic_crops
    
    def save_crop_image(self, image: np.ndarray, class_name: str, crop_id: str) -> str:
        """Save crop image to appropriate class directory."""
        output_path = self.output_dir / class_name / f"{crop_id}.png"
        cv2.imwrite(str(output_path), image)
        return str(output_path)
    
    def create_classification_dataset(self, 
                                    detection_model_path: str = "models/best.pt",
                                    synthetic_crops: int = 1000):
        """Create complete classification dataset."""
        print("🚀 Building classification dataset...\n")
        
        # Step 1: Generate synthetic crops (we know these labels)
        synthetic_data = self.generate_synthetic_crops(synthetic_crops)
        
        for crop_data in tqdm(synthetic_data, desc="Saving synthetic crops"):
            class_name = crop_data['state']
            crop_id = crop_data['crop_id']
            
            # Save image
            self.save_crop_image(crop_data['image'], class_name, crop_id)
            self.stats[class_name] += 1
            self.stats['total_crops'] += 1
        
        # Step 2: Extract crops from FUNSD using detection model
        funsd_dirs = [
            Path("data/raw/funsd/dataset/training_data/images"),
            Path("data/raw/funsd/dataset/testing_data/images")
        ]
        
        for funsd_dir in funsd_dirs:
            if funsd_dir.exists():
                crops = self.extract_checkbox_crops_from_detection(
                    detection_model_path, funsd_dir, min_conf=0.5
                )
                
                # Process detected crops
                for crop_info in tqdm(crops, desc=f"Processing {funsd_dir.name}"):
                    img_path = Path(crop_info['image_path'])
                    bbox = crop_info['bbox']
                    crop_id = crop_info['crop_id']
                    
                    # Load and crop image
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        x1, y1, x2, y2 = bbox
                        crop = img[y1:y2, x1:x2]
                        
                        if crop.size > 0:
                            # Resize to standard size
                            crop_resized = cv2.resize(crop, (64, 64))
                            
                            # For FUNSD, we don't have ground truth labels, 
                            # so classify as 'unclear' for now
                            # TODO: Could use heuristics based on crop appearance
                            class_name = 'unclear'
                            
                            self.save_crop_image(crop_resized, class_name, crop_id)
                            self.stats[class_name] += 1
                            self.stats['total_crops'] += 1
        
        # Step 3: Create train/val/test splits
        self.create_splits()
        
        # Step 4: Print statistics
        self.print_statistics()
    
    def create_splits(self, train_ratio: float = 0.7, val_ratio: float = 0.2):
        """Create train/val/test splits for each class."""
        print("\n📂 Creating train/val/test splits...")
        
        for split in ['train', 'val', 'test']:
            for class_name in ['checked', 'unchecked', 'unclear']:
                (self.output_dir / split / class_name).mkdir(parents=True, exist_ok=True)
        
        for class_name in ['checked', 'unchecked', 'unclear']:
            class_dir = self.output_dir / class_name
            images = list(class_dir.glob("*.png"))
            
            if not images:
                continue
                
            random.shuffle(images)
            
            # Calculate split indices
            n_total = len(images)
            n_train = int(n_total * train_ratio)
            n_val = int(n_total * val_ratio)
            
            splits = {
                'train': images[:n_train],
                'val': images[n_train:n_train + n_val],
                'test': images[n_train + n_val:]
            }
            
            # Move images to split directories
            for split, split_images in splits.items():
                for img_path in split_images:
                    dest_path = self.output_dir / split / class_name / img_path.name
                    img_path.rename(dest_path)
        
        # Remove empty class directories
        for class_name in ['checked', 'unchecked', 'unclear']:
            class_dir = self.output_dir / class_name
            if class_dir.exists() and not list(class_dir.iterdir()):
                class_dir.rmdir()
    
    def print_statistics(self):
        """Print dataset statistics."""
        print("\n" + "="*50)
        print("📊 CLASSIFICATION DATASET STATISTICS")
        print("="*50)
        
        # Count final split sizes
        split_stats = {}
        for split in ['train', 'val', 'test']:
            split_stats[split] = {}
            for class_name in ['checked', 'unchecked', 'unclear']:
                split_dir = self.output_dir / split / class_name
                if split_dir.exists():
                    count = len(list(split_dir.glob("*.png")))
                    split_stats[split][class_name] = count
                else:
                    split_stats[split][class_name] = 0
        
        # Print class distribution
        for class_name in ['checked', 'unchecked', 'unclear']:
            total = sum(split_stats[split][class_name] for split in ['train', 'val', 'test'])
            train = split_stats['train'][class_name]
            val = split_stats['val'][class_name]
            test = split_stats['test'][class_name]
            
            print(f"{class_name.upper():>10}: {total:>4} total | Train: {train:>3} | Val: {val:>3} | Test: {test:>3}")
        
        # Print totals
        total_all = sum(sum(split_stats[split].values()) for split in ['train', 'val', 'test'])
        total_train = sum(split_stats['train'].values())
        total_val = sum(split_stats['val'].values())
        total_test = sum(split_stats['test'].values())
        
        print("-" * 50)
        print(f"{'TOTAL':>10}: {total_all:>4} total | Train: {total_train:>3} | Val: {total_val:>3} | Test: {total_test:>3}")
        print("="*50)
        print(f"✅ Classification dataset ready!")
        print(f"📁 Location: {self.output_dir.absolute()}")
        print(f"🎯 Ready for EfficientNet training")
        print("="*50)

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Prepare classification dataset")
    parser.add_argument("--output-dir", type=str, default="data/classification",
                       help="Output directory")
    parser.add_argument("--detection-model", type=str, default="models/best.pt",
                       help="Path to trained detection model")
    parser.add_argument("--synthetic-crops", type=int, default=1000,
                       help="Number of synthetic crops to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # Build dataset
    builder = ClassificationDatasetBuilder(output_dir=args.output_dir)
    builder.create_classification_dataset(
        detection_model_path=args.detection_model,
        synthetic_crops=args.synthetic_crops
    )
    
    return 0

if __name__ == "__main__":
    exit(main())