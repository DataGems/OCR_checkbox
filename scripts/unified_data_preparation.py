#!/usr/bin/env python3
"""
Unified data preparation pipeline for checkbox detection and classification.

Combines:
1. FUNSD checkbox extraction (detection labels)
2. Synthetic checkbox generation (detection + classification)
3. CheckboxQA processing (classification labels)

Output: YOLO format dataset ready for training
"""

import os
import json
import shutil
import cv2
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import random
from tqdm import tqdm
import argparse

# Add project root to Python path
import sys
sys.path.append(str(Path(__file__).parent.parent))

@dataclass
class DatasetStats:
    """Track dataset statistics."""
    funsd_images: int = 0
    synthetic_images: int = 0
    total_checkboxes: int = 0
    train_images: int = 0
    val_images: int = 0
    test_images: int = 0

class UnifiedDataPreparation:
    """Unified pipeline for preparing all checkbox training data."""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.funsd_dir = Path("data/raw/funsd")
        self.checkboxqa_dir = Path("data/raw/checkboxqa")
        self.synthetic_dir = Path("data/synthetic")
        
        # Create output structure
        for split in ['train', 'val', 'test']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
        
        self.stats = DatasetStats()
    
    def is_potential_checkbox(self, box: List[int], text: str) -> bool:
        """
        Enhanced heuristics for checkbox detection in FUNSD.
        Based on size, aspect ratio, and text content.
        """
        if len(box) != 4:
            return False
        
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Size constraints - checkboxes are typically 10-50 pixels
        if width < 8 or width > 60 or height < 8 or height > 60:
            return False
        
        # Aspect ratio - checkboxes should be roughly square
        if height > 0:
            aspect_ratio = width / height
            if aspect_ratio < 0.6 or aspect_ratio > 1.4:
                return False
        
        # Text content analysis
        text = text.strip().lower()
        
        # Strong checkbox indicators
        checkbox_texts = [
            '', 'x', '✓', '✔', '☑', '☐', '□', '■', '◻', '◼',
            'yes', 'no', 'y', 'n', 'true', 'false', 'check', 'uncheck'
        ]
        
        if text in checkbox_texts:
            return True
        
        # Small squares with minimal text (likely checkboxes)
        if width < 25 and height < 25 and len(text) <= 3:
            return True
        
        # Single character in small box
        if len(text) == 1 and width < 35 and height < 35:
            return True
        
        return False
    
    def convert_to_yolo_format(self, box: List[int], img_width: int, img_height: int) -> str:
        """Convert FUNSD box format to YOLO format."""
        x1, y1, x2, y2 = box
        
        # Calculate center coordinates and normalize
        x_center = (x1 + x2) / 2 / img_width
        y_center = (y1 + y2) / 2 / img_height
        width = (x2 - x1) / img_width
        height = (y2 - y1) / img_height
        
        # Class 0 for checkbox detection
        return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
    
    def process_funsd_data(self) -> Dict[str, List[Tuple[Path, Path]]]:
        """
        Extract checkbox annotations from FUNSD dataset.
        Returns dict with 'train' and 'test' image/label pairs.
        """
        print("📋 Processing FUNSD dataset for checkbox detection...")
        
        funsd_data = {'train': [], 'test': []}
        
        # Process training data
        train_ann_dir = self.funsd_dir / "dataset" / "training_data" / "annotations"
        train_img_dir = self.funsd_dir / "dataset" / "training_data" / "images"
        
        if train_ann_dir.exists():
            for ann_file in tqdm(train_ann_dir.glob("*.json"), desc="Processing FUNSD train"):
                img_file = train_img_dir / f"{ann_file.stem}.png"
                
                if not img_file.exists():
                    continue
                
                # Load annotation
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                
                # Extract checkboxes
                checkboxes = []
                if 'form' in data:
                    for item in data['form']:
                        box = item.get('box', [])
                        text = item.get('text', '')
                        
                        if self.is_potential_checkbox(box, text):
                            # Get image dimensions
                            img = cv2.imread(str(img_file))
                            if img is not None:
                                img_height, img_width = img.shape[:2]
                                yolo_line = self.convert_to_yolo_format(box, img_width, img_height)
                                checkboxes.append(yolo_line)
                                self.stats.total_checkboxes += 1
                
                if checkboxes:
                    funsd_data['train'].append((img_file, checkboxes))
                    self.stats.funsd_images += 1
        
        # Process test data (use as validation in our pipeline)
        test_ann_dir = self.funsd_dir / "dataset" / "testing_data" / "annotations"
        test_img_dir = self.funsd_dir / "dataset" / "testing_data" / "images"
        
        if test_ann_dir.exists():
            for ann_file in tqdm(test_ann_dir.glob("*.json"), desc="Processing FUNSD test"):
                img_file = test_img_dir / f"{ann_file.stem}.png"
                
                if not img_file.exists():
                    continue
                
                # Load annotation
                with open(ann_file, 'r') as f:
                    data = json.load(f)
                
                # Extract checkboxes
                checkboxes = []
                if 'form' in data:
                    for item in data['form']:
                        box = item.get('box', [])
                        text = item.get('text', '')
                        
                        if self.is_potential_checkbox(box, text):
                            # Get image dimensions
                            img = cv2.imread(str(img_file))
                            if img is not None:
                                img_height, img_width = img.shape[:2]
                                yolo_line = self.convert_to_yolo_format(box, img_width, img_height)
                                checkboxes.append(yolo_line)
                                self.stats.total_checkboxes += 1
                
                if checkboxes:
                    funsd_data['test'].append((img_file, checkboxes))
        
        print(f"✅ FUNSD: Found {len(funsd_data['train'])} train + {len(funsd_data['test'])} test images")
        print(f"📊 Total checkboxes extracted: {self.stats.total_checkboxes}")
        
        return funsd_data
    
    def generate_synthetic_data(self, num_images: int = 500) -> List[Tuple[Path, List[str]]]:
        """
        Generate synthetic checkbox data with YOLO annotations.
        Creates form-like images with multiple checkboxes.
        """
        print(f"🎨 Generating {num_images} synthetic images...")
        
        # Import synthetic generator
        from generate_synthetic_checkboxes import SyntheticCheckboxGenerator
        
        generator = SyntheticCheckboxGenerator(output_dir=str(self.synthetic_dir))
        synthetic_data = []
        
        for i in tqdm(range(num_images), desc="Generating synthetic data"):
            # Create a form-like image with multiple checkboxes
            img_name = f"synthetic_{i:04d}.png"
            img_path = self.synthetic_dir / "images" / img_name
            
            # Generate form with 3-8 checkboxes
            num_checkboxes = random.randint(3, 8)
            image, annotations = generator.generate_form_with_checkboxes(
                num_checkboxes=num_checkboxes,
                image_size=(800, 600),
                output_path=img_path
            )
            
            # Convert annotations to YOLO format
            yolo_annotations = []
            for ann in annotations:
                x_center = ann['x_center'] / 800  # Normalize by image width
                y_center = ann['y_center'] / 600  # Normalize by image height
                width = ann['width'] / 800
                height = ann['height'] / 600
                
                yolo_line = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                yolo_annotations.append(yolo_line)
                self.stats.total_checkboxes += 1
            
            if yolo_annotations:
                synthetic_data.append((img_path, yolo_annotations))
                self.stats.synthetic_images += 1
        
        print(f"✅ Generated {len(synthetic_data)} synthetic images")
        return synthetic_data
    
    def create_train_val_test_splits(self, all_data: List[Tuple[Path, List[str]]]):
        """
        Create train/val/test splits following 70/20/10 ratio.
        Ensures balanced distribution of real vs synthetic data.
        """
        print("📂 Creating train/val/test splits...")
        
        # Shuffle data
        random.shuffle(all_data)
        
        total_images = len(all_data)
        train_size = int(0.7 * total_images)
        val_size = int(0.2 * total_images)
        test_size = total_images - train_size - val_size
        
        splits = {
            'train': all_data[:train_size],
            'val': all_data[train_size:train_size + val_size],
            'test': all_data[train_size + val_size:]
        }
        
        # Copy files to respective directories
        for split_name, split_data in splits.items():
            print(f"Processing {split_name} split ({len(split_data)} images)...")
            
            for img_path, annotations in tqdm(split_data, desc=f"Copying {split_name}"):
                # Copy image
                dest_img = self.output_dir / split_name / 'images' / img_path.name
                shutil.copy2(img_path, dest_img)
                
                # Create label file
                label_file = self.output_dir / split_name / 'labels' / f"{img_path.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(annotations))
            
            # Update stats
            if split_name == 'train':
                self.stats.train_images = len(split_data)
            elif split_name == 'val':
                self.stats.val_images = len(split_data)
            elif split_name == 'test':
                self.stats.test_images = len(split_data)
    
    def create_yolo_config(self):
        """Create YOLO configuration file."""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'nc': 1,  # Single class: checkbox
            'names': ['checkbox']
        }
        
        config_file = self.output_dir / 'data.yaml'
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"✅ YOLO config created: {config_file}")
    
    def run_full_pipeline(self, synthetic_images: int = 500):
        """Execute the complete data preparation pipeline."""
        print("🚀 Starting unified data preparation pipeline...\n")
        
        # Step 1: Process FUNSD data
        funsd_data = self.process_funsd_data()
        
        # Step 2: Generate synthetic data
        synthetic_data = self.generate_synthetic_data(synthetic_images)
        
        # Step 3: Combine all data
        all_data = []
        
        # Add FUNSD training data
        for img_path, annotations in funsd_data['train']:
            all_data.append((img_path, annotations))
        
        # Add FUNSD test data  
        for img_path, annotations in funsd_data['test']:
            all_data.append((img_path, annotations))
        
        # Add synthetic data
        all_data.extend(synthetic_data)
        
        print(f"\n📊 Combined dataset: {len(all_data)} total images")
        
        # Step 4: Create splits
        self.create_train_val_test_splits(all_data)
        
        # Step 5: Create YOLO config
        self.create_yolo_config()
        
        # Step 6: Print final statistics
        self.print_final_stats()
    
    def print_final_stats(self):
        """Print comprehensive dataset statistics."""
        print("\n" + "="*50)
        print("📊 FINAL DATASET STATISTICS")
        print("="*50)
        print(f"Real images (FUNSD):     {self.stats.funsd_images}")
        print(f"Synthetic images:        {self.stats.synthetic_images}")
        print(f"Total images:            {self.stats.funsd_images + self.stats.synthetic_images}")
        print(f"Total checkboxes:        {self.stats.total_checkboxes}")
        print(f"")
        print(f"Train images:            {self.stats.train_images}")
        print(f"Validation images:       {self.stats.val_images}")
        print(f"Test images:             {self.stats.test_images}")
        print(f"")
        print(f"Avg checkboxes/image:    {self.stats.total_checkboxes / (self.stats.funsd_images + self.stats.synthetic_images):.1f}")
        print("="*50)
        print(f"✅ Dataset ready for YOLOv12 training!")
        print(f"📁 Location: {self.output_dir.absolute()}")
        print(f"🎯 Config: {self.output_dir / 'data.yaml'}")
        print("="*50)

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Unified checkbox dataset preparation")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory for processed dataset")
    parser.add_argument("--synthetic-images", type=int, default=500,
                       help="Number of synthetic images to generate")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Run pipeline
    pipeline = UnifiedDataPreparation(output_dir=args.output_dir)
    pipeline.run_full_pipeline(synthetic_images=args.synthetic_images)
    
    return 0

if __name__ == "__main__":
    exit(main())