#!/usr/bin/env python3
"""
Extract potential checkbox annotations from FUNSD dataset.
Uses heuristics to identify checkbox-like elements.
"""

import json
import shutil
from pathlib import Path
from typing import List, Tuple

def is_potential_checkbox(box: List[int], text: str) -> bool:
    """
    Determine if a bounding box might be a checkbox based on:
    - Size (small, roughly square)
    - Text content (empty, single char like X, checkmark symbols)
    - Aspect ratio
    """
    if len(box) != 4:
        return False
    
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    
    # Size constraints (checkboxes are typically 10-40 pixels)
    if width < 5 or width > 50 or height < 5 or height > 50:
        return False
    
    # Aspect ratio (should be roughly square)
    if height > 0:
        aspect_ratio = width / height
        if aspect_ratio < 0.7 or aspect_ratio > 1.3:
            return False
    
    # Text content checks
    text = text.strip().lower()
    checkbox_texts = ['', 'x', '✓', '✔', '☑', '☐', '□', '■', 'yes', 'no', 'y', 'n']
    
    # Small square with no text or checkbox-like text
    if text in checkbox_texts:
        return True
    
    # Very small elements are likely checkboxes
    if width < 20 and height < 20 and len(text) <= 2:
        return True
    
    return False

def convert_to_yolo_format(box: List[int], img_width: int, img_height: int) -> str:
    """Convert bounding box to YOLO format."""
    x1, y1, x2, y2 = box
    
    # Calculate center coordinates
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # Class 0 for all checkboxes (we'll classify state later)
    return f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"

def process_funsd_annotations(input_dir: Path, output_dir: Path):
    """Extract checkbox annotations from FUNSD dataset."""
    
    # Create output directories
    for split in ['train', 'val']:
        (output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    stats = {'train': 0, 'val': 0, 'checkboxes': 0}
    
    # Process training data
    train_ann_dir = input_dir / "dataset" / "training_data" / "annotations"
    train_img_dir = input_dir / "dataset" / "training_data" / "images"
    
    if train_ann_dir.exists():
        for ann_file in train_ann_dir.glob("*.json"):
            img_file = train_img_dir / f"{ann_file.stem}.png"
            
            if not img_file.exists():
                continue
            
            # Load annotation
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # Extract potential checkboxes
            checkboxes = []
            if 'form' in data:
                for item in data['form']:
                    box = item.get('box', [])
                    text = item.get('text', '')
                    
                    if is_potential_checkbox(box, text):
                        # Get actual image dimensions
                        import cv2
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            img_height, img_width = img.shape[:2]
                        else:
                            img_width, img_height = 1000, 1000  # fallback
                        yolo_line = convert_to_yolo_format(box, img_width, img_height)
                        checkboxes.append(yolo_line)
                        stats['checkboxes'] += 1
            
            # If we found checkboxes, copy image and save labels
            if checkboxes:
                # Copy image
                shutil.copy(img_file, output_dir / 'train' / 'images' / img_file.name)
                
                # Save labels
                label_file = output_dir / 'train' / 'labels' / f"{img_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(checkboxes))
                
                stats['train'] += 1
    
    # Process test data as validation
    test_ann_dir = input_dir / "dataset" / "testing_data" / "annotations"
    test_img_dir = input_dir / "dataset" / "testing_data" / "images"
    
    if test_ann_dir.exists():
        for ann_file in test_ann_dir.glob("*.json"):
            img_file = test_img_dir / f"{ann_file.stem}.png"
            
            if not img_file.exists():
                continue
            
            # Load annotation
            with open(ann_file, 'r') as f:
                data = json.load(f)
            
            # Extract potential checkboxes
            checkboxes = []
            if 'form' in data:
                for item in data['form']:
                    box = item.get('box', [])
                    text = item.get('text', '')
                    
                    if is_potential_checkbox(box, text):
                        # Get actual image dimensions
                        import cv2
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            img_height, img_width = img.shape[:2]
                        else:
                            img_width, img_height = 1000, 1000  # fallback
                        yolo_line = convert_to_yolo_format(box, img_width, img_height)
                        checkboxes.append(yolo_line)
                        stats['checkboxes'] += 1
            
            # If we found checkboxes, copy image and save labels
            if checkboxes:
                # Copy image
                shutil.copy(img_file, output_dir / 'val' / 'images' / img_file.name)
                
                # Save labels
                label_file = output_dir / 'val' / 'labels' / f"{img_file.stem}.txt"
                with open(label_file, 'w') as f:
                    f.write('\n'.join(checkboxes))
                
                stats['val'] += 1
    
    # Create data.yaml
    data_yaml = {
        'path': str(output_dir.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'nc': 1,  # Single class for checkbox detection
        'names': ['checkbox']
    }
    
    import yaml
    with open(output_dir / 'data.yaml', 'w') as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
    
    print(f"✅ Processed FUNSD dataset")
    print(f"📊 Training images with checkboxes: {stats['train']}")
    print(f"📊 Validation images with checkboxes: {stats['val']}")
    print(f"📊 Total checkboxes found: {stats['checkboxes']}")
    print(f"📁 data.yaml created at: {output_dir / 'data.yaml'}")
    
    return stats

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract checkbox labels from FUNSD")
    parser.add_argument("--input-dir", type=str, default="data/raw/funsd",
                       help="FUNSD dataset directory")
    parser.add_argument("--output-dir", type=str, default="data/processed_funsd",
                       help="Output directory for processed data")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return 1
    
    stats = process_funsd_annotations(input_dir, output_dir)
    
    if stats['checkboxes'] == 0:
        print("⚠️ No checkboxes found. You may need to:")
        print("1. Adjust the heuristics in is_potential_checkbox()")
        print("2. Use synthetic data generation")
        print("3. Download and annotate CheckboxQA PDFs")
    
    return 0

if __name__ == "__main__":
    exit(main())