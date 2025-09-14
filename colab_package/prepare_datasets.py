#!/usr/bin/env python3
"""
Data preparation pipeline for checkbox detection.
Processes PDFs, applies preprocessing, and creates training datasets.
"""

import os
import sys
import cv2
import numpy as np
import json
import shutil
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
import pdf2image
from PIL import Image
import logging
import yaml

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.pdf_processing import PDFProcessor


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoundingBox:
    """Bounding box for checkbox annotation."""
    x: float
    y: float
    width: float
    height: float
    label: str  # checked, unchecked, unclear
    confidence: float = 1.0
    
    def to_yolo(self, img_width: int, img_height: int) -> str:
        """Convert to YOLO format (normalized center coordinates)."""
        # Class mapping
        class_map = {"unchecked": 0, "checked": 1, "unclear": 2}
        class_id = class_map.get(self.label, 0)
        
        # Calculate center coordinates and normalize
        x_center = (self.x + self.width / 2) / img_width
        y_center = (self.y + self.height / 2) / img_height
        w_norm = self.width / img_width
        h_norm = self.height / img_height
        
        return f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}"


class DatasetProcessor:
    """Process and prepare datasets for training."""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.pdf_processor = PDFProcessor()
        
        # Create directory structure
        self._create_directory_structure()
        
        # Dataset statistics
        self.stats = {
            "total_images": 0,
            "total_checkboxes": 0,
            "checkbox_states": {"checked": 0, "unchecked": 0, "unclear": 0},
            "processing_errors": 0
        }
    
    def _create_directory_structure(self):
        """Create output directory structure."""
        splits = ["train", "val", "test"]
        subdirs = ["images", "labels", "crops/checked", "crops/unchecked", "crops/unclear"]
        
        for split in splits:
            for subdir in subdirs:
                (self.output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    
    def process_checkboxqa(self, data_dir: Path) -> Dict:
        """
        Process CheckboxQA dataset.
        
        Args:
            data_dir: Path to CheckboxQA data directory
            
        Returns:
            Processing statistics
        """
        logger.info("Processing CheckboxQA dataset...")
        
        # Load annotations
        gold_file = data_dir / "data" / "gold.jsonl"
        doc_map_file = data_dir / "data" / "document_url_map.json"
        
        if not gold_file.exists():
            logger.error(f"Gold annotations not found: {gold_file}")
            return {}
        
        # Load document URLs
        doc_urls = {}
        if doc_map_file.exists():
            with open(doc_map_file, 'r') as f:
                doc_urls = json.load(f)
        
        # Process annotations
        annotations = []
        with open(gold_file, 'r') as f:
            for line in f:
                if line.strip():
                    annotations.append(json.loads(line))
        
        logger.info(f"Found {len(annotations)} annotations")
        
        # Download PDFs script exists - we'll note this for the user
        download_script = data_dir / "dowload_documents.py"  # Note: typo in original
        if download_script.exists():
            logger.info(f"PDF download script available at: {download_script}")
            logger.info("Run this script first to download the actual PDF documents")
        
        # Process available images/PDFs
        # For now, we'll prepare the structure for when PDFs are downloaded
        processed_count = 0
        
        # Create splits (70% train, 15% val, 15% test)
        np.random.shuffle(annotations)
        train_split = int(0.7 * len(annotations))
        val_split = int(0.85 * len(annotations))
        
        splits = {
            "train": annotations[:train_split],
            "val": annotations[train_split:val_split],
            "test": annotations[val_split:]
        }
        
        # Save split information
        for split_name, split_data in splits.items():
            split_file = self.output_dir / split_name / f"checkboxqa_{split_name}.json"
            with open(split_file, 'w') as f:
                json.dump(split_data, f, indent=2)
        
        logger.info(f"Created splits - Train: {len(splits['train'])}, "
                   f"Val: {len(splits['val'])}, Test: {len(splits['test'])}")
        
        return {
            "dataset": "CheckboxQA",
            "total_annotations": len(annotations),
            "splits": {k: len(v) for k, v in splits.items()},
            "note": "Run download_documents.py to get actual PDFs"
        }
    
    def process_funsd(self, data_dir: Path) -> Dict:
        """
        Process FUNSD dataset for checkbox detection.
        
        Args:
            data_dir: Path to FUNSD data directory
            
        Returns:
            Processing statistics
        """
        logger.info("Processing FUNSD dataset...")
        
        funsd_path = data_dir / "dataset"
        if not funsd_path.exists():
            logger.error(f"FUNSD dataset not found at: {funsd_path}")
            return {}
        
        processed_stats = {
            "train": {"images": 0, "checkboxes": 0},
            "test": {"images": 0, "checkboxes": 0}
        }
        
        # Process training and test data
        for split in ["training_data", "testing_data"]:
            split_path = funsd_path / split
            if not split_path.exists():
                continue
            
            images_dir = split_path / "images"
            annotations_dir = split_path / "annotations"
            
            # Determine output split
            output_split = "train" if split == "training_data" else "test"
            
            # Process each image
            for img_file in tqdm(images_dir.glob("*.png"), desc=f"Processing {split}"):
                # Load annotation
                ann_file = annotations_dir / f"{img_file.stem}.json"
                if not ann_file.exists():
                    continue
                
                with open(ann_file, 'r') as f:
                    annotation = json.load(f)
                
                # Load and preprocess image
                img = cv2.imread(str(img_file))
                if img is None:
                    continue
                
                # Apply preprocessing
                img = self.pdf_processor.deskew_image(img)
                img = self.pdf_processor.denoise_image(img)
                
                # Extract checkbox annotations
                checkboxes = self._extract_checkboxes_from_funsd(annotation, img.shape)
                
                if checkboxes:
                    # Save processed image
                    output_img = self.output_dir / output_split / "images" / img_file.name
                    cv2.imwrite(str(output_img), img)
                    
                    # Save YOLO format labels
                    label_file = self.output_dir / output_split / "labels" / f"{img_file.stem}.txt"
                    with open(label_file, 'w') as f:
                        for bbox in checkboxes:
                            f.write(bbox.to_yolo(img.shape[1], img.shape[0]) + "\n")
                    
                    # Extract and save checkbox crops
                    self._extract_checkbox_crops(img, checkboxes, img_file.stem, output_split)
                    
                    # Update statistics
                    processed_stats[output_split]["images"] += 1
                    processed_stats[output_split]["checkboxes"] += len(checkboxes)
                    
                    for bbox in checkboxes:
                        self.stats["checkbox_states"][bbox.label] += 1
        
        logger.info(f"FUNSD processing complete: {processed_stats}")
        return processed_stats
    
    def _extract_checkboxes_from_funsd(self, annotation: Dict, 
                                      img_shape: Tuple[int, int, int]) -> List[BoundingBox]:
        """
        Extract checkbox annotations from FUNSD format.
        
        FUNSD doesn't have explicit checkbox annotations, so we'll look for:
        - Small square regions
        - Form elements that might be checkboxes
        """
        checkboxes = []
        
        # Look through form elements
        if "form" in annotation:
            for item in annotation["form"]:
                box = item.get("box", [])
                if len(box) == 4:
                    x, y, x2, y2 = box
                    width = x2 - x
                    height = y2 - y
                    
                    # Heuristic: small square regions might be checkboxes
                    aspect_ratio = width / height if height > 0 else 0
                    if 0.8 < aspect_ratio < 1.2 and 10 < width < 50:
                        # Try to determine state from text
                        text = item.get("text", "").lower()
                        if any(mark in text for mark in ["x", "✓", "✔", "☑"]):
                            label = "checked"
                        elif text.strip() == "":
                            label = "unchecked"
                        else:
                            label = "unclear"
                        
                        checkboxes.append(BoundingBox(
                            x=x, y=y, width=width, height=height, label=label
                        ))
        
        return checkboxes
    
    def _extract_checkbox_crops(self, img: np.ndarray, checkboxes: List[BoundingBox],
                               img_id: str, split: str):
        """Extract and save individual checkbox crops."""
        for i, bbox in enumerate(checkboxes):
            # Extract crop with padding
            padding = 5
            x1 = max(0, int(bbox.x - padding))
            y1 = max(0, int(bbox.y - padding))
            x2 = min(img.shape[1], int(bbox.x + bbox.width + padding))
            y2 = min(img.shape[0], int(bbox.y + bbox.height + padding))
            
            crop = img[y1:y2, x1:x2]
            
            # Save crop
            crop_path = (self.output_dir / split / "crops" / bbox.label / 
                        f"{img_id}_checkbox_{i}.png")
            cv2.imwrite(str(crop_path), crop)
    
    def create_augmented_dataset(self, augmentation_factor: int = 3):
        """
        Apply augmentations to training data.
        
        Args:
            augmentation_factor: Number of augmented versions per image
        """
        logger.info(f"Creating augmented dataset with factor {augmentation_factor}...")
        
        train_images = self.output_dir / "train" / "images"
        aug_dir = self.output_dir / "train_augmented"
        aug_dir.mkdir(exist_ok=True)
        
        # Import augmentation pipeline
        from scripts.augment_real_checkboxes import CheckboxAugmenter
        augmenter = CheckboxAugmenter()
        
        # Process each training image
        for img_file in tqdm(train_images.glob("*.png"), desc="Augmenting"):
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Generate augmented versions
            aug_images = augmenter.augment_checkbox(img, num_augmentations=augmentation_factor)
            
            # Save augmented images and labels
            label_file = self.output_dir / "train" / "labels" / f"{img_file.stem}.txt"
            if label_file.exists():
                with open(label_file, 'r') as f:
                    labels = f.read()
                
                for i, aug_img in enumerate(aug_images):
                    # Save augmented image
                    aug_img_path = aug_dir / "images" / f"{img_file.stem}_aug_{i}.png"
                    aug_img_path.parent.mkdir(exist_ok=True)
                    cv2.imwrite(str(aug_img_path), aug_img)
                    
                    # Copy labels
                    aug_label_path = aug_dir / "labels" / f"{img_file.stem}_aug_{i}.txt"
                    aug_label_path.parent.mkdir(exist_ok=True)
                    with open(aug_label_path, 'w') as f:
                        f.write(labels)
    
    def prepare_yolo_dataset(self):
        """
        Prepare dataset in YOLO format with proper configuration.
        """
        logger.info("Preparing YOLO format dataset...")
        
        # Create data.yaml for YOLO training
        data_yaml = {
            "path": str(self.output_dir.absolute()),
            "train": "train/images",
            "val": "val/images",
            "test": "test/images",
            "nc": 3,  # number of classes
            "names": ["unchecked", "checked", "unclear"]
        }
        
        yaml_path = self.output_dir / "data.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(data_yaml, f, default_flow_style=False)
        
        logger.info(f"YOLO configuration saved to: {yaml_path}")
        
        # Create dataset statistics
        stats_file = self.output_dir / "dataset_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Dataset statistics saved to: {stats_file}")
    
    def process_pdf_directory(self, pdf_dir: Path, output_split: str = "train"):
        """
        Process a directory of PDF files.
        
        Args:
            pdf_dir: Directory containing PDF files
            output_split: Which split to save to (train/val/test)
        """
        logger.info(f"Processing PDFs from: {pdf_dir}")
        
        pdf_files = list(pdf_dir.glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
            try:
                # Convert PDF to images
                images = pdf2image.convert_from_path(
                    pdf_file,
                    dpi=300,  # 300 DPI as specified
                    fmt='PNG',
                    thread_count=4
                )
                
                # Process each page
                for page_num, pil_image in enumerate(images):
                    # Convert to OpenCV format
                    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
                    
                    # Apply preprocessing
                    img = self.pdf_processor.deskew_image(img)
                    img = self.pdf_processor.denoise_image(img)
                    
                    # Save processed image
                    img_name = f"{pdf_file.stem}_page_{page_num:03d}.png"
                    output_path = self.output_dir / output_split / "images" / img_name
                    cv2.imwrite(str(output_path), img)
                    
                    self.stats["total_images"] += 1
                    
            except Exception as e:
                logger.error(f"Error processing {pdf_file}: {str(e)}")
                self.stats["processing_errors"] += 1


def main():
    """Main data preparation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare datasets for checkbox detection")
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Base data directory")
    parser.add_argument("--output-dir", type=str, default="data/processed",
                       help="Output directory for processed data")
    parser.add_argument("--process-checkboxqa", action="store_true",
                       help="Process CheckboxQA dataset")
    parser.add_argument("--process-funsd", action="store_true",
                       help="Process FUNSD dataset")
    parser.add_argument("--process-pdfs", type=str,
                       help="Process PDFs from specified directory")
    parser.add_argument("--augment", action="store_true",
                       help="Create augmented training data")
    parser.add_argument("--augmentation-factor", type=int, default=3,
                       help="Number of augmentations per image")
    
    args = parser.parse_args()
    
    # Initialize processor
    processor = DatasetProcessor(args.output_dir)
    
    data_path = Path(args.data_dir)
    
    # Process datasets
    if args.process_checkboxqa:
        checkboxqa_path = data_path / "raw" / "CheckboxQA"
        if checkboxqa_path.exists():
            processor.process_checkboxqa(checkboxqa_path)
        else:
            logger.error(f"CheckboxQA not found at: {checkboxqa_path}")
    
    if args.process_funsd:
        funsd_path = data_path / "raw" / "funsd"
        if funsd_path.exists():
            processor.process_funsd(funsd_path)
        else:
            logger.error(f"FUNSD not found at: {funsd_path}")
    
    if args.process_pdfs:
        pdf_path = Path(args.process_pdfs)
        if pdf_path.exists():
            processor.process_pdf_directory(pdf_path)
        else:
            logger.error(f"PDF directory not found: {pdf_path}")
    
    # Apply augmentations
    if args.augment:
        processor.create_augmented_dataset(args.augmentation_factor)
    
    # Prepare YOLO format
    processor.prepare_yolo_dataset()
    
    logger.info("Data preparation complete!")
    logger.info(f"Processed data saved to: {processor.output_dir}")


if __name__ == "__main__":
    main()