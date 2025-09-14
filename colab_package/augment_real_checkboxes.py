#!/usr/bin/env python3
"""
Augment real checkbox images with various transformations.
Used to expand training data from real checkboxes extracted from documents.
"""

import os
import sys
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
from typing import List, Tuple, Dict
import json
from tqdm import tqdm

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))


class CheckboxAugmenter:
    """Augment checkbox images with document-specific transformations."""
    
    def __init__(self):
        # Define augmentation pipeline for checkboxes
        self.transform = A.Compose([
            # Geometric transformations
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.2,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
                p=0.5
            ),
            
            # Perspective and affine
            A.Perspective(
                scale=(0.02, 0.05),
                keep_size=True,
                pad_mode=cv2.BORDER_CONSTANT,
                pad_val=[255, 255, 255],
                p=0.3
            ),
            A.ElasticTransform(
                alpha=50,
                sigma=5,
                alpha_affine=10,
                border_mode=cv2.BORDER_CONSTANT,
                value=[255, 255, 255],
                p=0.3
            ),
            
            # Image quality degradations (simulate scan artifacts)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=(3, 5), p=1.0),
            ], p=0.5),
            
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            ], p=0.4),
            
            # Brightness and contrast (simulate different scan settings)
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=1.0
                ),
                A.RandomGamma(gamma_limit=(80, 120), p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            ], p=0.5),
            
            # Simulate document artifacts
            A.OneOf([
                # Simulate shadow/uneven lighting
                A.RandomShadow(
                    shadow_roi=(0, 0, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    shadow_dimension=3,
                    p=1.0
                ),
                # Simulate fold or crease
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.3,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=[255, 255, 255],
                    p=1.0
                ),
            ], p=0.3),
            
            # Simulate ink/print variations
            A.OneOf([
                # Lighter ink (faded)
                A.ToGray(p=1.0),
                # Darker/heavier ink
                A.Equalize(p=1.0),
            ], p=0.2),
            
            # Compression artifacts
            A.ImageCompression(
                quality_lower=60,
                quality_upper=100,
                p=0.3
            ),
        ])
        
        # Specific augmentations for checkbox states
        self.state_specific_augmentations = {
            "checked": self._get_checked_augmentations(),
            "unchecked": self._get_unchecked_augmentations(),
            "unclear": self._get_unclear_augmentations()
        }
    
    def _get_checked_augmentations(self) -> A.Compose:
        """Augmentations specific to checked checkboxes."""
        return A.Compose([
            # Simulate incomplete or partial check marks
            A.CoarseDropout(
                max_holes=3,
                max_height=5,
                max_width=5,
                min_holes=1,
                min_height=2,
                min_width=2,
                fill_value=255,
                p=0.3
            ),
        ])
    
    def _get_unchecked_augmentations(self) -> A.Compose:
        """Augmentations specific to unchecked checkboxes."""
        return A.Compose([
            # Add random small marks that shouldn't be confused with checks
            A.OneOf([
                # Small dots
                A.CoarseDropout(
                    max_holes=2,
                    max_height=2,
                    max_width=2,
                    min_holes=1,
                    min_height=1,
                    min_width=1,
                    fill_value=0,
                    p=1.0
                ),
            ], p=0.2),
        ])
    
    def _get_unclear_augmentations(self) -> A.Compose:
        """Augmentations specific to unclear checkboxes."""
        return A.Compose([
            # Make more unclear with additional noise/blur
            A.OneOf([
                A.GaussianBlur(blur_limit=(5, 9), p=1.0),
                A.GaussNoise(var_limit=(50.0, 100.0), p=1.0),
                # Simulate smudging
                A.MotionBlur(blur_limit=(7, 15), p=1.0),
            ], p=0.5),
        ])
    
    def augment_checkbox(self, image: np.ndarray, state: str = None, 
                        num_augmentations: int = 5) -> List[np.ndarray]:
        """
        Generate augmented versions of a checkbox image.
        
        Args:
            image: Original checkbox image
            state: Checkbox state (checked/unchecked/unclear)
            num_augmentations: Number of augmented versions to generate
            
        Returns:
            List of augmented images
        """
        augmented_images = []
        
        for i in range(num_augmentations):
            # Apply general augmentations
            augmented = self.transform(image=image)['image']
            
            # Apply state-specific augmentations if state is known
            if state and state in self.state_specific_augmentations:
                augmented = self.state_specific_augmentations[state](image=augmented)['image']
            
            augmented_images.append(augmented)
        
        return augmented_images
    
    def augment_dataset(self, input_dir: str, output_dir: str, 
                       augmentations_per_image: int = 5):
        """
        Augment an entire dataset of checkbox images.
        
        Args:
            input_dir: Directory containing original checkbox images
            output_dir: Directory to save augmented images
            augmentations_per_image: Number of augmentations per original image
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['checked', 'unchecked', 'unclear']:
            (output_path / subdir).mkdir(exist_ok=True)
        
        # Process images
        image_files = list(input_path.rglob("*.png")) + list(input_path.rglob("*.jpg"))
        
        print(f"Found {len(image_files)} images to augment")
        
        augmentation_stats = {"checked": 0, "unchecked": 0, "unclear": 0}
        
        for img_path in tqdm(image_files, desc="Augmenting images"):
            # Load image
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            
            # Determine state from path or filename
            state = None
            for s in ['checked', 'unchecked', 'unclear']:
                if s in str(img_path).lower():
                    state = s
                    break
            
            # Generate augmentations
            augmented_images = self.augment_checkbox(img, state, augmentations_per_image)
            
            # Save augmented images
            base_name = img_path.stem
            save_dir = output_path / state if state else output_path
            
            for i, aug_img in enumerate(augmented_images):
                save_path = save_dir / f"{base_name}_aug_{i}.png"
                cv2.imwrite(str(save_path), aug_img)
                
                if state:
                    augmentation_stats[state] += 1
        
        # Save augmentation statistics
        stats_path = output_path / "augmentation_stats.json"
        with open(stats_path, 'w') as f:
            json.dump({
                "total_original_images": len(image_files),
                "augmentations_per_image": augmentations_per_image,
                "total_augmented_images": sum(augmentation_stats.values()),
                "state_distribution": augmentation_stats
            }, f, indent=2)
        
        print(f"\nAugmentation complete!")
        print(f"Statistics saved to: {stats_path}")
        print(f"State distribution: {augmentation_stats}")


def create_mixed_training_set(real_data_dir: str, synthetic_data_dir: str,
                            output_dir: str, synthetic_ratio: float = 0.5):
    """
    Create a mixed training set with real and synthetic data.
    
    Args:
        real_data_dir: Directory with real checkbox data
        synthetic_data_dir: Directory with synthetic checkbox data
        output_dir: Output directory for mixed dataset
        synthetic_ratio: Maximum ratio of synthetic data (default 0.5 = 50%)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create train/val/test splits
    for split in ['train', 'val', 'test']:
        (output_path / split / 'images').mkdir(parents=True, exist_ok=True)
        (output_path / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Count real data
    real_images = list(Path(real_data_dir).rglob("*.png")) + \
                 list(Path(real_data_dir).rglob("*.jpg"))
    num_real = len(real_images)
    
    # Calculate synthetic data limit
    max_synthetic = int(num_real * synthetic_ratio)
    
    print(f"Real images: {num_real}")
    print(f"Maximum synthetic images: {max_synthetic}")
    
    # Copy real data (with train/val/test split)
    # Assuming 70% train, 15% val, 15% test
    np.random.shuffle(real_images)
    
    train_split = int(0.7 * num_real)
    val_split = int(0.85 * num_real)
    
    splits = {
        'train': real_images[:train_split],
        'val': real_images[train_split:val_split],
        'test': real_images[val_split:]
    }
    
    # Copy real images to splits
    for split_name, images in splits.items():
        for img_path in images:
            # Copy image
            dest_path = output_path / split_name / 'images' / img_path.name
            shutil.copy2(img_path, dest_path)
            
            # Copy corresponding label if exists
            label_path = img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
            if label_path.exists():
                dest_label = output_path / split_name / 'labels' / f"{img_path.stem}.txt"
                shutil.copy2(label_path, dest_label)
    
    # Add synthetic data (only to training set)
    synthetic_images = list(Path(synthetic_data_dir).rglob("*.png"))[:max_synthetic]
    
    for img_path in synthetic_images:
        # Copy to train only
        dest_path = output_path / 'train' / 'images' / f"synthetic_{img_path.name}"
        shutil.copy2(img_path, dest_path)
        
        # Copy label
        label_path = img_path.parent.parent / 'labels' / f"{img_path.stem}.txt"
        if label_path.exists():
            dest_label = output_path / 'train' / 'labels' / f"synthetic_{img_path.stem}.txt"
            shutil.copy2(label_path, dest_label)
    
    print(f"\nMixed dataset created in: {output_path}")
    print(f"Train: {len(list((output_path / 'train' / 'images').glob('*')))} images")
    print(f"Val: {len(list((output_path / 'val' / 'images').glob('*')))} images")
    print(f"Test: {len(list((output_path / 'test' / 'images').glob('*')))} images")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Augment checkbox images")
    parser.add_argument("--input-dir", type=str, required=True,
                       help="Input directory with checkbox images")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for augmented images")
    parser.add_argument("--augmentations", type=int, default=5,
                       help="Number of augmentations per image")
    
    args = parser.parse_args()
    
    augmenter = CheckboxAugmenter()
    augmenter.augment_dataset(args.input_dir, args.output_dir, args.augmentations)


if __name__ == "__main__":
    main()