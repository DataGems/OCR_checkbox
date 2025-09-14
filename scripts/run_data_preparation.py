#!/usr/bin/env python3
"""
Run the complete data preparation pipeline.
Orchestrates PDF processing, augmentation, and dataset creation.
"""

import os
import sys
import subprocess
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_command(cmd: str, description: str) -> bool:
    """Run shell command and log results."""
    logger.info(f"Starting: {description}")
    logger.info(f"Command: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            check=True
        )
        logger.info(f"Success: {description}")
        if result.stdout.strip():
            logger.info(f"Output: {result.stdout}")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        logger.error(f"Error: {e.stderr}")
        return False


def main():
    """Execute complete data preparation pipeline."""
    logger.info("Starting data preparation pipeline...")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    # Step 1: Process FUNSD dataset (available)
    logger.info("=" * 50)
    logger.info("Step 1: Processing FUNSD dataset")
    cmd = "python scripts/prepare_datasets.py --process-funsd --augment --augmentation-factor 3"
    if not run_command(cmd, "Process FUNSD dataset"):
        logger.error("Failed to process FUNSD dataset")
        return 1
    
    # Step 2: Process CheckboxQA metadata
    logger.info("=" * 50)
    logger.info("Step 2: Processing CheckboxQA metadata")
    cmd = "python scripts/prepare_datasets.py --process-checkboxqa"
    if not run_command(cmd, "Process CheckboxQA metadata"):
        logger.warning("CheckboxQA processing failed - continuing with available data")
    
    # Step 3: Generate synthetic data
    logger.info("=" * 50)
    logger.info("Step 3: Generating synthetic checkbox data")
    cmd = "python scripts/generate_synthetic_checkboxes.py --num-forms 200 --output-dir data/synthetic"
    if not run_command(cmd, "Generate synthetic checkboxes"):
        logger.warning("Synthetic generation failed - continuing with real data only")
    
    # Step 4: Create mixed training set
    logger.info("=" * 50)
    logger.info("Step 4: Creating final training datasets")
    
    # Check if we have data to work with
    data_dir = Path("data/processed")
    if not data_dir.exists() or not list(data_dir.glob("*/images/*.png")):
        logger.error("No processed data found. Check previous steps.")
        return 1
    
    # Verify outputs
    train_images = list((data_dir / "train" / "images").glob("*.png"))
    val_images = list((data_dir / "val" / "images").glob("*.png"))
    test_images = list((data_dir / "test" / "images").glob("*.png"))
    
    logger.info(f"Dataset created successfully:")
    logger.info(f"  Training images: {len(train_images)}")
    logger.info(f"  Validation images: {len(val_images)}")
    logger.info(f"  Test images: {len(test_images)}")
    
    # Check for YOLO config
    yaml_config = data_dir / "data.yaml"
    if yaml_config.exists():
        logger.info(f"  YOLO config: {yaml_config}")
    else:
        logger.warning("YOLO config file not found")
    
    logger.info("=" * 50)
    logger.info("Data preparation pipeline completed successfully!")
    logger.info(f"Processed data available in: {data_dir}")
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)