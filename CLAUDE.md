# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computer vision pipeline for detecting and classifying checkboxes in PDF documents using YOLOv8 (detection) and EfficientNet-B0 (classification). The project addresses limitations in current Large Vision Language Models (LVLMs) which have 30-50% misclassification rates on checkbox detection tasks.

### Key Goals
- Detect checkboxes in PDF documents with mAP@0.5 > 0.85
- Classify checkbox states (checked/unchecked/unclear) with >95% accuracy
- Process documents in <2 minutes on RTX 4060 GPU
- Achieve JSON-F1 score > 0.90 for end-to-end performance

## Common Development Commands

### Setup and Dependencies
```bash
./run.sh setup           # Initial setup (install deps, run tests)
./run.sh dev-setup       # Setup with dev dependencies
uv sync                  # Install production dependencies
uv sync --group dev      # Install with development dependencies
```

### Code Quality
```bash
./run.sh format          # Format code with black (line-length: 88)
./run.sh lint            # Run flake8 linting
black src/ scripts/      # Direct formatting
flake8 src/ scripts/     # Direct linting
```

### Testing
```bash
./run.sh test            # Run installation test
pytest tests/            # Run unit tests
python scripts/test_installation.py  # Direct test command
```

### Running the Pipeline
```bash
./run.sh demo                    # Interactive demo
./run.sh infer <pdf_path>        # Run inference on a PDF
python scripts/infer.py <pdf>    # Direct inference
```

### Data Management
```bash
./run.sh download-sample    # Download sample data
./run.sh download-all       # Download all datasets
./run.sh clean              # Clean generated files
python scripts/run_data_preparation.py  # Complete data prep pipeline
python scripts/prepare_datasets.py --help  # Manual dataset processing
python scripts/generate_synthetic_checkboxes.py --help  # Generate synthetic data
```

## Architecture Overview

The pipeline follows a two-stage approach:

1. **Detection Stage** (`src/detection/`): YOLOv12 locates checkboxes in document images using attention mechanisms for improved small object detection
2. **Classification Stage** (`src/classification/`): EfficientNet-B0 classifies checkbox states (checked/unchecked/unclear)

Key modules:
- `src/pipeline/`: End-to-end processing orchestration with batch support
- `src/utils/`: Configuration management (YAML-based) and PDF processing
- `configs/`: YAML configuration files for different components

Data flow: PDF → Image Conversion → Preprocessing → Detection → Classification → JSON Output

### Training Commands
```bash
python scripts/train_yolo12_detection.py --data data/processed/data.yaml --model yolo12n.pt
python scripts/train_yolo12_detection.py --data data/processed/data.yaml --model yolo12s.pt --export
```

## Configuration

All configuration is YAML-based in `configs/`:
- `default.yaml`: Main configuration with YOLOv12 detection settings
- YOLOv12 variants: nano (n), small (s), medium (m), large (l), extra-large (x)
- EfficientNet-B0 classification settings

## Performance Requirements

When modifying code, ensure these targets are maintained:
- Processing speed: <2 minutes per document on RTX 4060
- Detection mAP@0.5: >0.85
- Classification accuracy: >0.95
- Memory usage: <6GB VRAM peak

### Detection Evaluation Metrics

**IoU (Intersection over Union)**: Standard metric for bounding box evaluation
- IoU = Area of Overlap / Area of Union
- IoU ≥ 0.5: Standard threshold - predicted box must overlap ≥50% with ground truth
- IoU ≥ 0.75: Stricter threshold for high precision applications
- IoU ≥ 0.3: Lenient threshold, sometimes used for very small objects

**YOLOv12 Metrics**:
- **mAP@0.5**: Mean Average Precision at IoU=0.5 threshold (project target: >0.85)
- **mAP@0.5:0.95**: Mean across IoU thresholds from 0.5 to 0.95 (comprehensive measure)
- **Precision/Recall**: At different confidence thresholds

**Why IoU=0.5 for Checkboxes**: Balances accuracy requirements with practical detection needs for small (20-40 pixel) objects. Allows reasonable positioning/sizing variation while ensuring meaningful overlap with ground truth annotations.

## Development Notes

- Python 3.9+ required
- Uses UV package manager (not pip)
- GPU support via PyTorch CUDA
- Modular design - each component can be used independently
- Batch processing supported for efficiency
- Comprehensive error handling expected in all modules

## Training Framework Alternatives

### Current Setup (Recommended)
- **Framework**: Ultralytics (official YOLOv12 implementation)
- **Augmentation**: YOLO built-in (hsv_h, degrees, scale, mosaic)
- **Optimizer**: AdamW (default for attention models)
- **Monitoring**: TensorBoard (automatic with Ultralytics)
- **Loss**: YOLO default (box + class + distribution focal loss)

### Alternative Options (If Current Setup Insufficient)

#### **Model Size Alternatives**:
```bash
# If YOLOv12n doesn't hit mAP@0.5 > 0.85
python scripts/train_yolo12_detection.py --model yolo12s.pt  # 9.3M params, more accurate
python scripts/train_yolo12_detection.py --model yolo12m.pt  # 20.2M params, highest accuracy
```

#### **Augmentation Alternatives**:
```python
# More aggressive Albumentations pipeline
import albumentations as A
transform = A.Compose([
    A.Rotate(limit=10, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
    A.OpticalDistortion(distort_limit=0.05, shift_limit=0.05, p=0.3)
])
```

#### **Optimizer Alternatives**:
```python
# In training config - switch optimizer if convergence issues
optimizer: 'SGD'          # Sometimes better for YOLO
optimizer: 'Adam'         # Faster convergence
optimizer: 'AdamW'        # Better for attention models (current)
```

#### **Monitoring Alternatives**:
```python
# Weights & Biases for better experiment tracking
import wandb
wandb.init(project="checkbox-detection")
# Add to training script if need collaborative monitoring
```

#### **Loss Function Alternatives**:
```python
# Custom loss weights if class imbalance issues
box: 7.5        # Bounding box loss weight (current)  
cls: 0.5        # Classification loss weight (current)
dfl: 1.5        # Distribution focal loss weight (current)
# Adjust ratios if detection vs classification performance imbalanced
```

#### **Training Strategy Alternatives**:
```python
# If standard training fails
- Increase synthetic data ratio (max 50%)
- Add focal loss for hard examples  
- Use progressive resizing (start 320→640)
- Implement curriculum learning (easy→hard samples)
- Try ensemble of multiple model sizes
```