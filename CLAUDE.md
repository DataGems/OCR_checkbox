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
```

## Architecture Overview

The pipeline follows a two-stage approach:

1. **Detection Stage** (`src/detection/`): YOLOv8 locates checkboxes in document images
2. **Classification Stage** (`src/classification/`): EfficientNet-B0 classifies checkbox states (checked/unchecked/unclear)

Key modules:
- `src/pipeline/`: End-to-end processing orchestration with batch support
- `src/utils/`: Configuration management (YAML-based) and PDF processing
- `configs/`: YAML configuration files for different components

Data flow: PDF → Image Conversion → Preprocessing → Detection → Classification → JSON Output

## Configuration

All configuration is YAML-based in `configs/`:
- `detection.yaml`: YOLOv8 detection settings
- `classification.yaml`: EfficientNet classification settings
- `pipeline.yaml`: End-to-end pipeline configuration

## Performance Requirements

When modifying code, ensure these targets are maintained:
- Processing speed: <2 minutes per document on RTX 4060
- Detection mAP@0.5: >0.85
- Classification accuracy: >0.95
- Memory usage: <6GB VRAM peak

## Development Notes

- Python 3.9+ required
- Uses UV package manager (not pip)
- GPU support via PyTorch CUDA
- Modular design - each component can be used independently
- Batch processing supported for efficiency
- Comprehensive error handling expected in all modules