# OCR Checkbox Detection & Classification

Production-ready pipeline for detecting and classifying checkbox states in PDF documents using YOLOv8 detection and EfficientNet classification.

## Features

- 📄 **PDF Processing**: Convert PDFs to images with preprocessing (deskewing, denoising)
- 🔍 **Checkbox Detection**: YOLOv8-based detection with configurable confidence thresholds
- ✅ **State Classification**: EfficientNet-based classification (checked/unchecked/unclear)
- ⚡ **Performance Optimized**: Batch processing, GPU acceleration, <2min processing target
- 📊 **Comprehensive Output**: JSON format matching competition requirements
- 🛠️ **Fully Configurable**: YAML-based configuration for all parameters
- 🐳 **Production Ready**: Docker support and complete deployment pipeline

## Quick Start

### 1. Setup Environment

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup the project
cd '/Users/davidpepper/Dropbox/Jobstuff/Resume/Optym/AI Scientist'
git clone https://github.com/YOUR_USERNAME/OCR_checkbox.git
cd OCR_checkbox

# Install dependencies
uv sync

# Activate environment
source .venv/bin/activate
```

### 2. Test Installation

```bash
# Run comprehensive installation test
python scripts/test_installation.py

# Run quick demo
python scripts/demo.py
```

### 3. Process Your First PDF

```bash
# Create sample data
python scripts/download_data.py --sample

# Process a PDF document
python scripts/infer.py data/sample/test_form.pdf

# Process with custom settings
python scripts/infer.py your_document.pdf \
  --output results.json \
  --detection-confidence 0.5 \
  --classification-confidence 0.8 \
  --save-visualizations
```

## Project Structure

```
OCR_checkbox/
├── src/                    # Source code
│   ├── detection/          # YOLOv8 checkbox detection
│   │   └── detector.py     # Detection model and training
│   ├── classification/     # EfficientNet state classification  
│   │   └── classifier.py   # Classification model and training
│   ├── pipeline/           # End-to-end processing
│   │   └── pipeline.py     # Complete pipeline integration
│   └── utils/              # Shared utilities
│       ├── config.py       # Configuration management
│       └── pdf_processing.py # PDF and image processing
├── data/                   # Datasets and processed data
│   ├── raw/               # Raw downloaded datasets
│   ├── processed/         # Processed training data
│   └── sample/            # Sample test files
├── models/                 # Trained model weights
├── configs/                # Configuration files
│   └── default.yaml       # Default configuration
├── scripts/                # Utility scripts
│   ├── infer.py           # Main inference script
│   ├── download_data.py   # Dataset download utility
│   ├── test_installation.py # Installation verification
│   └── demo.py            # Interactive demo
├── notebooks/              # Jupyter notebooks for experimentation
└── docs/                   # Documentation
    └── approach.md         # Technical approach document
```

## Usage Examples

### Basic Inference

```bash
# Process single PDF
python scripts/infer.py document.pdf

# Process with custom output location
python scripts/infer.py document.pdf --output results/document_analysis.json

# Enable visualizations and crop saving
python scripts/infer.py document.pdf --save-visualizations --save-crops
```

### Advanced Configuration

```bash
# Use custom configuration file
python scripts/infer.py document.pdf --config configs/high_accuracy.yaml

# Override specific settings
python scripts/infer.py document.pdf \
  --detection-confidence 0.3 \
  --classification-confidence 0.9 \
  --batch-size 16 \
  --device cuda
```

### Programmatic Usage

```python
from src.pipeline.pipeline import CheckboxPipeline
from src.utils.config import load_config

# Load configuration
config = load_config("configs/default.yaml")

# Initialize pipeline
pipeline = CheckboxPipeline(config)

# Process document
results = pipeline.process_pdf(
    "document.pdf",
    output_path="results.json",
    save_visualizations=True
)

print(f"Found {results['total_checkboxes']} checkboxes")
print(f"Processing took {results['processing_time_seconds']:.2f} seconds")
```

## Configuration

The system uses YAML configuration files for all settings. Key parameters:

### Detection Settings
```yaml
detection:
  model_name: "yolov8n.pt"     # Base model size (n/s/m/l/x)
  img_size: 640                # Input resolution
  batch_size: 16               # Training batch size
  epochs: 100                  # Maximum training epochs
  learning_rate: 0.01          # Learning rate
  device: "0"                  # GPU device ("cpu" for CPU)
```

### Classification Settings
```yaml
classification:
  model_name: "efficientnet_b0" # Model architecture
  num_classes: 3                # checked/unchecked/unclear
  img_size: 128                 # Checkbox crop size
  batch_size: 32                # Batch size
  learning_rate: 0.001          # Learning rate
```

### Pipeline Settings
```yaml
pipeline:
  detection_confidence: 0.25     # Minimum detection confidence
  classification_confidence: 0.8 # Minimum classification confidence
  batch_processing: true         # Enable batch processing
  max_batch_size: 8             # Maximum batch size
```

### Data Processing
```yaml
data:
  pdf_dpi: 300                  # PDF conversion resolution
  apply_deskew: true            # Apply deskewing preprocessing
  apply_denoise: true           # Apply Gaussian denoising
  denoise_sigma: 0.5            # Denoising strength
```

## Output Format

The system outputs results in JSON format matching the competition requirements:

```json
[
  {
    "page": 1,
    "checkbox_id": "page_1_cbx_001",
    "coordinates": [100, 150, 130, 180],
    "state": "checked",
    "confidence": 0.95
  },
  {
    "page": 1, 
    "checkbox_id": "page_1_cbx_002",
    "coordinates": [100, 200, 130, 230],
    "state": "unchecked",
    "confidence": 0.87
  }
]
```

## Performance Targets

- **Processing Speed**: <2 minutes per document on RTX 4060 GPU
- **Detection mAP@0.5**: >0.85 target
- **Classification Accuracy**: >0.95 target  
- **Memory Usage**: <6GB VRAM peak
- **JSON-F1 Score**: >0.90 target

## Development

### Setup Development Environment

```bash
# Install with development dependencies
uv sync --group dev

# Install pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Format code
black src/ scripts/

# Check linting
flake8 src/ scripts/
```

### Training Custom Models

1. **Prepare Data**: Organize your training data in YOLO format
2. **Configure Training**: Edit training parameters in configs/
3. **Train Detection**: Use YOLOv8 training pipeline
4. **Train Classification**: Use EfficientNet training pipeline
5. **Evaluate Models**: Test on validation datasets

### Adding New Features

1. **Detection Models**: Modify `src/detection/detector.py`
2. **Classification Models**: Modify `src/classification/classifier.py` 
3. **Pipeline Logic**: Modify `src/pipeline/pipeline.py`
4. **Configuration**: Add parameters to `src/utils/config.py`
5. **Tests**: Add tests to verify functionality

## Datasets

Supported datasets for training and evaluation:

- **CheckboxQA**: Primary benchmark dataset
- **FUNSD**: Form understanding dataset 
- **RVL-CDIP**: Document classification dataset (forms subset)
- **Custom Synthetic**: Generated checkbox variations

```bash
# Download all datasets
python scripts/download_data.py --all

# Download specific dataset
python scripts/download_data.py --checkboxqa --funsd
```

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python scripts/infer.py document.pdf --batch-size 4

# Use CPU processing
python scripts/infer.py document.pdf --device cpu
```

**2. Low Detection Accuracy**
```yaml
# Lower detection confidence threshold
detection_confidence: 0.1

# Increase image resolution  
img_size: 1024
```

**3. Slow Processing**
```yaml
# Enable batch processing
batch_processing: true
max_batch_size: 16

# Enable quantization
enable_quantization: true
```

### Getting Help

1. **Check Logs**: Enable verbose logging in configuration
2. **Run Tests**: Use `python scripts/test_installation.py`
3. **Check Documentation**: See `docs/approach.md` for technical details
4. **GPU Issues**: Verify CUDA installation and compatibility

## Requirements

- **Python**: 3.9+ (recommended: 3.11)
- **GPU**: CUDA-compatible (recommended: RTX 4060 or better)
- **Memory**: 8GB+ VRAM for optimal performance
- **Storage**: 5GB+ for datasets and models
- **OS**: Linux, macOS, Windows (with WSL2)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Ensure all tests pass
5. Submit a pull request

## Citation

If you use this work, please cite:

```bibtex
@software{checkbox_detection_2025,
  title={OCR Checkbox Detection and Classification Pipeline},
  author={David Pepper},
  year={2025},
  url={https://github.com/YOUR_USERNAME/OCR_checkbox}
}
```
