# Colab Classification Training

Run this in Google Colab to train the EfficientNet classification model.

## Setup
```python
# 1. Install dependencies
!pip install ultralytics torch torchvision opencv-python matplotlib seaborn scikit-learn tqdm

# 2. Upload files to Colab
# Upload: best.pt (your detection model)
# Upload: generate_classification_dataset.py
# Upload: train_classification.py

# 3. Create directories
!mkdir -p models data/classification

# 4. Move your detection model
!mv best.pt models/
```

## Generate Dataset
```python
# Run dataset generation
exec(open('generate_classification_dataset.py').read())
```

## Train Classifier
```python
# Run classification training  
exec(open('train_classification.py').read())
```

## Quick Test
```python
# Test the trained classifier
from train_classification import EfficientNetClassifier
import torch

# Load trained model
classifier = EfficientNetClassifier()
classifier.load_model('models/best_classifier.pt')

print("✅ Classification model trained and loaded!")
print("Target accuracy: >95% for checkbox state classification")
```

## Expected Results
- **Dataset**: ~800 samples (40% synthetic, 60% real from FUNSD)  
- **Training time**: ~10-15 minutes on Colab GPU
- **Target accuracy**: >95% validation accuracy
- **Classes**: checked, unchecked, unclear

This completes the classification component of our hybrid pipeline!