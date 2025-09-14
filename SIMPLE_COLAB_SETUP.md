# 🚀 Simple Colab Training - Minimal Setup

## Step 1: Upload Files to Colab

Just upload these 3 files to Colab:
1. `scripts/unified_data_preparation.py`
2. `scripts/generate_synthetic_checkboxes.py` 
3. `scripts/extract_checkbox_labels_from_funsd.py`

## Step 2: Run These Commands in Colab

```python
# Install dependencies
!pip install ultralytics opencv-python pillow tqdm pyyaml

# Create directory structure
!mkdir -p data/raw/funsd/dataset/training_data/{images,annotations}
!mkdir -p data/raw/funsd/dataset/testing_data/{images,annotations}
!mkdir -p data/processed/{train,val,test}/{images,labels}

# Download FUNSD dataset
!wget -q "https://guillaumejaume.github.io/FUNSD/dataset.zip"
!unzip -q dataset.zip -d data/raw/funsd/
```

## Step 3: Prepare Data

```python
# Run the unified data preparation
!python unified_data_preparation.py --synthetic-images 300

# This creates:
# - data/processed/train/ (images + labels)
# - data/processed/val/ (images + labels) 
# - data/processed/test/ (images + labels)
# - data/processed/data.yaml (YOLO config)
```

## Step 4: Train YOLOv12

```python
from ultralytics import YOLO

# Load YOLOv12 model
model = YOLO('yolo12n.pt')  # Downloads automatically

# Train
results = model.train(
    data='data/processed/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    device='0'  # Use GPU
)

# Best model saved at: runs/detect/train/weights/best.pt
```

## Step 5: Test Results

```python
# Load best model
model = YOLO('runs/detect/train/weights/best.pt')

# Test on validation set
metrics = model.val()
print(f"mAP@0.5: {metrics.box.map50:.3f}")

# Quick inference test
results = model('data/processed/test/images/')[0]
results.show()  # Display detection
```

## That's It!

- **No complex scripts**
- **No file management hassles** 
- **Direct commands you can copy-paste**
- **Should work in ~30 minutes**

Target: mAP@0.5 > 0.85 after 50 epochs.