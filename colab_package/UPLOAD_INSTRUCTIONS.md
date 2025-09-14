# 🚀 Colab Upload Instructions

## Quick Start for Colab Training

### 1. Open Google Colab
- Go to https://colab.research.google.com/
- Make sure you're signed in to your Google account
- Select "GPU" runtime (Runtime → Change runtime type → T4 or A100)

### 2. Upload This Package
- In Colab, create a new notebook
- Upload the `colab_notebook_template.ipynb` file
- Or copy-paste the cells from the template

### 3. Upload Project Files
When the notebook asks, upload these files:
- **`colab_package.zip`** - Contains all project code and data
- Follow the notebook instructions step by step

### 4. Expected Training Time
- **T4 GPU (Free)**: 2-3 hours
- **A100 GPU (Pro)**: 1-2 hours
- **Target**: mAP@0.5 > 0.85

### 5. What You'll Get
- Trained YOLOv12 model (`best.pt`)
- Training metrics and plots
- Downloadable results package

## 🎯 Quick Commands for Colab

Once uploaded, these commands will work in Colab:

```python
# In the first cell:
!unzip colab_package.zip -d /content/
%cd /content/colab_package

# Run setup:
!python colab/setup_colab_environment.py

# Process data in Colab:
!python scripts/prepare_datasets.py --process-funsd --augment

# Generate synthetic data:
!python scripts/generate_synthetic_checkboxes.py --num-forms 100

# Train model:
!python colab/train_yolo12_colab.py
```

## 📊 Dataset Info

Your package includes:
- **CheckboxQA**: Checkbox-specific dataset (metadata)
- **FUNSD**: Form understanding dataset (149 train, 50 test images)
- **Synthetic generator**: Creates additional training data
- **YOLOv12 trainer**: Optimized for Colab

The training will create ~500-1000 training images total, which is perfect for checkbox detection.

**Ready?** Upload to Colab and start training! 🚀