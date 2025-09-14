# 🚀 Google Colab Training Guide

This guide shows how to train your YOLOv12 checkbox detection model on Google Colab with GPU acceleration.

## 📋 Prerequisites

1. **Google Account** with Colab access
2. **Colab Pro subscription** ($10/month) - Recommended for A100 GPU access
3. **Prepared dataset** from local development

## 🎯 Workflow Overview

```
Local Development → Colab Training → Download Results → Local Inference
```

## 📦 Step 1: Prepare Your Data Locally

First, run the data preparation on your local machine:

```bash
# Process your datasets
python scripts/prepare_datasets.py --process-funsd --augment

# Generate synthetic data
python scripts/generate_synthetic_checkboxes.py --num-forms 200

# Create final dataset
python scripts/run_data_preparation.py
```

This creates your training dataset in `data/processed/` with:
- `train/`: Training images and labels
- `val/`: Validation images and labels  
- `test/`: Test images and labels
- `data.yaml`: YOLO configuration file

## 📤 Step 2: Upload to Google Colab

### Option A: Direct Upload
1. Zip your `data/processed/` folder
2. Upload the colab files and data zip directly in the notebook

### Option B: Google Drive (Recommended for large datasets)
1. Upload your dataset to Google Drive:
   ```
   Google Drive/checkbox_detection_data/
   ├── data/
   │   └── processed/
   │       ├── train/
   │       ├── val/
   │       ├── test/
   │       └── data.yaml
   ├── colab/
   │   ├── setup_colab_environment.py
   │   ├── train_yolo12_colab.py
   │   └── colab_notebook_template.ipynb
   ```

## 🚀 Step 3: Open Colab and Start Training

1. **Open the notebook**: Upload `colab/colab_notebook_template.ipynb` to Colab
2. **Select GPU runtime**: Runtime → Change runtime type → A100 GPU (Pro) or T4 (Free)
3. **Run the cells sequentially**:
   - Environment setup
   - Data upload/mounting
   - YOLOv12 training
   - Results monitoring
   - Model download

## ⚡ Step 4: Expected Performance

### Training Times (Estimated):
- **A100 (Colab Pro)**: 1-2 hours for 100 epochs
- **T4 (Free Colab)**: 3-4 hours for 100 epochs
- **V100 (Colab Pro)**: 2-3 hours for 100 epochs

### Memory Usage:
- **YOLOv12n**: ~3-4GB VRAM
- **YOLOv12s**: ~5-6GB VRAM  
- **YOLOv12m**: ~8-10GB VRAM

### Target Metrics:
- **mAP@0.5**: >0.85 (Project requirement)
- **Training should converge**: Within 50-100 epochs
- **Session time**: <12 hours (Colab limit)

## 📊 Step 5: Monitor Training

The Colab notebook includes:
- **Real-time metrics**: mAP, precision, recall, loss curves
- **Automatic checkpointing**: Saves every 5 epochs
- **Drive backup**: Results automatically backed up
- **Session monitoring**: Tracks time to avoid disconnection

## 📥 Step 6: Download Results

After training completes:
1. **Trained models**: `best.pt` and `last.pt` weights
2. **Training plots**: Results graphs, confusion matrices
3. **Metrics**: JSON file with final performance
4. **Automatic zip**: Ready for download

## 🔄 Step 7: Use Results Locally

Copy the downloaded model back to your local project:

```bash
# Copy trained model to local project
cp downloaded_model/best.pt models/yolo12_detection/

# Test locally
python scripts/test_yolo12.py

# Run inference
python scripts/infer.py your_test_document.pdf
```

## 🎯 Troubleshooting

### Common Issues:

**"GPU not available"**
- Check runtime type (Runtime → Change runtime type)
- Free Colab has limited GPU hours
- Consider upgrading to Pro

**"Session disconnected"**
- Free Colab has 12-hour limit
- Use checkpointing (built into our training script)
- Resume from last checkpoint

**"Out of memory"**
- Reduce batch size in config
- Use smaller model (yolo12n instead of yolo12s)
- Clear GPU memory: `torch.cuda.empty_cache()`

**"Dataset not found"**
- Verify data.yaml path in the error message
- Check file structure matches expected format
- Ensure all image/label pairs exist

### Performance Tips:

1. **Use A100 GPU** (Colab Pro) for fastest training
2. **Enable automatic mixed precision** (already configured)
3. **Optimal batch size**: Let YOLO auto-detect (`batch: -1`)
4. **Monitor GPU utilization**: Should be >80% during training

## 🎉 Success Criteria

Your training is successful when:
- ✅ **mAP@0.5 > 0.85** (meets project requirements)
- ✅ **Convergence**: Loss curves stabilize
- ✅ **No overfitting**: Validation metrics improve with training
- ✅ **Fast inference**: <200ms per image on GPU

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section
2. Review Colab's GPU limits and quotas
3. Ensure your dataset format matches YOLO requirements
4. Consider starting with a smaller dataset for initial testing

---

**Ready to train?** Upload the notebook template to Colab and follow the step-by-step guide!