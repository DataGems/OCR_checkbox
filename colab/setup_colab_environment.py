"""
Setup script for Google Colab environment.
Installs dependencies, mounts Drive, and prepares the training environment.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_colab_environment():
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def install_dependencies():
    """Install required packages."""
    print("🔧 Installing dependencies...")
    
    # Update pip first
    subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
    
    # Install main packages
    packages = [
        "ultralytics>=8.3.0",  # Latest version with YOLOv12
        "torch>=2.0.0",
        "torchvision",
        "opencv-python",
        "albumentations",
        "pdf2image",
        "Pillow",
        "numpy",
        "pandas",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pyyaml",
        "timm",  # For EfficientNet
        "scikit-learn",
        "scipy",
        "pymupdf"  # Alternative PDF processing
    ]
    
    for package in packages:
        print(f"📦 Installing {package}...")
        subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
    
    print("✅ Dependencies installed successfully!")


def mount_google_drive():
    """Mount Google Drive."""
    if not check_colab_environment():
        print("⚠️ Not in Colab - skipping Drive mount")
        return
    
    print("📁 Mounting Google Drive...")
    from google.colab import drive
    drive.mount('/content/drive')
    print("✅ Google Drive mounted!")


def setup_project_structure():
    """Setup project directory structure."""
    print("📂 Setting up project structure...")
    
    # Create main directories
    base_dir = Path("/content/checkbox_detection")
    base_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    subdirs = [
        "data/raw",
        "data/processed/train/images",
        "data/processed/train/labels", 
        "data/processed/val/images",
        "data/processed/val/labels",
        "data/processed/test/images",
        "data/processed/test/labels",
        "data/synthetic",
        "models",
        "configs",
        "scripts",
        "src/detection",
        "src/classification",
        "src/pipeline",
        "src/utils",
        "results",
        "logs"
    ]
    
    for subdir in subdirs:
        (base_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Change to project directory
    os.chdir(str(base_dir))
    print(f"📂 Project structure created at: {base_dir}")
    return base_dir


def setup_gpu_environment():
    """Setup and verify GPU environment."""
    print("🎮 Setting up GPU environment...")
    
    import torch
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 GPU Available: {gpu_name}")
        print(f"🔢 GPU Count: {gpu_count}")
        print(f"🧠 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Check compute capability for FlashAttention
        capability = torch.cuda.get_device_capability(0)
        if capability[0] >= 7:
            print("✨ FlashAttention supported - YOLOv12 optimizations enabled!")
        else:
            print("⚠️ Older GPU - standard attention will be used")
        
        # Test GPU
        test_tensor = torch.randn(1000, 1000).cuda()
        result = torch.mm(test_tensor, test_tensor)
        print("✅ GPU test successful!")
        
    else:
        print("❌ No GPU available - using CPU (will be slow)")
    
    return torch.cuda.is_available()


def download_sample_data():
    """Download and setup sample data."""
    print("📥 Setting up sample data...")
    
    # We'll copy this from the mounted Drive or download from GitHub
    print("💡 Upload your prepared dataset to Google Drive")
    print("   - Folder: /content/drive/MyDrive/checkbox_detection_data/")
    print("   - Include: data/processed/ with train/val/test splits")


def create_colab_config():
    """Create Colab-specific configuration."""
    print("⚙️ Creating Colab configuration...")
    
    config_content = """# Colab-optimized configuration for checkbox detection

# Global settings
project_name: "checkbox_detection_colab"
experiment_name: "yolo12_training"
seed: 42
verbose: true

# Detection model configuration (YOLOv12) - Colab optimized
detection:
  model_name: "yolo12n.pt"     # Start with nano for speed
  img_size: 640
  batch_size: 32               # Increased for GPU
  epochs: 100
  learning_rate: 0.01
  patience: 15                 # More patience for better results
  save_period: 5               # Save more frequently
  device: "0"                  # GPU device
  workers: 2                   # Reduced for Colab stability
  
  # Colab-optimized augmentation
  hsv_h: 0.01
  hsv_s: 0.3
  hsv_v: 0.2
  degrees: 5.0
  translate: 0.05
  scale: 0.2
  shear: 0.0
  perspective: 0.0
  flipud: 0.0
  fliplr: 0.0
  mosaic: 0.5
  mixup: 0.0

# Classification model configuration
classification:
  model_name: "efficientnet_b0"
  num_classes: 3
  img_size: 128
  batch_size: 64               # Increased for GPU
  epochs: 50
  learning_rate: 0.001
  weight_decay: 0.0001
  patience: 10
  device: "cuda"
  workers: 2

# Data configuration
data:
  raw_data_dir: "/content/checkbox_detection/data/raw"
  processed_data_dir: "/content/checkbox_detection/data/processed"
  train_split: 0.7
  val_split: 0.15
  test_split: 0.15
  pdf_dpi: 300
  apply_deskew: true
  apply_denoise: true
  denoise_sigma: 0.5
  synthetic_ratio: 0.3

# Pipeline configuration - Colab optimized
pipeline:
  detection_confidence: 0.25
  nms_iou_threshold: 0.45
  max_detections: 1000
  classification_confidence: 0.8
  batch_processing: true
  max_batch_size: 16           # Conservative for Colab
  enable_quantization: false
  output_format: "json"
  include_crops: false
  confidence_threshold: 0.5
"""
    
    config_path = Path("/content/checkbox_detection/configs/colab_config.yaml")
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ Colab config created at: {config_path}")
    return config_path


def main():
    """Main setup function for Colab."""
    print("🚀 Setting up OCR Checkbox Detection in Google Colab")
    print("=" * 60)
    
    # Check environment
    is_colab = check_colab_environment()
    if is_colab:
        print("✅ Running in Google Colab")
    else:
        print("⚠️ Not in Colab - some features may not work")
    
    # Setup steps
    try:
        # 1. Install dependencies
        install_dependencies()
        
        # 2. Mount Drive (if in Colab)
        if is_colab:
            mount_google_drive()
        
        # 3. Setup project structure
        project_dir = setup_project_structure()
        
        # 4. Setup GPU
        has_gpu = setup_gpu_environment()
        
        # 5. Create config
        config_path = create_colab_config()
        
        # 6. Setup data
        download_sample_data()
        
        print("\n🎉 Setup completed successfully!")
        print("=" * 60)
        print("📋 Next steps:")
        print("1. Upload your dataset to Google Drive")
        print("2. Copy dataset to Colab:")
        print("   !cp -r /content/drive/MyDrive/checkbox_detection_data/* /content/checkbox_detection/data/")
        print("3. Run training:")
        print("   !python scripts/train_yolo12_colab.py")
        print("4. Download results:")
        print("   Copy trained models back to Drive")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)