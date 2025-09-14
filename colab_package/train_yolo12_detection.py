#!/usr/bin/env python3
"""
Train YOLOv12 model for checkbox detection.
Uses the latest YOLOv12 architecture with attention mechanisms for improved small object detection.
"""

import os
import sys
import torch
from pathlib import Path
import argparse
import logging
from typing import Dict, Any

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO
from src.utils.config import load_config, DetectionConfig


# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_yolo12_availability():
    """Check if YOLOv12 models are available."""
    try:
        # Try to load YOLOv12 nano model
        model = YOLO('yolo12n.pt')
        logger.info("✅ YOLOv12 models are available!")
        return True
    except Exception as e:
        logger.error(f"❌ YOLOv12 not available: {e}")
        logger.info("💡 Make sure you have the latest Ultralytics version:")
        logger.info("   pip install ultralytics>=8.3.0")
        return False


def setup_environment():
    """Setup training environment and check requirements."""
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        logger.info(f"🚀 GPU Available: {gpu_name} (Count: {gpu_count})")
        logger.info(f"🔥 CUDA Version: {torch.version.cuda}")
        
        # Check for FlashAttention support (YOLOv12 optimization)
        gpu_capability = torch.cuda.get_device_capability(0)
        if gpu_capability[0] >= 7:  # Turing (T4), Ampere (RTX 30/A100), Ada Lovelace (RTX 40)
            logger.info("✨ FlashAttention supported - YOLOv12 will run optimally!")
        else:
            logger.warning("⚠️  Older GPU detected - YOLOv12 will use standard attention")
    else:
        logger.warning("⚠️  No GPU detected - training will be slow on CPU")


def train_yolo12_detection(
    data_config: str,
    model_variant: str = "yolo12n.pt",
    config: DetectionConfig = None,
    output_dir: str = "models/yolo12_detection"
) -> Dict[str, Any]:
    """
    Train YOLOv12 model for checkbox detection.
    
    Args:
        data_config: Path to YOLO dataset configuration file
        model_variant: YOLOv12 model variant (yolo12n.pt, yolo12s.pt, etc.)
        config: Detection configuration
        output_dir: Directory to save trained model
        
    Returns:
        Training results dictionary
    """
    
    logger.info(f"🎯 Starting YOLOv12 Training")
    logger.info(f"📊 Dataset config: {data_config}")
    logger.info(f"🤖 Model variant: {model_variant}")
    
    # Load model
    model = YOLO(model_variant)
    logger.info(f"📥 Loaded {model_variant} with {model.model.yaml} architecture")
    
    # Setup training arguments
    if config is None:
        config = DetectionConfig()
    
    train_args = {
        'data': data_config,
        'epochs': config.epochs,
        'imgsz': config.img_size,
        'batch': config.batch_size,
        'lr0': config.learning_rate,
        'lrf': 0.01,  # Final learning rate (lr0 * lrf)
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'patience': config.patience,
        'save_period': config.save_period,
        'device': config.device,
        'workers': config.workers,
        'project': str(Path(output_dir).parent),
        'name': Path(output_dir).name,
        'exist_ok': True,
        'verbose': True,
        'seed': 42,
        
        # YOLOv12-specific optimizations
        'amp': True,  # Automatic Mixed Precision
        'fraction': 1.0,  # Use full dataset
        'profile': False,  # Disable profiling for speed
        
        # Data augmentation (tuned for checkboxes)
        'hsv_h': 0.01,    # Minimal hue augmentation (forms are usually B&W)
        'hsv_s': 0.3,     # Reduced saturation 
        'hsv_v': 0.2,     # Reduced value augmentation
        'degrees': 5.0,   # Small rotation (documents can be slightly skewed)
        'translate': 0.05, # Minimal translation
        'scale': 0.2,     # Scale variation for different checkbox sizes
        'shear': 0.0,     # No shearing (preserves checkbox shape)
        'perspective': 0.0, # No perspective (2D documents)
        'flipud': 0.0,    # No vertical flips (checkboxes have orientation)
        'fliplr': 0.0,    # No horizontal flips 
        'mosaic': 0.5,    # Reduced mosaic (can break checkbox context)
        'mixup': 0.0,     # No mixup (preserves checkbox integrity)
        'copy_paste': 0.0, # No copy-paste
        
        # Loss function weights
        'box': 7.5,       # Bounding box loss weight
        'cls': 0.5,       # Classification loss weight (single class)
        'dfl': 1.5,       # Distribution focal loss weight
        
        # Optimization
        'optimizer': 'AdamW',  # AdamW optimizer (better for attention models)
        'close_mosaic': 10,    # Disable mosaic in last N epochs for stable training
    }
    
    logger.info("🏋️ Training Arguments:")
    for key, value in train_args.items():
        if key not in ['data']:  # Don't log full path
            logger.info(f"  {key}: {value}")
    
    # Start training
    logger.info("🚀 Starting training...")
    results = model.train(**train_args)
    
    # Training completed
    output_path = Path(output_dir)
    best_model = output_path / "weights" / "best.pt"
    last_model = output_path / "weights" / "last.pt"
    
    if best_model.exists():
        logger.info(f"✅ Training completed successfully!")
        logger.info(f"📁 Best model: {best_model}")
        logger.info(f"📁 Last model: {last_model}")
        
        # Load best model for evaluation
        best_model_obj = YOLO(str(best_model))
        
        # Quick validation
        val_results = best_model_obj.val(data=data_config, verbose=False)
        
        logger.info("📊 Final Validation Metrics:")
        logger.info(f"  mAP50: {val_results.box.map50:.4f}")
        logger.info(f"  mAP50-95: {val_results.box.map:.4f}")
        logger.info(f"  Precision: {val_results.box.mp:.4f}")
        logger.info(f"  Recall: {val_results.box.mr:.4f}")
        
        return {
            'success': True,
            'best_model_path': str(best_model),
            'last_model_path': str(last_model),
            'mAP50': float(val_results.box.map50),
            'mAP50_95': float(val_results.box.map),
            'precision': float(val_results.box.mp),
            'recall': float(val_results.box.mr),
            'training_args': train_args
        }
    else:
        logger.error("❌ Training failed - no model weights found")
        return {'success': False}


def export_model_for_deployment(model_path: str, export_formats: list = ['onnx']):
    """
    Export trained model for deployment.
    
    Args:
        model_path: Path to trained model
        export_formats: List of export formats (onnx, torchscript, etc.)
    """
    logger.info(f"📦 Exporting model for deployment...")
    
    model = YOLO(model_path)
    
    for fmt in export_formats:
        try:
            exported_path = model.export(
                format=fmt,
                dynamic=True,  # Dynamic input shapes
                simplify=True if fmt == 'onnx' else False,
                optimize=True,
                half=False  # Keep FP32 for compatibility
            )
            logger.info(f"✅ Exported {fmt}: {exported_path}")
        except Exception as e:
            logger.error(f"❌ Failed to export {fmt}: {e}")


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train YOLOv12 for checkbox detection")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to YOLO dataset configuration file")
    parser.add_argument("--model", type=str, default="yolo12n.pt",
                       choices=["yolo12n.pt", "yolo12s.pt", "yolo12m.pt", "yolo12l.pt", "yolo12x.pt"],
                       help="YOLOv12 model variant")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Configuration file")
    parser.add_argument("--output-dir", type=str, default="models/yolo12_detection",
                       help="Output directory for trained model")
    parser.add_argument("--export", action="store_true",
                       help="Export model after training")
    parser.add_argument("--export-formats", nargs="+", default=["onnx"],
                       help="Export formats")
    
    args = parser.parse_args()
    
    # Setup environment
    setup_environment()
    
    # Check YOLOv12 availability  
    if not check_yolo12_availability():
        logger.error("YOLOv12 not available. Please update Ultralytics.")
        return 1
    
    # Load configuration
    config_dict = load_config(args.config)
    detection_config = DetectionConfig(**config_dict.get('detection', {}))
    detection_config.model_name = args.model  # Override with CLI argument
    
    # Check if dataset config exists
    if not Path(args.data).exists():
        logger.error(f"Dataset config not found: {args.data}")
        logger.info("Run data preparation first:")
        logger.info("  python scripts/prepare_datasets.py --process-funsd")
        return 1
    
    # Train model
    results = train_yolo12_detection(
        data_config=args.data,
        model_variant=args.model,
        config=detection_config,
        output_dir=args.output_dir
    )
    
    if results['success']:
        logger.info("🎉 Training completed successfully!")
        
        # Export model if requested
        if args.export:
            export_model_for_deployment(
                results['best_model_path'], 
                args.export_formats
            )
        
        logger.info("📋 Training Summary:")
        logger.info(f"  Best model: {results['best_model_path']}")
        logger.info(f"  mAP@0.5: {results['mAP50']:.4f}")
        logger.info(f"  mAP@0.5:0.95: {results['mAP50_95']:.4f}")
        
        if results['mAP50'] > 0.85:
            logger.info("🎯 TARGET ACHIEVED: mAP@0.5 > 0.85!")
        else:
            logger.info(f"🎯 Target: mAP@0.5 > 0.85 (Current: {results['mAP50']:.4f})")
        
        return 0
    else:
        logger.error("❌ Training failed")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)