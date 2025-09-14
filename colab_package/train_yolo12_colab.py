#!/usr/bin/env python3
"""
Colab-optimized YOLOv12 training script for checkbox detection.
Handles Colab-specific requirements like session management and Drive sync.
"""

import os
import sys
import time
import json
import shutil
from pathlib import Path
from datetime import datetime
import torch

# Colab-specific imports
try:
    from google.colab import files, drive
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

from ultralytics import YOLO
import yaml


class ColabTrainingManager:
    """Manages YOLOv12 training in Google Colab environment."""
    
    def __init__(self):
        self.project_dir = Path("/content/checkbox_detection")
        self.drive_backup_dir = Path("/content/drive/MyDrive/checkbox_detection_results")
        self.start_time = datetime.now()
        
        # Create backup directory
        if IS_COLAB:
            self.drive_backup_dir.mkdir(parents=True, exist_ok=True)
    
    def check_session_time(self):
        """Check how long the session has been running."""
        elapsed = datetime.now() - self.start_time
        elapsed_hours = elapsed.total_seconds() / 3600
        
        print(f"⏰ Session time: {elapsed_hours:.1f} hours")
        
        if elapsed_hours > 10:
            print("⚠️ Session has been running for >10 hours - consider saving progress")
        
        return elapsed_hours
    
    def backup_to_drive(self, model_path: Path, metrics: dict):
        """Backup training results to Google Drive."""
        if not IS_COLAB:
            print("Not in Colab - skipping Drive backup")
            return
        
        print("💾 Backing up results to Google Drive...")
        
        # Create timestamped backup folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_dir = self.drive_backup_dir / f"training_{timestamp}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Backup model files
            if model_path.exists():
                weights_dir = model_path / "weights"
                if weights_dir.exists():
                    shutil.copytree(weights_dir, backup_dir / "weights")
                    print(f"✅ Model weights backed up to: {backup_dir / 'weights'}")
            
            # Backup training results
            results_dir = model_path / "runs" if (model_path / "runs").exists() else model_path.parent
            for result_file in results_dir.glob("*.png"):
                shutil.copy2(result_file, backup_dir)
            
            # Save training metrics
            with open(backup_dir / "training_metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            # Save training log
            with open(backup_dir / "training_log.txt", 'w') as f:
                f.write(f"Training completed at: {datetime.now()}\n")
                f.write(f"Session duration: {self.check_session_time():.1f} hours\n")
                f.write(f"Final metrics: {metrics}\n")
            
            print(f"✅ Complete backup saved to: {backup_dir}")
            
        except Exception as e:
            print(f"❌ Backup failed: {e}")
    
    def train_with_checkpoints(self, data_config: str, model_variant: str = "yolo12n.pt"):
        """Train with automatic checkpointing for long sessions."""
        
        print("🚀 Starting Colab YOLOv12 Training")
        print(f"📊 Data config: {data_config}")
        print(f"🤖 Model: {model_variant}")
        
        # Check GPU
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f} GB)")
        else:
            print("❌ No GPU available!")
            return None
        
        # Load model
        model = YOLO(model_variant)
        print(f"📥 Loaded {model_variant}")
        
        # Colab-optimized training arguments
        train_args = {
            'data': data_config,
            'epochs': 100,
            'imgsz': 640,
            'batch': -1,  # Auto batch size
            'lr0': 0.01,
            'lrf': 0.01,
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'patience': 20,  # More patience for better results
            'save_period': 5,   # Save every 5 epochs
            'device': 0,
            'workers': 2,  # Conservative for Colab
            'project': str(self.project_dir / "models"),
            'name': f"yolo12_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'exist_ok': True,
            'verbose': True,
            'seed': 42,
            'deterministic': False,  # Allow some randomness for better results
            
            # Mixed precision and optimization
            'amp': True,  # Automatic Mixed Precision
            'fraction': 1.0,  # Use full dataset
            'profile': False,
            
            # Colab-friendly augmentation (minimal for checkboxes)
            'hsv_h': 0.01,
            'hsv_s': 0.3,
            'hsv_v': 0.2,
            'degrees': 5.0,
            'translate': 0.05,
            'scale': 0.15,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'mosaic': 0.3,  # Reduced for checkbox context
            'mixup': 0.0,
            'copy_paste': 0.0,
            
            # Loss weights
            'box': 7.5,
            'cls': 0.5,
            'dfl': 1.5,
            
            # Optimizer
            'optimizer': 'AdamW',
            'close_mosaic': 15,  # Disable mosaic in last 15 epochs
        }
        
        print("🏋️ Training configuration:")
        for key, value in train_args.items():
            if key != 'data':
                print(f"  {key}: {value}")
        
        # Start training with session monitoring
        try:
            print("\n🚀 Starting training...")
            results = model.train(**train_args)
            
            # Get model path
            model_path = Path(train_args['project']) / train_args['name']
            
            # Evaluate final model
            if (model_path / "weights" / "best.pt").exists():
                print("📊 Evaluating final model...")
                best_model = YOLO(str(model_path / "weights" / "best.pt"))
                val_results = best_model.val(data=data_config, verbose=True)
                
                final_metrics = {
                    'mAP50': float(val_results.box.map50),
                    'mAP50_95': float(val_results.box.map),
                    'precision': float(val_results.box.mp),
                    'recall': float(val_results.box.mr),
                    'training_time_hours': self.check_session_time(),
                    'model_path': str(model_path),
                    'model_variant': model_variant
                }
                
                print("\n🎉 Training completed!")
                print("📊 Final Results:")
                print(f"  mAP@0.5: {final_metrics['mAP50']:.4f}")
                print(f"  mAP@0.5:0.95: {final_metrics['mAP50_95']:.4f}")
                print(f"  Precision: {final_metrics['precision']:.4f}")
                print(f"  Recall: {final_metrics['recall']:.4f}")
                print(f"  Training time: {final_metrics['training_time_hours']:.1f} hours")
                
                # Check if target achieved
                if final_metrics['mAP50'] > 0.85:
                    print("🎯 TARGET ACHIEVED: mAP@0.5 > 0.85! 🎉")
                else:
                    print(f"🎯 Target: mAP@0.5 > 0.85 (Current: {final_metrics['mAP50']:.4f})")
                
                # Backup results
                self.backup_to_drive(model_path, final_metrics)
                
                return final_metrics
            
        except KeyboardInterrupt:
            print("⏸️ Training interrupted by user")
            # Still try to backup partial results
            model_path = Path(train_args['project']) / train_args['name']
            if model_path.exists():
                self.backup_to_drive(model_path, {"status": "interrupted"})
        
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return None
    
    def download_results(self):
        """Download results to local machine."""
        if not IS_COLAB:
            print("Not in Colab - no download needed")
            return
        
        print("📥 Preparing results for download...")
        
        # Create download package
        download_dir = Path("/content/download_package")
        download_dir.mkdir(exist_ok=True)
        
        # Copy model weights
        models_dir = self.project_dir / "models"
        if models_dir.exists():
            for weights_dir in models_dir.glob("*/weights"):
                if (weights_dir / "best.pt").exists():
                    model_name = weights_dir.parent.name
                    target_dir = download_dir / f"model_{model_name}"
                    target_dir.mkdir(exist_ok=True)
                    
                    # Copy essential files
                    shutil.copy2(weights_dir / "best.pt", target_dir)
                    if (weights_dir / "last.pt").exists():
                        shutil.copy2(weights_dir / "last.pt", target_dir)
        
        # Copy training plots
        for plot_file in self.project_dir.rglob("*.png"):
            if "results" in plot_file.name or "confusion" in plot_file.name:
                shutil.copy2(plot_file, download_dir)
        
        # Create zip file
        shutil.make_archive("/content/checkbox_detection_results", 'zip', download_dir)
        
        print("✅ Results package ready for download")
        print("Run the following in a new cell to download:")
        print("files.download('/content/checkbox_detection_results.zip')")


def main():
    """Main Colab training function."""
    print("🚀 YOLOv12 Checkbox Detection Training for Google Colab")
    print("=" * 60)
    
    # Initialize training manager
    trainer = ColabTrainingManager()
    
    # Check for data
    data_config = "/content/checkbox_detection/data/processed/data.yaml"
    
    if not Path(data_config).exists():
        print("❌ Dataset not found!")
        print("📋 Setup steps:")
        print("1. Upload your prepared dataset to Google Drive")
        print("2. Copy to Colab:")
        print("   !cp -r /content/drive/MyDrive/checkbox_detection_data/* /content/checkbox_detection/data/")
        print("3. Verify data.yaml exists at:", data_config)
        return 1
    
    # Start training
    results = trainer.train_with_checkpoints(data_config, "yolo12n.pt")
    
    if results:
        print("\n🎉 Training completed successfully!")
        print("📊 Results backed up to Google Drive")
        print("📥 Run trainer.download_results() to prepare local download")
        return 0
    else:
        print("❌ Training failed")
        return 1


# Colab convenience functions
def quick_train():
    """Quick training function for Colab cells."""
    return main()

def setup_and_train():
    """Setup environment and start training."""
    # Run setup first
    exec(open('/content/checkbox_detection/colab/setup_colab_environment.py').read())
    
    # Then train
    return quick_train()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)