#!/usr/bin/env python3
"""
Test YOLOv12 availability and basic functionality.
Quick check to ensure YOLOv12 models work correctly.
"""

import sys
import torch
from pathlib import Path
import logging
import numpy as np
import cv2

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from ultralytics import YOLO

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_yolo12_availability():
    """Test if YOLOv12 models are available and working."""
    
    logger.info("🧪 Testing YOLOv12 Availability")
    logger.info("=" * 50)
    
    # Test different YOLOv12 variants
    variants = ['yolo12n.pt', 'yolo12s.pt', 'yolo12m.pt']
    
    for variant in variants:
        try:
            logger.info(f"📦 Testing {variant}...")
            
            # Load model
            model = YOLO(variant)
            logger.info(f"✅ Successfully loaded {variant}")
            
            # Get model info
            logger.info(f"   📊 Parameters: {sum(p.numel() for p in model.model.parameters()):,}")
            logger.info(f"   🏗️ Architecture: {model.model.yaml.get('yaml_file', 'Unknown')}")
            
            # Test inference on dummy image
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # Run inference
            results = model.predict(dummy_img, verbose=False)
            logger.info(f"   🚀 Inference successful")
            
            # Check speed
            if hasattr(results[0], 'speed'):
                speed = results[0].speed
                total_time = sum(speed.values()) if isinstance(speed, dict) else speed
                logger.info(f"   ⚡ Speed: {total_time:.1f}ms")
            
            logger.info(f"✅ {variant} working correctly!\n")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {variant}: {e}")
            if "yolo12" in str(e).lower():
                logger.error("   💡 Try updating Ultralytics: pip install ultralytics>=8.3.0")
            logger.error("")


def test_gpu_compatibility():
    """Test GPU compatibility for YOLOv12."""
    
    logger.info("🎮 Testing GPU Compatibility")
    logger.info("=" * 50)
    
    if not torch.cuda.is_available():
        logger.warning("❌ CUDA not available - using CPU")
        return
    
    # Get GPU info
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    
    logger.info(f"🚀 GPU: {gpu_name}")
    logger.info(f"🔥 CUDA Version: {cuda_version}")
    logger.info(f"📊 GPU Count: {gpu_count}")
    
    # Check FlashAttention compatibility
    gpu_capability = torch.cuda.get_device_capability(0)
    major, minor = gpu_capability
    
    logger.info(f"🏗️ GPU Compute Capability: {major}.{minor}")
    
    if major >= 7:  # Turing, Ampere, Ada Lovelace, Hopper
        logger.info("✨ FlashAttention supported - YOLOv12 will run optimally!")
        
        # Test with GPU
        try:
            model = YOLO('yolo12n.pt')
            dummy_img = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            # GPU inference
            results = model.predict(dummy_img, device=0, verbose=False)
            logger.info("🚀 GPU inference successful!")
            
        except Exception as e:
            logger.error(f"❌ GPU inference failed: {e}")
    
    elif major >= 6:  # Pascal
        logger.warning("⚠️ Older GPU - YOLOv12 will use standard attention")
        logger.warning("   Still supported but may be slower than newer GPUs")
    
    else:
        logger.warning("⚠️ Very old GPU - may have compatibility issues")


def test_checkbox_detection():
    """Test YOLOv12 on a synthetic checkbox image."""
    
    logger.info("📋 Testing Checkbox Detection")
    logger.info("=" * 50)
    
    try:
        # Create synthetic checkbox image
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White background
        
        # Draw some checkboxes
        checkboxes = [
            (50, 50, 30, 30),   # x, y, w, h
            (200, 100, 25, 25),
            (350, 150, 35, 35),
        ]
        
        for x, y, w, h in checkboxes:
            # Draw checkbox border
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            
            # Draw checkmark in first checkbox
            if x == 50:
                cv2.line(img, (x + 5, y + 15), (x + 12, y + 22), (0, 0, 0), 2)
                cv2.line(img, (x + 12, y + 22), (x + 25, y + 8), (0, 0, 0), 2)
        
        # Save test image
        test_img_path = Path("test_checkbox.png")
        cv2.imwrite(str(test_img_path), img)
        logger.info(f"💾 Created test image: {test_img_path}")
        
        # Load YOLOv12 model
        model = YOLO('yolo12n.pt')
        
        # Run detection (this will be on COCO classes initially)
        results = model.predict(img, conf=0.1, verbose=False)
        
        # Check results
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            num_detections = len(results[0].boxes)
            logger.info(f"🎯 Found {num_detections} objects (COCO classes)")
            
            # Show top detections
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()
            
            for i, (conf, cls) in enumerate(zip(confidences[:3], classes[:3])):
                class_name = model.names[int(cls)]
                logger.info(f"   {i+1}. {class_name}: {conf:.3f}")
        
        else:
            logger.info("🤷 No objects detected (expected - model not trained on checkboxes yet)")
        
        logger.info("✅ Basic detection test completed")
        
        # Clean up
        if test_img_path.exists():
            test_img_path.unlink()
        
    except Exception as e:
        logger.error(f"❌ Checkbox detection test failed: {e}")


def main():
    """Run all tests."""
    logger.info("🧪 YOLOv12 Compatibility Test Suite")
    logger.info("=" * 60)
    
    # Test 1: YOLOv12 availability
    test_yolo12_availability()
    
    # Test 2: GPU compatibility
    test_gpu_compatibility()
    
    # Test 3: Basic detection
    test_checkbox_detection()
    
    logger.info("🎉 Test suite completed!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Prepare your checkbox dataset")
    logger.info("2. Run: python scripts/prepare_datasets.py --process-funsd")
    logger.info("3. Train: python scripts/train_yolo12_detection.py --data data/processed/data.yaml")


if __name__ == "__main__":
    main()