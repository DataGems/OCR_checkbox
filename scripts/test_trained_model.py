#!/usr/bin/env python3
"""
Test the trained YOLOv12 checkbox detection model.
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

def test_trained_model():
    """Test the trained model with sample images."""
    
    # Load trained model
    model_path = Path("models") / "best.pt"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please ensure you've downloaded best.pt from Colab to the models/ folder")
        return False
    
    print(f"📦 Loading model from {model_path}")
    model = YOLO(str(model_path))
    
    # Test on some sample images
    test_dirs = [
        Path("data/raw/funsd/dataset/training_data/images"),
        Path("data/raw/funsd/dataset/testing_data/images")
    ]
    
    results_found = False
    
    for test_dir in test_dirs:
        if not test_dir.exists():
            continue
            
        print(f"\n🔍 Testing on images from {test_dir}")
        
        # Test on first few images
        for i, img_path in enumerate(test_dir.glob("*.png")):
            if i >= 3:  # Test only first 3 images
                break
                
            print(f"\n📸 Testing: {img_path.name}")
            
            # Run inference
            results = model(str(img_path))
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    results_found = True
                    print(f"   ✅ Found {len(boxes)} checkboxes")
                    
                    # Print detection details
                    for j, box in enumerate(boxes):
                        conf = box.conf[0].item()
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        print(f"      Box {j+1}: confidence={conf:.3f}, bbox=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")
                    
                    # Save annotated image
                    output_path = Path("test_results") / f"detected_{img_path.name}"
                    output_path.parent.mkdir(exist_ok=True)
                    
                    annotated = result.plot()
                    cv2.imwrite(str(output_path), annotated)
                    print(f"   💾 Saved result: {output_path}")
                    
                else:
                    print(f"   ⚠️  No checkboxes detected")
    
    if results_found:
        print(f"\n✅ Model testing completed! Check test_results/ for annotated images.")
        return True
    else:
        print(f"\n⚠️  No checkboxes detected in test images. This might indicate:")
        print("   - Images don't contain checkboxes")
        print("   - Model confidence threshold too high")
        print("   - Model needs different input preprocessing")
        return False

def test_model_info():
    """Print model information."""
    model_path = Path("models") / "best.pt"
    
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        return
    
    print(f"📊 Model Information:")
    model = YOLO(str(model_path))
    
    # Print model summary
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    print(f"Model file: {model_path}")
    print(f"Model type: YOLOv12")
    print(f"Classes: {model.names}")
    print(f"Number of classes: {len(model.names)}")
    
    # Get model metrics if available
    try:
        results = model.val()
        if results:
            print(f"Validation mAP@0.5: {results.box.map50:.3f}")
            print(f"Validation mAP@0.5:0.95: {results.box.map:.3f}")
    except:
        print("No validation metrics available")
    
    print("="*50)

if __name__ == "__main__":
    print("🚀 Testing Trained YOLOv12 Checkbox Detection Model\n")
    
    # First show model info
    test_model_info()
    
    # Then test on sample images
    success = test_trained_model()
    
    if success:
        print(f"\n🎯 Next steps:")
        print(f"   1. Check test_results/ for detection examples")
        print(f"   2. Update src/detection/detector.py to use this model") 
        print(f"   3. Test the full pipeline with python scripts/infer.py")
    else:
        print(f"\n🔧 Troubleshooting:")
        print(f"   1. Verify model file is correct: ls -la models/best.pt")
        print(f"   2. Try lowering confidence: model.predict(conf=0.1)")
        print(f"   3. Check input image format and size")