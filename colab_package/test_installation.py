#!/usr/bin/env python3
"""
Test script to verify checkbox detection pipeline installation and functionality.
"""

import sys
import traceback
from pathlib import Path
import tempfile
import numpy as np
import cv2

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test core dependencies
        import torch
        import torchvision
        import cv2
        import numpy as np
        from PIL import Image
        import yaml
        import albumentations
        
        print("✓ Core dependencies imported successfully")
        
        # Test project modules
        from src.utils.config import Config, load_config, get_default_config
        from src.utils.pdf_processing import pdf_to_images, preprocess_image
        from src.detection.detector import CheckboxDetector
        from src.classification.classifier import CheckboxClassifier
        from src.pipeline.pipeline import CheckboxPipeline
        
        print("✓ Project modules imported successfully")
        
        # Test specific components
        import ultralytics
        import timm
        
        print("✓ ML framework modules imported successfully")
        
        return True
        
    except Exception as e:
        print(f"✗ Import failed: {str(e)}")
        traceback.print_exc()
        return False


def test_config():
    """Test configuration loading and management."""
    print("\\nTesting configuration...")
    
    try:
        # Test default config creation
        config = get_default_config()
        
        print("✓ Default config created")
        
        # Test config file loading
        config_path = Path("configs/default.yaml")
        if config_path.exists():
            config = load_config(config_path)
            print("✓ Config loaded from file")
        else:
            print("⚠ Default config file not found, using programmatic config")
        
        # Test config parameters
        assert hasattr(config, 'detection')
        assert hasattr(config, 'classification')
        assert hasattr(config, 'data')
        assert hasattr(config, 'pipeline')
        
        print("✓ Config structure validated")
        
        return True
        
    except Exception as e:
        print(f"✗ Config test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_pdf_processing():
    """Test PDF processing utilities."""
    print("\\nTesting PDF processing...")
    
    try:
        from src.utils.pdf_processing import preprocess_image, normalize_image_for_model
        
        # Create a test image
        test_image = np.ones((400, 600, 3), dtype=np.uint8) * 255  # White image
        
        # Test preprocessing
        processed = preprocess_image(
            test_image,
            apply_deskew=False,  # Skip deskew for synthetic image
            apply_denoise=True
        )
        
        assert processed.shape == test_image.shape
        print("✓ Image preprocessing works")
        
        # Test normalization
        normalized = normalize_image_for_model(test_image)
        assert normalized.dtype == np.float32
        assert 0 <= normalized.max() <= 1
        print("✓ Image normalization works")
        
        return True
        
    except Exception as e:
        print(f"✗ PDF processing test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_detection_model():
    """Test detection model initialization."""
    print("\\nTesting detection model...")
    
    try:
        from src.detection.detector import CheckboxDetector
        from src.utils.config import DetectionConfig
        
        # Create detector with base model
        config = DetectionConfig()
        detector = CheckboxDetector(config)
        
        print("✓ Detection model initialized")
        
        # Test prediction on dummy image
        test_image = np.ones((640, 640, 3), dtype=np.uint8) * 255
        
        # This should work even with untrained model
        results = detector.predict(test_image, confidence=0.1)
        assert isinstance(results, list)
        assert len(results) == 1  # One result for one image
        
        print("✓ Detection inference works")
        
        return True
        
    except Exception as e:
        print(f"✗ Detection model test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_classification_model():
    """Test classification model initialization."""
    print("\\nTesting classification model...")
    
    try:
        from src.classification.classifier import CheckboxClassifier
        from src.utils.config import ClassificationConfig
        
        # Create classifier
        config = ClassificationConfig()
        classifier = CheckboxClassifier(config)
        
        print("✓ Classification model initialized")
        
        # Test prediction on dummy checkbox crop
        test_crop = np.ones((128, 128, 3), dtype=np.uint8) * 255
        
        results = classifier.predict(test_crop)
        assert isinstance(results, list)
        assert len(results) == 1
        assert 'predicted_class' in results[0]
        assert 'class_name' in results[0]
        assert 'confidence' in results[0]
        
        print("✓ Classification inference works")
        
        return True
        
    except Exception as e:
        print(f"✗ Classification model test failed: {str(e)}")
        traceback.print_exc()
        return False


def test_pipeline():
    """Test complete pipeline initialization."""
    print("\\nTesting complete pipeline...")
    
    try:
        from src.pipeline.pipeline import CheckboxPipeline
        from src.utils.config import get_default_config
        
        # Initialize pipeline
        config = get_default_config()
        pipeline = CheckboxPipeline(config)
        
        print("✓ Pipeline initialized")
        
        # Test pipeline info
        info = pipeline.get_pipeline_info()
        assert 'detection_model' in info
        assert 'classification_model' in info
        assert 'pipeline_settings' in info
        
        print("✓ Pipeline info accessible")
        
        return True
        
    except Exception as e:
        print(f"✗ Pipeline test failed: {str(e)}")
        traceback.print_exc()
        return False


def create_test_pdf():
    """Create a simple test PDF for end-to-end testing."""
    print("\\nCreating test PDF...")
    
    try:
        from PIL import Image as PILImage
        import tempfile
        
        # Create test image with checkboxes
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255  # White background
        
        # Draw sample checkboxes
        checkboxes = [
            (100, 100, 30, 30, False),   # Unchecked
            (100, 150, 30, 30, True),    # Checked
            (100, 200, 30, 30, False),   # Unchecked
        ]
        
        for x, y, w, h, checked in checkboxes:
            # Draw checkbox border
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            
            # Draw check mark if checked
            if checked:
                cv2.line(img, (x + 5, y + 15), (x + 12, y + 22), (0, 0, 0), 3)
                cv2.line(img, (x + 12, y + 22), (x + 25, y + 8), (0, 0, 0), 3)
        
        # Add labels
        cv2.putText(img, "Test Form", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Option A", (140, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "Option B", (140, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        cv2.putText(img, "Option C", (140, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
        # Convert to PIL and save as PDF
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        # Create temp directory
        temp_dir = Path("data/sample")
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        test_pdf_path = temp_dir / "test_form.pdf"
        pil_img.save(test_pdf_path, "PDF", resolution=300.0)
        
        print(f"✓ Test PDF created: {test_pdf_path}")
        return test_pdf_path
        
    except Exception as e:
        print(f"✗ Test PDF creation failed: {str(e)}")
        traceback.print_exc()
        return None


def test_end_to_end():
    """Test complete end-to-end pipeline."""
    print("\\nTesting end-to-end pipeline...")
    
    try:
        # Create test PDF
        test_pdf_path = create_test_pdf()
        if test_pdf_path is None:
            return False
        
        from src.pipeline.pipeline import CheckboxPipeline
        from src.utils.config import get_default_config
        
        # Initialize pipeline
        config = get_default_config()
        
        # Lower confidence thresholds for testing with untrained models
        config.pipeline.detection_confidence = 0.01
        config.pipeline.classification_confidence = 0.1
        config.pipeline.confidence_threshold = 0.1
        
        pipeline = CheckboxPipeline(config)
        
        # Process test PDF
        results = pipeline.process_pdf(
            test_pdf_path,
            output_path=test_pdf_path.parent / "test_results.json"
        )
        
        assert 'total_pages' in results
        assert 'total_checkboxes' in results
        assert 'processing_time_seconds' in results
        assert 'pages' in results
        
        print(f"✓ End-to-end processing completed")
        print(f"  Pages processed: {results['total_pages']}")
        print(f"  Processing time: {results['processing_time_seconds']:.2f}s")
        print(f"  Checkboxes found: {results['total_checkboxes']}")
        
        return True
        
    except Exception as e:
        print(f"✗ End-to-end test failed: {str(e)}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("Checkbox Detection Pipeline - Installation Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("PDF Processing Test", test_pdf_processing),
        ("Detection Model Test", test_detection_model),
        ("Classification Model Test", test_classification_model),
        ("Pipeline Test", test_pipeline),
        ("End-to-End Test", test_end_to_end),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\n{test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"✗ {test_name} FAILED")
        except KeyboardInterrupt:
            print("\\nTests interrupted by user")
            break
        except Exception as e:
            print(f"✗ {test_name} FAILED with exception: {str(e)}")
    
    # Summary
    print("\\n" + "=" * 50)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        
        # Additional setup recommendations
        print("\\nNext steps:")
        print("1. Download datasets: python scripts/download_data.py --sample")
        print("2. Test inference: python scripts/infer.py data/sample/test_form.pdf")
        print("3. Check the configs/default.yaml file for configuration options")
        
        return 0
    else:
        print(f"⚠️  {total - passed} test(s) failed. Check error messages above.")
        
        print("\\nTroubleshooting:")
        print("1. Make sure all dependencies are installed: uv sync")
        print("2. Check that you're in the project root directory")
        print("3. Verify CUDA setup if using GPU")
        
        return 1


if __name__ == "__main__":
    sys.exit(main())
