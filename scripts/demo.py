#!/usr/bin/env python3
"""
Quick start script demonstrating the complete checkbox detection pipeline.
This script shows how to:
1. Initialize the system with default settings
2. Process a PDF document
3. Analyze the results

Run this after setting up the environment and downloading sample data.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.config import get_default_config, load_config
from src.pipeline.pipeline import CheckboxPipeline


def demo_basic_usage():
    """Demonstrate basic usage of the checkbox detection pipeline."""
    print("🚀 Checkbox Detection Pipeline Demo")
    print("=" * 50)
    
    # Step 1: Load configuration
    print("\\n📋 Step 1: Loading configuration...")
    try:
        config = load_config("configs/default.yaml")
        print("✓ Configuration loaded from file")
    except:
        config = get_default_config()
        print("✓ Using default configuration")
    
    # Display key settings
    print(f"   Detection model: {config.detection.model_name}")
    print(f"   Classification model: {config.classification.model_name}")
    print(f"   Detection confidence: {config.pipeline.detection_confidence}")
    print(f"   Classification confidence: {config.pipeline.classification_confidence}")
    
    # Step 2: Initialize pipeline
    print("\\n🔧 Step 2: Initializing pipeline...")
    pipeline = CheckboxPipeline(config)
    
    # Display pipeline info
    info = pipeline.get_pipeline_info()
    print("✓ Pipeline initialized")
    print(f"   Detection device: {info['detection_model']['device']}")
    print(f"   Classification device: {info['classification_model']['device']}")
    
    # Step 3: Check for sample data
    print("\\n📁 Step 3: Looking for sample data...")
    sample_pdf = Path("data/sample/test_form.pdf")
    
    if not sample_pdf.exists():
        print("⚠️  Sample PDF not found. Creating one...")
        try:
            from scripts.test_installation import create_test_pdf
            sample_pdf = create_test_pdf()
            if sample_pdf is None:
                print("❌ Failed to create sample PDF")
                return False
        except Exception as e:
            print(f"❌ Error creating sample PDF: {e}")
            return False
    
    print(f"✓ Sample PDF found: {sample_pdf}")
    
    # Step 4: Process the PDF
    print("\\n⚙️  Step 4: Processing PDF...")
    print("   This may take a moment...")
    
    try:
        # Set output path
        output_path = sample_pdf.parent / "demo_results.json"
        
        # Process with visualizations for demo
        results = pipeline.process_pdf(
            sample_pdf,
            output_path=output_path,
            save_visualizations=True,
            save_crops=True
        )
        
        print("✓ Processing completed!")
        
    except Exception as e:
        print(f"❌ Processing failed: {e}")
        return False
    
    # Step 5: Analyze results
    print("\\n📊 Step 5: Analyzing results...")
    print(f"   Total pages: {results['total_pages']}")
    print(f"   Processing time: {results['processing_time_seconds']:.2f} seconds")
    print(f"   Total checkboxes found: {results['total_checkboxes']}")
    
    if results['total_checkboxes'] > 0:
        print("\\n   Checkbox states:")
        for state, count in results['state_summary'].items():
            print(f"     {state.capitalize()}: {count}")
        
        print("\\n   Detailed results:")
        for page in results['pages']:
            if page['detections']:
                print(f"     Page {page['page']}: {len(page['detections'])} checkboxes")
                for checkbox in page['detections'][:3]:  # Show first 3
                    print(f"       - {checkbox['state']} (conf: {checkbox['classification_confidence']:.2f})")
                if len(page['detections']) > 3:
                    print(f"       ... and {len(page['detections']) - 3} more")
    else:
        print("   No checkboxes detected (this is expected with untrained models)")
    
    print(f"\\n💾 Results saved to: {output_path}")
    
    # Step 6: Performance check
    print("\\n⏱️  Step 6: Performance check...")
    if results['processing_time_seconds'] <= 120:
        print("✅ Processing completed within 2-minute requirement")
    else:
        print("⚠️  Processing exceeded 2-minute requirement")
        print("   Consider optimizing settings or using GPU acceleration")
    
    return True


def show_next_steps():
    """Show next steps for users."""
    print("\\n🎯 Next Steps:")
    print("=" * 50)
    print("1. 📚 Download real datasets:")
    print("   python scripts/download_data.py --all")
    print()
    print("2. 🎯 Process your own PDF:")
    print("   python scripts/infer.py your_document.pdf")
    print()
    print("3. ⚙️  Customize configuration:")
    print("   Edit configs/default.yaml")
    print()
    print("4. 🚀 Train custom models:")
    print("   - Prepare your training data")
    print("   - Train detection model with your checkbox annotations")
    print("   - Train classification model with labeled checkbox crops")
    print()
    print("5. 📈 Evaluate performance:")
    print("   - Test on validation datasets")
    print("   - Measure mAP for detection")
    print("   - Measure accuracy for classification")
    print()
    print("📖 For detailed documentation, see:")
    print("   - README.md for setup and usage")
    print("   - docs/approach.md for technical details")
    print("   - configs/default.yaml for all configuration options")


def main():
    """Main demo function."""
    try:
        # Run basic demo
        success = demo_basic_usage()
        
        if success:
            show_next_steps()
            print("\\n🎉 Demo completed successfully!")
            return 0
        else:
            print("\\n❌ Demo failed. Check error messages above.")
            print("\\n🔧 Troubleshooting:")
            print("1. Make sure environment is activated: source .venv/bin/activate")
            print("2. Install dependencies: uv sync")
            print("3. Run installation test: python scripts/test_installation.py")
            return 1
            
    except KeyboardInterrupt:
        print("\\n\\n⏹️  Demo interrupted by user")
        return 1
    except Exception as e:
        print(f"\\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
