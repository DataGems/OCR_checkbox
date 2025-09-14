#!/usr/bin/env python3
"""
Main inference script for checkbox detection and classification pipeline.
Usage: python -m src.scripts.infer <pdf_path> [options]
"""

import argparse
import sys
from pathlib import Path
import json

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.config import load_config
from src.pipeline.pipeline import CheckboxPipeline


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(
        description="Run checkbox detection and classification on PDF documents"
    )
    
    # Required arguments
    parser.add_argument(
        "pdf_path",
        type=str,
        help="Path to PDF file to process"
    )
    
    # Optional arguments
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)"
    )
    
    parser.add_argument(
        "--detection-model",
        type=str,
        help="Path to trained detection model weights"
    )
    
    parser.add_argument(
        "--classification-model", 
        type=str,
        help="Path to trained classification model weights"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Output path for results JSON file"
    )
    
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save detection visualization images"
    )
    
    parser.add_argument(
        "--save-crops",
        action="store_true", 
        help="Save individual checkbox crops"
    )
    
    parser.add_argument(
        "--detection-confidence",
        type=float,
        help="Detection confidence threshold (overrides config)"
    )
    
    parser.add_argument(
        "--classification-confidence",
        type=float,
        help="Classification confidence threshold (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for classification (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "0", "1", "2", "3"],
        help="Device to use for inference"
    )
    
    args = parser.parse_args()
    
    try:
        # Load configuration
        print(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        
        # Apply command line overrides
        if args.detection_confidence is not None:
            config.pipeline.detection_confidence = args.detection_confidence
        
        if args.classification_confidence is not None:
            config.pipeline.classification_confidence = args.classification_confidence
            
        if args.batch_size is not None:
            config.classification.batch_size = args.batch_size
            
        if args.device is not None:
            config.detection.device = args.device
            config.classification.device = args.device
        
        # Initialize pipeline
        print("Initializing pipeline...")
        pipeline = CheckboxPipeline(
            config,
            detection_model_path=args.detection_model,
            classification_model_path=args.classification_model
        )
        
        # Print pipeline info
        info = pipeline.get_pipeline_info()
        print("\\nPipeline Configuration:")
        print(f"  Detection: {info['detection_model']['model_name']} on {info['detection_model']['device']}")
        print(f"  Classification: {info['classification_model']['model_name']} on {info['classification_model']['device']}")
        print(f"  Detection confidence: {info['pipeline_settings']['detection_confidence']}")
        print(f"  Classification confidence: {info['pipeline_settings']['classification_confidence']}")
        print()
        
        # Determine output path
        pdf_path = Path(args.pdf_path)
        if args.output:
            output_path = Path(args.output)
        else:
            output_path = pdf_path.parent / f"{pdf_path.stem}_results.json"
        
        # Process PDF
        print(f"Processing PDF: {pdf_path}")
        results = pipeline.process_pdf(
            pdf_path,
            output_path=output_path,
            save_visualizations=args.save_visualizations,
            save_crops=args.save_crops
        )
        
        # Print summary
        print("\\n" + "="*50)
        print("PROCESSING SUMMARY")
        print("="*50)
        print(f"Total pages processed: {results['total_pages']}")
        print(f"Total checkboxes found: {results['total_checkboxes']}")
        print(f"Processing time: {results['processing_time_seconds']:.2f} seconds")
        print(f"State distribution:")
        for state, count in results['state_summary'].items():
            print(f"  {state.capitalize()}: {count}")
        
        print(f"\\nResults saved to: {output_path}")
        
        # Check if processing was within time limit
        if results['processing_time_seconds'] <= 120:  # 2 minutes
            print("✓ Processing completed within 2-minute requirement")
        else:
            print("⚠ Processing exceeded 2-minute requirement")
            
        return 0
        
    except KeyboardInterrupt:
        print("\\nInterrupted by user")
        return 1
        
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
