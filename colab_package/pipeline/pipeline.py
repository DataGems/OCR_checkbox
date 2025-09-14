"""
End-to-end pipeline for checkbox detection and classification in PDF documents.
Integrates YOLO detection and EfficientNet classification for complete processing.
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import numpy as np
import cv2

from ..detection.detector import CheckboxDetector
from ..classification.classifier import CheckboxClassifier
from ..utils.pdf_processing import (
    pdf_to_images, preprocess_image, normalize_image_for_model
)
from ..utils.config import Config


class CheckboxPipeline:
    """
    Complete pipeline for checkbox detection and classification in PDFs.
    """
    
    def __init__(
        self,
        config: Config,
        detection_model_path: Optional[str] = None,
        classification_model_path: Optional[str] = None
    ):
        """
        Initialize checkbox pipeline.
        
        Args:
            config: Pipeline configuration
            detection_model_path: Path to trained detection model
            classification_model_path: Path to trained classification model
        """
        self.config = config
        
        # Initialize detection and classification models
        self.detector = CheckboxDetector(
            config.detection, 
            detection_model_path
        )
        self.classifier = CheckboxClassifier(
            config.classification,
            classification_model_path
        )
        
        print("Checkbox pipeline initialized")
        print(f"Detection model: {detection_model_path or 'base model'}")
        print(f"Classification model: {classification_model_path or 'untrained'}")
    
    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        save_visualizations: bool = False,
        save_crops: bool = False
    ) -> Dict[str, Any]:
        """
        Process a complete PDF document for checkbox detection and classification.
        
        Args:
            pdf_path: Path to PDF file
            output_path: Output path for results JSON (optional)
            save_visualizations: Whether to save detection visualizations
            save_crops: Whether to save individual checkbox crops
            
        Returns:
            Complete processing results dictionary
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        print(f"Processing PDF: {pdf_path}")
        start_time = time.time()
        
        # Convert PDF to images
        print("Converting PDF to images...")
        try:
            pages = pdf_to_images(
                pdf_path, 
                dpi=self.config.data.pdf_dpi
            )
        except Exception as e:
            raise RuntimeError(f"Failed to convert PDF to images: {str(e)}")
        
        print(f"Converted {len(pages)} pages")
        
        # Process each page
        all_results = []
        total_checkboxes = 0
        
        for page_idx, page_image in enumerate(pages):
            print(f"Processing page {page_idx + 1}/{len(pages)}")
            
            # Preprocess page
            processed_page = preprocess_image(
                page_image,
                apply_deskew=self.config.data.apply_deskew,
                apply_denoise=self.config.data.apply_denoise,
                denoise_sigma=self.config.data.denoise_sigma
            )
            
            # Detect checkboxes
            detection_results = self.detector.predict(
                processed_page,
                confidence=self.config.pipeline.detection_confidence,
                iou_threshold=self.config.pipeline.nms_iou_threshold,
                max_detections=self.config.pipeline.max_detections
            )[0]  # Get first (only) result
            
            detections = detection_results['detections']
            print(f"Found {len(detections)} checkboxes on page {page_idx + 1}")
            
            if not detections:
                # No checkboxes found on this page
                page_result = {
                    'page': page_idx + 1,
                    'image_shape': detection_results['image_shape'],
                    'detections': [],
                    'processing_time': detection_results['inference_time']
                }
                all_results.append(page_result)
                continue
            
            # Extract checkbox crops
            crops = self.detector.extract_crops(
                processed_page,
                detections,
                crop_size=(self.config.classification.img_size, 
                          self.config.classification.img_size),
                padding=5
            )
            
            # Classify checkbox states
            if crops:
                if self.config.pipeline.batch_processing and len(crops) > 1:
                    # Use batch prediction for efficiency
                    classifications = self.classifier.predict_batch(
                        crops,
                        batch_size=self.config.pipeline.max_batch_size
                    )
                else:
                    # Single predictions
                    classifications = []
                    for crop in crops:
                        classification = self.classifier.predict(crop)[0]
                        classifications.append(classification)
            else:
                classifications = []
            
            # Combine detection and classification results
            page_checkboxes = []
            for i, (detection, classification) in enumerate(zip(detections, classifications)):
                # Apply confidence thresholding
                if classification['confidence'] < self.config.pipeline.classification_confidence:
                    continue
                
                checkbox_result = {
                    'page': page_idx + 1,
                    'checkbox_id': f"page_{page_idx+1}_cbx_{i+1:03d}",
                    'coordinates': detection['bbox'],  # [x1, y1, x2, y2]
                    'detection_confidence': detection['confidence'],
                    'state': classification['class_name'],
                    'classification_confidence': classification['confidence'],
                    'state_probabilities': classification.get('probabilities', {}),
                    'area': detection['area'],
                    'center': detection['center']
                }
                
                # Apply final confidence threshold
                if classification['confidence'] >= self.config.pipeline.confidence_threshold:
                    page_checkboxes.append(checkbox_result)
            
            total_checkboxes += len(page_checkboxes)
            
            # Save page results
            page_result = {
                'page': page_idx + 1,
                'image_shape': detection_results['image_shape'],
                'detections': page_checkboxes,
                'processing_time': detection_results['inference_time'],
                'total_detected': len(detections),
                'total_classified': len(page_checkboxes)
            }
            all_results.append(page_result)
            
            # Save visualizations if requested
            if save_visualizations:
                vis_image = self.detector.visualize_detections(
                    processed_page, 
                    detections,
                    confidence_threshold=self.config.pipeline.detection_confidence
                )
                vis_path = pdf_path.parent / f"{pdf_path.stem}_page_{page_idx+1}_detections.jpg"
                cv2.imwrite(str(vis_path), vis_image)
            
            # Save crops if requested
            if save_crops and crops:
                crops_dir = pdf_path.parent / f"{pdf_path.stem}_crops"
                crops_dir.mkdir(exist_ok=True)
                
                for i, (crop, classification) in enumerate(zip(crops, classifications)):
                    crop_filename = f"page_{page_idx+1}_cbx_{i+1:03d}_{classification['class_name']}.jpg"
                    crop_path = crops_dir / crop_filename
                    cv2.imwrite(str(crop_path), crop)
        
        # Calculate summary statistics
        total_time = time.time() - start_time
        
        # Count states
        state_counts = {'checked': 0, 'unchecked': 0, 'unclear': 0}
        for page_result in all_results:
            for checkbox in page_result['detections']:
                state_counts[checkbox['state']] += 1
        
        # Create final results
        results = {
            'pdf_path': str(pdf_path),
            'processing_time_seconds': round(total_time, 2),
            'total_pages': len(pages),
            'total_checkboxes': total_checkboxes,
            'state_summary': state_counts,
            'pages': all_results,
            'pipeline_config': {
                'detection_confidence': self.config.pipeline.detection_confidence,
                'classification_confidence': self.config.pipeline.classification_confidence,
                'confidence_threshold': self.config.pipeline.confidence_threshold,
                'pdf_dpi': self.config.data.pdf_dpi
            }
        }
        
        print(f"Processing completed in {total_time:.2f} seconds")
        print(f"Total checkboxes found: {total_checkboxes}")
        print(f"State summary: {state_counts}")
        
        # Save results if output path specified
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            if self.config.pipeline.output_format.lower() == 'json':
                self._save_json_results(results, output_path)
            elif self.config.pipeline.output_format.lower() == 'csv':
                self._save_csv_results(results, output_path)
        
        return results
    
    def _save_json_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save results in JSON format (matches requirement specification)."""
        # Convert to required JSON format
        json_results = []
        
        for page_result in results['pages']:
            for checkbox in page_result['detections']:
                json_entry = {
                    'page': checkbox['page'],
                    'checkbox_id': checkbox['checkbox_id'],
                    'coordinates': checkbox['coordinates'],
                    'state': 'checked' if checkbox['state'] == 'checked' else 'unchecked',  # Simplified states
                    'confidence': checkbox['classification_confidence']
                }
                json_results.append(json_entry)
        
        # Save both the required format and detailed results
        required_format_path = output_path.with_suffix('.json')
        with open(required_format_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        detailed_results_path = output_path.with_suffix('_detailed.json')
        with open(detailed_results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"Results saved to: {required_format_path}")
        print(f"Detailed results saved to: {detailed_results_path}")
    
    def _save_csv_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save results in CSV format."""
        import pandas as pd
        
        rows = []
        for page_result in results['pages']:
            for checkbox in page_result['detections']:
                row = {
                    'pdf_path': results['pdf_path'],
                    'page': checkbox['page'],
                    'checkbox_id': checkbox['checkbox_id'],
                    'x1': checkbox['coordinates'][0],
                    'y1': checkbox['coordinates'][1], 
                    'x2': checkbox['coordinates'][2],
                    'y2': checkbox['coordinates'][3],
                    'state': checkbox['state'],
                    'detection_confidence': checkbox['detection_confidence'],
                    'classification_confidence': checkbox['classification_confidence'],
                    'area': checkbox['area'],
                    'center_x': checkbox['center'][0],
                    'center_y': checkbox['center'][1]
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = output_path.with_suffix('.csv')
        df.to_csv(csv_path, index=False)
        print(f"CSV results saved to: {csv_path}")
    
    def process_batch(
        self,
        pdf_paths: List[Union[str, Path]],
        output_dir: Union[str, Path],
        parallel: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDFs in batch.
        
        Args:
            pdf_paths: List of PDF file paths
            output_dir: Output directory for results
            parallel: Whether to use parallel processing (not implemented yet)
            
        Returns:
            List of processing results for each PDF
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        all_results = []
        
        for i, pdf_path in enumerate(pdf_paths):
            print(f"Processing PDF {i+1}/{len(pdf_paths)}: {pdf_path}")
            
            try:
                pdf_path = Path(pdf_path)
                output_path = output_dir / f"{pdf_path.stem}_results.json"
                
                result = self.process_pdf(
                    pdf_path,
                    output_path=output_path,
                    save_visualizations=self.config.pipeline.include_crops,
                    save_crops=self.config.pipeline.include_crops
                )
                
                result['status'] = 'success'
                all_results.append(result)
                
            except Exception as e:
                error_result = {
                    'pdf_path': str(pdf_path),
                    'status': 'error',
                    'error_message': str(e),
                    'total_checkboxes': 0,
                    'processing_time_seconds': 0
                }
                all_results.append(error_result)
                print(f"Error processing {pdf_path}: {str(e)}")
        
        # Save batch summary
        batch_summary = {
            'total_pdfs': len(pdf_paths),
            'successful': sum(1 for r in all_results if r['status'] == 'success'),
            'failed': sum(1 for r in all_results if r['status'] == 'error'),
            'total_checkboxes': sum(r['total_checkboxes'] for r in all_results),
            'total_processing_time': sum(r['processing_time_seconds'] for r in all_results),
            'results': all_results
        }
        
        summary_path = output_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(batch_summary, f, indent=2, default=str)
        
        print(f"Batch processing completed. Summary saved to: {summary_path}")
        return all_results
    
    def evaluate_pipeline(
        self,
        test_pdfs: List[Union[str, Path]],
        ground_truth: Dict[str, Any],
        output_dir: Union[str, Path]
    ) -> Dict[str, Any]:
        """
        Evaluate complete pipeline performance against ground truth.
        
        Args:
            test_pdfs: List of test PDF paths
            ground_truth: Ground truth annotations
            output_dir: Output directory for evaluation results
            
        Returns:
            Evaluation metrics dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process test PDFs
        results = self.process_batch(
            test_pdfs,
            output_dir / "predictions"
        )
        
        # Calculate evaluation metrics
        # This would need to be implemented based on specific evaluation requirements
        # For now, return basic statistics
        
        successful_results = [r for r in results if r['status'] == 'success']
        
        metrics = {
            'total_pdfs': len(test_pdfs),
            'successful_processing': len(successful_results),
            'processing_rate': len(successful_results) / len(test_pdfs),
            'average_processing_time': np.mean([r['processing_time_seconds'] for r in successful_results]) if successful_results else 0,
            'total_checkboxes_detected': sum(r['total_checkboxes'] for r in successful_results),
            'average_checkboxes_per_pdf': np.mean([r['total_checkboxes'] for r in successful_results]) if successful_results else 0
        }
        
        # Save evaluation results
        eval_path = output_dir / "evaluation_metrics.json"
        with open(eval_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        
        print(f"Pipeline evaluation completed. Metrics saved to: {eval_path}")
        return metrics
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the current pipeline configuration."""
        return {
            'detection_model': {
                'model_name': self.config.detection.model_name,
                'input_size': self.config.detection.img_size,
                'device': str(self.detector.device)
            },
            'classification_model': {
                'model_name': self.config.classification.model_name,
                'input_size': self.config.classification.img_size,
                'num_classes': self.config.classification.num_classes,
                'class_names': self.classifier.class_names,
                'device': str(self.classifier.device)
            },
            'pipeline_settings': {
                'detection_confidence': self.config.pipeline.detection_confidence,
                'classification_confidence': self.config.pipeline.classification_confidence,
                'final_confidence_threshold': self.config.pipeline.confidence_threshold,
                'batch_processing': self.config.pipeline.batch_processing,
                'max_batch_size': self.config.pipeline.max_batch_size
            },
            'preprocessing': {
                'pdf_dpi': self.config.data.pdf_dpi,
                'apply_deskew': self.config.data.apply_deskew,
                'apply_denoise': self.config.data.apply_denoise,
                'denoise_sigma': self.config.data.denoise_sigma
            }
        }
