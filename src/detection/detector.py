"""
YOLO-based checkbox detection module.
Handles training, inference, and evaluation of YOLOv12 models for checkbox localization.
Leverages attention mechanisms and advanced architecture for improved small object detection.
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
from ultralytics import YOLO
import yaml

from ..utils.config import DetectionConfig


class CheckboxDetector:
    """
    YOLOv12-based checkbox detector for document images.
    Features attention mechanisms and advanced architecture for superior small object detection.
    """
    
    def __init__(self, config: DetectionConfig, model_path: Optional[str] = None):
        """
        Initialize checkbox detector.
        
        Args:
            config: Detection configuration
            model_path: Path to trained model weights (optional)
        """
        self.config = config
        self.model = None
        self.device = self._setup_device()
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            # Try to load trained model first, fallback to base model
            trained_model = Path("models/best.pt")
            if trained_model.exists():
                print(f"✅ Loading trained model: {trained_model}")
                self.load_model(str(trained_model))
            else:
                # Load base model for training
                print(f"📦 Loading base model: {config.model_name}")
                self.model = YOLO(config.model_name)
    
    def _setup_device(self) -> str:
        """Setup and validate device configuration."""
        if self.config.device == "cpu":
            return "cpu"
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            return "cpu"
        
        # Validate GPU device
        try:
            device_id = int(self.config.device)
            if device_id >= torch.cuda.device_count():
                print(f"GPU {device_id} not available, using GPU 0")
                return "0"
            return self.config.device
        except ValueError:
            return "0" if torch.cuda.is_available() else "cpu"
    
    def train(
        self, 
        dataset_config_path: Union[str, Path],
        output_dir: Union[str, Path] = "models/detection"
    ) -> Dict[str, Any]:
        """
        Train YOLOv12 model on checkbox detection dataset.
        
        Args:
            dataset_config_path: Path to YOLO dataset configuration file
            output_dir: Directory to save trained models
            
        Returns:
            Training results dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        train_args = {
            'data': str(dataset_config_path),
            'epochs': self.config.epochs,
            'imgsz': self.config.img_size,
            'batch': self.config.batch_size,
            'lr0': self.config.learning_rate,
            'patience': self.config.patience,
            'save_period': self.config.save_period,
            'device': self.device,
            'workers': self.config.workers,
            'project': str(output_dir.parent),
            'name': output_dir.name,
            'exist_ok': True,
            'verbose': True,
            
            # Augmentation parameters
            'hsv_h': self.config.hsv_h,
            'hsv_s': self.config.hsv_s,
            'hsv_v': self.config.hsv_v,
            'degrees': self.config.degrees,
            'translate': self.config.translate,
            'scale': self.config.scale,
            'shear': self.config.shear,
            'perspective': self.config.perspective,
            'flipud': self.config.flipud,
            'fliplr': self.config.fliplr,
            'mosaic': self.config.mosaic,
            'mixup': self.config.mixup,
        }
        
        print(f"Starting YOLOv12 training with config: {self.config.model_name}")
        print(f"Device: {self.device}")
        print(f"Image size: {self.config.img_size}")
        print(f"Batch size: {self.config.batch_size}")
        
        # Start training
        results = self.model.train(**train_args)
        
        # Save final model
        best_model_path = output_dir / "weights" / "best.pt"
        if best_model_path.exists():
            self.model = YOLO(str(best_model_path))
            print(f"Training completed. Best model saved to: {best_model_path}")
        
        return results
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """
        Load trained model weights.
        
        Args:
            model_path: Path to model weights file
        """
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = YOLO(str(model_path))
        print(f"Loaded model from: {model_path}")
    
    def predict(
        self,
        images: Union[np.ndarray, List[np.ndarray], str, Path],
        confidence: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        max_detections: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Run inference on images to detect checkboxes.
        
        Args:
            images: Input image(s) - single image, list of images, or path
            confidence: Confidence threshold (uses config default if None)
            iou_threshold: IoU threshold for NMS (uses config default if None)
            max_detections: Maximum detections per image (uses config default if None)
            
        Returns:
            List of detection results for each image
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Train a model or load pretrained weights.")
        
        # Use config defaults if not specified
        conf_threshold = confidence or 0.25
        iou_threshold = iou_threshold or 0.45
        max_det = max_detections or 1000
        
        # Run inference
        results = self.model.predict(
            images,
            conf=conf_threshold,
            iou=iou_threshold,
            max_det=max_det,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        parsed_results = []
        for result in results:
            detections = self._parse_yolo_result(result)
            parsed_results.append(detections)
        
        return parsed_results
    
    def _parse_yolo_result(self, result) -> Dict[str, Any]:
        """
        Parse YOLOv12 result object into standardized format.
        
        Args:
            result: YOLOv8 result object
            
        Returns:
            Parsed detection dictionary
        """
        detections = []
        
        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()
            
            for i, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                x1, y1, x2, y2 = box.astype(int)
                
                detection = {
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(conf),
                    'class_id': int(cls),
                    'class_name': 'checkbox',  # Single class for detection
                    'area': (x2 - x1) * (y2 - y1),
                    'center': [(x1 + x2) // 2, (y1 + y2) // 2]
                }
                detections.append(detection)
        
        return {
            'detections': detections,
            'image_shape': result.orig_shape,
            'inference_time': result.speed,  # Contains preprocess, inference, postprocess times
            'num_detections': len(detections)
        }
    
    def extract_crops(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        crop_size: Tuple[int, int] = (128, 128),
        padding: int = 5
    ) -> List[np.ndarray]:
        """
        Extract checkbox crops from detections.
        
        Args:
            image: Source image
            detections: List of detection dictionaries
            crop_size: Target size for crops (width, height)
            padding: Padding around detected boxes
            
        Returns:
            List of cropped checkbox images
        """
        crops = []
        h, w = image.shape[:2]
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            
            # Add padding
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding) 
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # Extract crop
            crop = image[y1:y2, x1:x2]
            
            if crop.size > 0:
                # Resize to target size
                crop_resized = cv2.resize(crop, crop_size, interpolation=cv2.INTER_CUBIC)
                crops.append(crop_resized)
        
        return crops
    
    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Dict[str, Any]],
        confidence_threshold: float = 0.5
    ) -> np.ndarray:
        """
        Visualize detections on image.
        
        Args:
            image: Source image
            detections: List of detection dictionaries
            confidence_threshold: Minimum confidence to display
            
        Returns:
            Image with detection visualizations
        """
        vis_image = image.copy()
        
        for detection in detections:
            if detection['confidence'] < confidence_threshold:
                continue
                
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"checkbox: {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                vis_image, 
                (x1, y1 - label_size[1] - 10),
                (x1 + label_size[0], y1),
                (0, 255, 0), 
                -1
            )
            cv2.putText(
                vis_image, 
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        return vis_image
    
    def evaluate(self, dataset_config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Evaluate model on validation dataset.
        
        Args:
            dataset_config_path: Path to YOLO dataset configuration
            
        Returns:
            Evaluation metrics dictionary
        """
        if self.model is None:
            raise RuntimeError("No model loaded for evaluation.")
        
        # Run validation
        metrics = self.model.val(
            data=str(dataset_config_path),
            device=self.device,
            verbose=True
        )
        
        return {
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'f1_score': float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr)) 
                       if (metrics.box.mp + metrics.box.mr) > 0 else 0.0
        }
    
    def export_model(
        self, 
        export_path: Union[str, Path],
        format: str = "onnx"
    ) -> Path:
        """
        Export model to different formats for deployment.
        
        Args:
            export_path: Output path for exported model
            format: Export format (onnx, torchscript, tflite, etc.)
            
        Returns:
            Path to exported model
        """
        if self.model is None:
            raise RuntimeError("No model loaded for export.")
        
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Export model
        exported_path = self.model.export(
            format=format,
            dynamic=True,  # Dynamic input shapes
            simplify=True  # Simplify ONNX model
        )
        
        print(f"Model exported to: {exported_path}")
        return Path(exported_path)


def create_yolo_dataset_config(
    train_images_dir: Union[str, Path],
    val_images_dir: Union[str, Path],
    train_labels_dir: Union[str, Path],
    val_labels_dir: Union[str, Path],
    output_path: Union[str, Path],
    class_names: List[str] = ["checkbox"]
) -> Path:
    """
    Create YOLO dataset configuration file.
    
    Args:
        train_images_dir: Training images directory
        val_images_dir: Validation images directory  
        train_labels_dir: Training labels directory
        val_labels_dir: Validation labels directory
        output_path: Output path for dataset config
        class_names: List of class names
        
    Returns:
        Path to created config file
    """
    config = {
        'train': str(Path(train_images_dir).absolute()),
        'val': str(Path(val_images_dir).absolute()),
        'nc': len(class_names),
        'names': class_names
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"YOLO dataset config created: {output_path}")
    return output_path
