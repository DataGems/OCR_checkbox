"""
Configuration management for checkbox detection pipeline.
Handles loading/saving of training configs, model parameters, and pipeline settings.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
import json


@dataclass
class DetectionConfig:
    """Configuration for YOLOv12 detection model."""
    model_name: str = "yolo12n.pt"  # Base YOLOv12 nano model
    img_size: int = 640  # Input image size
    batch_size: int = 16
    epochs: int = 100
    learning_rate: float = 0.01
    patience: int = 10  # Early stopping patience
    save_period: int = 10  # Save checkpoint every N epochs
    device: str = "0"  # GPU device
    workers: int = 4  # Number of data loading workers
    
    # Data augmentation
    hsv_h: float = 0.015  # Hue augmentation
    hsv_s: float = 0.7    # Saturation augmentation  
    hsv_v: float = 0.4    # Value augmentation
    degrees: float = 0.0  # Rotation degrees
    translate: float = 0.1  # Translation
    scale: float = 0.5    # Scale
    shear: float = 0.0    # Shear
    perspective: float = 0.0  # Perspective
    flipud: float = 0.0   # Vertical flip probability
    fliplr: float = 0.5   # Horizontal flip probability
    mosaic: float = 1.0   # Mosaic probability
    mixup: float = 0.0    # Mixup probability


@dataclass
class ClassificationConfig:
    """Configuration for checkbox state classification model."""
    model_name: str = "efficientnet_b0"
    num_classes: int = 3  # checked, unchecked, unclear
    img_size: int = 128   # Input size for checkbox crops
    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 7
    device: str = "cuda"
    workers: int = 4
    
    # Data augmentation
    rotation_range: int = 15
    brightness_range: float = 0.2
    contrast_range: float = 0.2
    gaussian_noise_std: float = 0.01


@dataclass
class DataConfig:
    """Configuration for dataset handling."""
    # Paths
    raw_data_dir: str = "data/raw"
    processed_data_dir: str = "data/processed" 
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # PDF processing
    pdf_dpi: int = 300
    apply_deskew: bool = True
    apply_denoise: bool = True
    denoise_sigma: float = 0.5
    
    # Dataset creation
    synthetic_ratio: float = 0.3  # Max 50% per requirements
    min_checkbox_size: int = 10   # Minimum checkbox size in pixels
    max_checkbox_size: int = 100  # Maximum checkbox size in pixels


@dataclass
class PipelineConfig:
    """Configuration for end-to-end pipeline."""
    detection_confidence: float = 0.25  # Minimum detection confidence
    classification_confidence: float = 0.8  # Minimum classification confidence
    nms_iou_threshold: float = 0.45  # Non-max suppression IoU threshold
    max_detections: int = 1000  # Maximum detections per image
    
    # Performance settings
    batch_processing: bool = True
    max_batch_size: int = 8
    enable_quantization: bool = False  # INT8 quantization for speed
    
    # Output settings
    output_format: str = "json"  # json, csv
    include_crops: bool = False  # Whether to save checkbox crops
    confidence_threshold: float = 0.5  # Minimum confidence for output


@dataclass
class Config:
    """Main configuration container."""
    detection: DetectionConfig
    classification: ClassificationConfig  
    data: DataConfig
    pipeline: PipelineConfig
    
    # Global settings
    project_name: str = "checkbox_detection"
    experiment_name: str = "baseline"
    seed: int = 42
    verbose: bool = True


def load_config(config_path: Union[str, Path]) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Loaded configuration object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Create config objects from nested dictionaries
    detection_config = DetectionConfig(**config_dict.get('detection', {}))
    classification_config = ClassificationConfig(**config_dict.get('classification', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    pipeline_config = PipelineConfig(**config_dict.get('pipeline', {}))
    
    # Extract global settings
    global_settings = {k: v for k, v in config_dict.items() 
                      if k not in ['detection', 'classification', 'data', 'pipeline']}
    
    return Config(
        detection=detection_config,
        classification=classification_config,
        data=data_config,
        pipeline=pipeline_config,
        **global_settings
    )


def save_config(config: Config, config_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration object to save
        config_path: Output path for configuration file
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to dictionary
    config_dict = {
        'detection': asdict(config.detection),
        'classification': asdict(config.classification),
        'data': asdict(config.data),
        'pipeline': asdict(config.pipeline),
        'project_name': config.project_name,
        'experiment_name': config.experiment_name,
        'seed': config.seed,
        'verbose': config.verbose
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def get_default_config() -> Config:
    """
    Get default configuration with sensible defaults.
    
    Returns:
        Default configuration object
    """
    return Config(
        detection=DetectionConfig(),
        classification=ClassificationConfig(),
        data=DataConfig(),
        pipeline=PipelineConfig()
    )


def update_config(config: Config, updates: Dict[str, Any]) -> Config:
    """
    Update configuration with new values.
    
    Args:
        config: Base configuration
        updates: Dictionary of updates (supports nested keys like 'detection.batch_size')
        
    Returns:
        Updated configuration
    """
    config_dict = {
        'detection': asdict(config.detection),
        'classification': asdict(config.classification),
        'data': asdict(config.data),
        'pipeline': asdict(config.pipeline),
        'project_name': config.project_name,
        'experiment_name': config.experiment_name,
        'seed': config.seed,
        'verbose': config.verbose
    }
    
    # Apply updates using dot notation
    for key, value in updates.items():
        if '.' in key:
            section, param = key.split('.', 1)
            if section in config_dict and isinstance(config_dict[section], dict):
                config_dict[section][param] = value
        else:
            config_dict[key] = value
    
    # Recreate config objects
    detection_config = DetectionConfig(**config_dict['detection'])
    classification_config = ClassificationConfig(**config_dict['classification'])
    data_config = DataConfig(**config_dict['data'])
    pipeline_config = PipelineConfig(**config_dict['pipeline'])
    
    global_settings = {k: v for k, v in config_dict.items() 
                      if k not in ['detection', 'classification', 'data', 'pipeline']}
    
    return Config(
        detection=detection_config,
        classification=classification_config,
        data=data_config,
        pipeline=pipeline_config,
        **global_settings
    )


def save_results(results: Dict[str, Any], output_path: Union[str, Path]) -> None:
    """
    Save evaluation results to JSON file.
    
    Args:
        results: Results dictionary
        output_path: Output file path
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)


def load_results(results_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load evaluation results from JSON file.
    
    Args:
        results_path: Path to results file
        
    Returns:
        Results dictionary
    """
    results_path = Path(results_path)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        return json.load(f)
