"""
Classification module for checkbox state recognition.
Uses EfficientNet to classify checkbox crops as checked/unchecked/unclear.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import timm
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union
import json
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

from ..utils.config import ClassificationConfig


class CheckboxDataset(Dataset):
    """
    Dataset class for checkbox classification.
    """
    
    def __init__(
        self,
        images: List[np.ndarray],
        labels: List[int],
        transform: Optional[A.Compose] = None
    ):
        """
        Initialize dataset.
        
        Args:
            images: List of checkbox crop images
            labels: List of labels (0=unchecked, 1=checked, 2=unclear)
            transform: Albumentations transform pipeline
        """
        self.images = images
        self.labels = labels
        self.transform = transform
        
        assert len(self.images) == len(self.labels), \
            "Number of images and labels must match"
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.images[idx]
        label = self.labels[idx]
        
        # Ensure image is RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Convert BGR to RGB if needed
            if isinstance(image, np.ndarray):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return image, label


class CheckboxClassifier:
    """
    EfficientNet-based checkbox state classifier.
    """
    
    def __init__(self, config: ClassificationConfig, model_path: Optional[str] = None):
        """
        Initialize checkbox classifier.
        
        Args:
            config: Classification configuration
            model_path: Path to trained model weights (optional)
        """
        self.config = config
        self.device = self._setup_device()
        self.model = self._create_model()
        self.criterion = nn.CrossEntropyLoss()
        self.class_names = ['unchecked', 'checked', 'unclear']
        
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
    
    def _setup_device(self) -> torch.device:
        """Setup device configuration."""
        if self.config.device == "cpu":
            return torch.device("cpu")
        
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            print("CUDA not available, using CPU")
            return torch.device("cpu")
    
    def _create_model(self) -> nn.Module:
        """Create EfficientNet model for classification."""
        # Load pretrained EfficientNet
        model = timm.create_model(
            self.config.model_name,
            pretrained=True,
            num_classes=self.config.num_classes
        )
        
        # Move to device
        model = model.to(self.device)
        
        return model
    
    def _get_train_transforms(self) -> A.Compose:
        """Get training data augmentation transforms."""
        return A.Compose([
            A.Resize(self.config.img_size, self.config.img_size),
            A.Rotate(
                limit=self.config.rotation_range,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,
                p=0.5
            ),
            A.RandomBrightnessContrast(
                brightness_limit=self.config.brightness_range,
                contrast_limit=self.config.contrast_range,
                p=0.5
            ),
            A.GaussNoise(
                var_limit=(0, self.config.gaussian_noise_std * 255),
                p=0.3
            ),
            A.HorizontalFlip(p=0.5),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet means
                std=[0.229, 0.224, 0.225]   # ImageNet stds
            ),
            ToTensorV2()
        ])
    
    def _get_val_transforms(self) -> A.Compose:
        """Get validation/inference transforms."""
        return A.Compose([
            A.Resize(self.config.img_size, self.config.img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def train(
        self,
        train_images: List[np.ndarray],
        train_labels: List[int],
        val_images: List[np.ndarray],
        val_labels: List[int],
        output_dir: Union[str, Path] = "models/classification"
    ) -> Dict[str, Any]:
        """
        Train classification model.
        
        Args:
            train_images: Training checkbox crops
            train_labels: Training labels
            val_images: Validation checkbox crops  
            val_labels: Validation labels
            output_dir: Directory to save model checkpoints
            
        Returns:
            Training history dictionary
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create datasets
        train_dataset = CheckboxDataset(
            train_images, train_labels, self._get_train_transforms()
        )
        val_dataset = CheckboxDataset(
            val_images, val_labels, self._get_val_transforms()
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.workers,
            pin_memory=True if self.device.type == 'cuda' else False
        )
        
        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=self.config.patience//2, factor=0.5
        )
        
        # Training loop
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        patience_counter = 0
        
        print(f"Starting training for {self.config.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model: {self.config.model_name}")
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        for epoch in range(self.config.epochs):
            # Training phase
            train_loss, train_acc = self._train_epoch(train_loader, optimizer)
            
            # Validation phase
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Update learning rate
            scheduler.step(val_acc)
            
            # Save metrics
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{self.config.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model(output_dir / "best_model.pth")
                print(f"New best model saved with validation accuracy: {val_acc:.4f}")
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
        
        # Load best model
        self.load_model(output_dir / "best_model.pth")
        
        # Save training history
        with open(output_dir / "training_history.json", 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        
        return history
    
    def _train_epoch(self, train_loader: DataLoader, optimizer: torch.optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(self.device)
            labels = labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def predict(
        self, 
        images: Union[np.ndarray, List[np.ndarray]],
        return_probabilities: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Predict checkbox states for input images.
        
        Args:
            images: Input checkbox crop(s)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            List of prediction dictionaries
        """
        if isinstance(images, np.ndarray):
            images = [images]
        
        self.model.eval()
        results = []
        
        transform = self._get_val_transforms()
        
        with torch.no_grad():
            for image in images:
                # Preprocess image
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                transformed = transform(image=image)
                input_tensor = transformed['image'].unsqueeze(0).to(self.device)
                
                # Get prediction
                outputs = self.model(input_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                predicted_class = outputs.argmax(1).item()
                confidence = probabilities.max(1)[0].item()
                
                result = {
                    'predicted_class': predicted_class,
                    'class_name': self.class_names[predicted_class],
                    'confidence': confidence
                }
                
                if return_probabilities:
                    probs = probabilities.cpu().numpy()[0]
                    result['probabilities'] = {
                        name: float(prob) for name, prob in zip(self.class_names, probs)
                    }
                
                results.append(result)
        
        return results
    
    def predict_batch(
        self,
        images: List[np.ndarray],
        batch_size: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Predict checkbox states for batch of images (more efficient).
        
        Args:
            images: List of checkbox crops
            batch_size: Batch size for prediction (uses config default if None)
            
        Returns:
            List of prediction dictionaries
        """
        batch_size = batch_size or self.config.batch_size
        
        # Create dataset and dataloader
        # Use dummy labels for inference
        dummy_labels = [0] * len(images)
        dataset = CheckboxDataset(images, dummy_labels, self._get_val_transforms())
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )
        
        self.model.eval()
        all_results = []
        
        with torch.no_grad():
            for batch_images, _ in dataloader:
                batch_images = batch_images.to(self.device)
                outputs = self.model(batch_images)
                probabilities = F.softmax(outputs, dim=1)
                
                predicted_classes = outputs.argmax(1).cpu().numpy()
                confidences = probabilities.max(1)[0].cpu().numpy()
                all_probabilities = probabilities.cpu().numpy()
                
                # Process batch results
                for i in range(len(predicted_classes)):
                    result = {
                        'predicted_class': int(predicted_classes[i]),
                        'class_name': self.class_names[predicted_classes[i]],
                        'confidence': float(confidences[i]),
                        'probabilities': {
                            name: float(prob) 
                            for name, prob in zip(self.class_names, all_probabilities[i])
                        }
                    }
                    all_results.append(result)
        
        return all_results
    
    def evaluate(
        self,
        test_images: List[np.ndarray],
        test_labels: List[int]
    ) -> Dict[str, Any]:
        """
        Evaluate model on test dataset.
        
        Args:
            test_images: Test checkbox crops
            test_labels: Test labels
            
        Returns:
            Evaluation metrics dictionary
        """
        test_dataset = CheckboxDataset(
            test_images, test_labels, self._get_val_transforms()
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.config.batch_size, shuffle=False
        )
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device)
                outputs = self.model(images)
                
                _, predicted = outputs.max(1)
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
        
        accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )
        cm = confusion_matrix(all_labels, all_predictions)
        
        # Per-class metrics
        per_class_precision, per_class_recall, per_class_f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None
        )
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist(),
            'per_class_metrics': {
                self.class_names[i]: {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1_score': float(per_class_f1[i])
                }
                for i in range(len(self.class_names))
            }
        }
        
        return metrics
    
    def save_model(self, model_path: Union[str, Path]) -> None:
        """Save model weights."""
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config.__dict__,
            'class_names': self.class_names
        }, model_path)
        
        print(f"Model saved to: {model_path}")
    
    def load_model(self, model_path: Union[str, Path]) -> None:
        """Load model weights."""
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load class names if available
        if 'class_names' in checkpoint:
            self.class_names = checkpoint['class_names']
        
        print(f"Model loaded from: {model_path}")
    
    def export_model(self, export_path: Union[str, Path]) -> None:
        """Export model to ONNX format."""
        self.model.eval()
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create dummy input
        dummy_input = torch.randn(
            1, 3, self.config.img_size, self.config.img_size, 
            device=self.device
        )
        
        # Export to ONNX
        torch.onnx.export(
            self.model,
            dummy_input,
            export_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        
        print(f"Model exported to ONNX: {export_path}")
