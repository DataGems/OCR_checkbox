#!/usr/bin/env python3
"""
Train EfficientNet-B0 classifier on balanced real+synthetic checkbox dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import cv2
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class CheckboxDataset(Dataset):
    """Dataset for checkbox classification."""
    
    def __init__(self, data_dir: Path, split: str = 'train', transform=None):
        self.data_dir = data_dir / split
        self.transform = transform
        
        # Class mapping
        self.classes = ['checked', 'unchecked', 'unclear']
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Load all samples
        self.samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.png'):
                    self.samples.append({
                        'path': img_path,
                        'label': self.class_to_idx[class_name],
                        'class_name': class_name
                    })
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(str(sample['path']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, sample['label']

class EfficientNetClassifier:
    """EfficientNet-B0 classifier for checkboxes."""
    
    def __init__(self, num_classes=3, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.num_classes = num_classes
        
        # Create model
        self.model = models.efficientnet_b0(pretrained=True)
        # Modify classifier head
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.model.classifier[1].in_features, num_classes)
        )
        
        self.model.to(device)
        
        # Transforms
        self.train_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),  # Larger than 32x32 for better features
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def create_data_loaders(self, data_dir: Path, batch_size=32):
        """Create train/val/test data loaders."""
        
        # Create datasets
        train_dataset = CheckboxDataset(data_dir, 'train', self.train_transform)
        val_dataset = CheckboxDataset(data_dir, 'val', self.val_transform)
        test_dataset = CheckboxDataset(data_dir, 'test', self.val_transform)
        
        # Calculate class weights for imbalanced data
        train_labels = [sample['label'] for sample in train_dataset.samples]
        class_counts = torch.bincount(torch.tensor(train_labels))
        total_samples = len(train_labels)
        class_weights = total_samples / (len(class_counts) * class_counts.float())
        
        print(f"Class weights: {class_weights}")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        return train_loader, val_loader, test_loader, class_weights
    
    def train(self, data_dir: Path, epochs=50, learning_rate=0.001, batch_size=32):
        """Train the classifier."""
        
        print(f"🚀 Training EfficientNet-B0 classifier...")
        print(f"   Device: {self.device}")
        print(f"   Epochs: {epochs}")
        print(f"   Learning rate: {learning_rate}")
        print(f"   Batch size: {batch_size}")
        
        # Create data loaders
        train_loader, val_loader, test_loader, class_weights = self.create_data_loaders(
            data_dir, batch_size
        )
        
        # Setup training
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        best_val_acc = 0.0
        best_model_path = Path("models/best_classifier.pt")
        best_model_path.parent.mkdir(exist_ok=True)
        
        # Training loop
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 30)
            
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in tqdm(train_loader, desc="Training"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            train_acc = 100 * train_correct / train_total
            train_loss = train_loss / len(train_loader)
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc="Validation"):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_acc = 100 * val_correct / val_total
            val_loss = val_loss / len(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['train_acc'].append(train_acc)
            history['val_acc'].append(val_acc)
            
            # Print progress
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_loss': val_loss,
                    'class_names': ['checked', 'unchecked', 'unclear']
                }, best_model_path)
                print(f"💾 New best model saved: {val_acc:.2f}%")
            
            # Learning rate scheduling
            scheduler.step(val_loss)
            
            # Early stopping
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print("Learning rate too low, stopping...")
                break
        
        print(f"\n🎯 Training complete!")
        print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        print(f"   Model saved: {best_model_path}")
        
        # Test evaluation
        print(f"\n🧪 Testing best model...")
        self.load_model(best_model_path)
        test_acc, test_report = self.evaluate(test_loader)
        print(f"Test accuracy: {test_acc:.2f}%")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def load_model(self, model_path: Path):
        """Load trained model."""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from {model_path}")
    
    def evaluate(self, data_loader):
        """Evaluate model on dataset."""
        self.model.eval()
        
        all_predictions = []
        all_labels = []
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in tqdm(data_loader, desc="Evaluating"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = 100 * correct / total
        
        # Generate classification report
        class_names = ['checked', 'unchecked', 'unclear']
        report = classification_report(all_labels, all_predictions, target_names=class_names)
        
        print(f"\nClassification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_predictions)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return accuracy, report
    
    def plot_training_history(self, history):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(history['train_loss'], label='Training Loss')
        ax1.plot(history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Accuracy plot
        ax2.plot(history['train_acc'], label='Training Accuracy')
        ax2.plot(history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def predict(self, image_path: str) -> Tuple[str, float]:
        """Predict single image."""
        self.model.eval()
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.val_transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(image)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        class_names = ['checked', 'unchecked', 'unclear']
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item()
        
        return predicted_class, confidence_score

def train_classifier(data_dir: str = "data/classification",
                    model_type: str = 'efficientnet-b0',
                    epochs: int = 30,
                    batch_size: int = 32,
                    learning_rate: float = 0.001):
    """Main training function."""
    
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"❌ Data directory not found: {data_path}")
        print("   Run generate_classification_dataset.py first")
        return
    
    # Initialize classifier
    classifier = EfficientNetClassifier()
    
    # Train model
    history = classifier.train(
        data_dir=data_path,
        epochs=epochs,
        learning_rate=learning_rate,
        batch_size=batch_size
    )
    
    return classifier, history

def main():
    """Train classification model."""
    classifier, history = train_classifier(
        data_dir="data/classification",
        epochs=30,
        batch_size=32,
        learning_rate=0.001
    )
    
    print("✅ Classification training complete!")

if __name__ == "__main__":
    main()