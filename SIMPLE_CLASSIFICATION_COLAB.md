# 🎯 Simple Classification Training in Colab

## Quick Setup for Checkbox State Classification

### Step 1: Install Dependencies
```python
!pip install torch torchvision efficientnet-pytorch pillow tqdm scikit-learn matplotlib
```

### Step 2: Create Simple Classification Dataset
```python
import os
import random
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

# Create synthetic checkbox dataset
def create_checkbox_dataset(num_samples=1000):
    """Generate synthetic checkbox images with labels."""
    
    os.makedirs('data/classification/train/checked', exist_ok=True)
    os.makedirs('data/classification/train/unchecked', exist_ok=True)
    os.makedirs('data/classification/val/checked', exist_ok=True)
    os.makedirs('data/classification/val/unchecked', exist_ok=True)
    
    for i in range(num_samples):
        # Random parameters
        size = random.randint(32, 64)
        is_checked = random.choice([True, False])
        
        # Create checkbox image
        img = Image.new('RGB', (64, 64), 'white')
        draw = ImageDraw.Draw(img)
        
        # Draw checkbox border
        box_size = size
        x = (64 - box_size) // 2
        y = (64 - box_size) // 2
        draw.rectangle([x, y, x+box_size, y+box_size], outline='black', width=2)
        
        # Draw check mark if checked
        if is_checked:
            # Draw X
            draw.line([x+5, y+5, x+box_size-5, y+box_size-5], fill='black', width=3)
            draw.line([x+box_size-5, y+5, x+5, y+box_size-5], fill='black', width=3)
        
        # Save to appropriate directory
        split = 'train' if i < num_samples * 0.8 else 'val'
        label = 'checked' if is_checked else 'unchecked'
        img.save(f'data/classification/{split}/{label}/checkbox_{i}.png')
    
    print(f"✅ Created {num_samples} synthetic checkboxes")

# Generate dataset
create_checkbox_dataset(2000)
```

### Step 3: Load CheckboxQA Labels (Optional Enhancement)
```python
import json

# Load CheckboxQA semantic labels we extracted
with open('checkboxqa_labels.json', 'r') as f:
    checkboxqa_labels = json.load(f)

# Count label distribution
label_counts = {'checked': 0, 'unchecked': 0, 'unclear': 0}
for item in checkboxqa_labels:
    label_counts[item['checkbox_state']] += 1

print(f"CheckboxQA label distribution: {label_counts}")
# Can use these to weight the training data
```

### Step 4: Train EfficientNet Classifier
```python
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torch.optim import Adam

# Simple dataset class
class CheckboxDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['checked', 'unchecked']
        self.images = []
        self.labels = []
        
        for i, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                self.images.append(os.path.join(class_dir, img_name))
                self.labels.append(i)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data transforms
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# Create datasets
train_dataset = CheckboxDataset('data/classification/train', transform)
val_dataset = CheckboxDataset('data/classification/val', transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Create model
model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
model = model.cuda()

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

# Training loop
print("🚀 Starting training...")
for epoch in range(10):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        images, labels = images.cuda(), labels.cuda()
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    # Validation
    model.eval()
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()
    
    print(f'Epoch {epoch+1}/10: Train Acc: {100.*correct/total:.2f}%, Val Acc: {100.*val_correct/val_total:.2f}%')

# Save model
torch.save(model.state_dict(), 'checkbox_classifier.pth')
print("✅ Model saved!")
```

### Step 5: Test the Classifier
```python
# Test on sample images
model.eval()
test_transform = transform

def classify_checkbox(image_path):
    image = Image.open(image_path).convert('RGB')
    image_tensor = test_transform(image).unsqueeze(0).cuda()
    
    with torch.no_grad():
        output = model(image_tensor)
        _, predicted = output.max(1)
        
    classes = ['checked', 'unchecked']
    confidence = torch.softmax(output, dim=1)[0][predicted].item()
    
    return classes[predicted.item()], confidence

# Test
result, conf = classify_checkbox('test_checkbox.png')
print(f"Classification: {result} (confidence: {conf:.2%})")
```

### Step 6: Download Trained Model
```python
from google.colab import files
files.download('checkbox_classifier.pth')
```

## Expected Results
- **Training Time**: ~5 minutes for 10 epochs
- **Accuracy**: >95% on synthetic data
- **Model Size**: ~16MB (EfficientNet-B0)

## Next Steps
1. Process real CheckboxQA PDFs to get actual checkbox crops
2. Fine-tune on real data for better generalization
3. Add "unclear" class for ambiguous checkboxes
4. Integrate with detection pipeline

This simple approach gets you a working classifier quickly!