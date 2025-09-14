# Colab Detection Test Script

Run this in Google Colab to test our detection model on real CheckboxQA documents.

```python
# 1. Setup
!pip install ultralytics requests PyMuPDF tqdm matplotlib opencv-python

# 2. Upload your trained model
# Upload 'best.pt' to Colab files (drag and drop)

# 3. Run detection test
import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
import fitz  # PyMuPDF

class ColabDetectionTester:
    def __init__(self, model_path="best.pt"):
        print(f"📦 Loading detection model: {model_path}")
        self.model = YOLO(model_path)
        
    def test_sample_document(self, image_url="https://example.com/sample.pdf"):
        """Quick test on a single document."""
        
        # For demo, let's create a simple test image with rectangles
        # (You would replace this with real PDF download/conversion)
        test_image = np.ones((800, 600, 3), dtype=np.uint8) * 255  # White background
        
        # Draw some rectangle shapes to simulate checkboxes
        cv2.rectangle(test_image, (100, 100), (120, 120), (0, 0, 0), 2)  # Empty checkbox
        cv2.rectangle(test_image, (200, 200), (220, 220), (0, 0, 0), -1)  # Filled checkbox
        cv2.rectangle(test_image, (300, 300), (320, 320), (0, 0, 0), 2)   # Empty checkbox
        
        # Run detection
        results = self.model(test_image, conf=0.3)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf
                    })
        
        # Visualize
        vis_image = test_image.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.5 else (0, 0, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, f"{conf:.2f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Detection Test - Found {len(detections)} checkboxes")
        plt.axis('off')
        plt.show()
        
        print(f"🔍 Detection Results:")
        print(f"   Total detections: {len(detections)}")
        if detections:
            confidences = [d['confidence'] for d in detections]
            print(f"   Average confidence: {np.mean(confidences):.3f}")
            print(f"   Min confidence: {np.min(confidences):.3f}")
            print(f"   Max confidence: {np.max(confidences):.3f}")
        
        return detections

# Run the test
tester = ColabDetectionTester("best.pt")
results = tester.test_sample_document()

print("✅ Detection test complete!")
print("Next: Test on real CheckboxQA PDFs when available")
```

## Instructions:
1. Open Google Colab
2. Upload your `best.pt` model file
3. Run the script above
4. This will test basic detection functionality

For real CheckboxQA testing, we'd need to:
- Download actual CheckboxQA PDFs 
- Convert PDFs to images
- Run detection on real documents

Would you like to run this in Colab now?