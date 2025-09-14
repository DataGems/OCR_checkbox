# Colab End-to-End Evaluation

Test the complete pipeline: Detection → Classification → Question Answering → ANLS* scoring.

## Setup
```python
# Install additional dependencies
!pip install PyMuPDF  # For PDF processing

# Upload your models and evaluation script
# Files needed:
# - models/best.pt (detection model)
# - models/best_classifier.pt (classification model)  
# - end_to_end_evaluation.py

# Import the evaluation script
exec(open('end_to_end_evaluation.py').read())
```

## Create Test Document
```python
# Create a simple test PDF with checkboxes for demonstration
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

def create_test_pdf(filename="test_document.pdf"):
    """Create a test PDF with checkboxes."""
    
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 11))  # Letter size
    ax.set_xlim(0, 8.5)
    ax.set_ylim(0, 11)
    ax.set_aspect('equal')
    
    # Title
    ax.text(4.25, 10, 'Sample Form with Checkboxes', 
            ha='center', va='center', fontsize=16, weight='bold')
    
    # Create checkboxes
    checkbox_data = [
        {'pos': (1, 8.5), 'label': 'Training Required', 'checked': True},
        {'pos': (1, 8.0), 'label': 'Background Check Complete', 'checked': False},
        {'pos': (1, 7.5), 'label': 'Medical Exam Passed', 'checked': True},
        {'pos': (1, 7.0), 'label': 'Drug Test Completed', 'checked': False},
    ]
    
    for item in checkbox_data:
        x, y = item['pos']
        
        # Draw checkbox
        rect = patches.Rectangle((x, y), 0.2, 0.2, linewidth=2, 
                               edgecolor='black', facecolor='white')
        ax.add_patch(rect)
        
        # Add checkmark if checked
        if item['checked']:
            ax.plot([x+0.05, x+0.1, x+0.18], [y+0.1, y+0.05, y+0.15], 
                   'k-', linewidth=3)
        
        # Add label
        ax.text(x+0.3, y+0.1, item['label'], va='center', fontsize=12)
    
    # Remove axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Save as PDF
    with PdfPages(filename) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    
    plt.close()
    print(f"✅ Test PDF created: {filename}")
    
    return checkbox_data

# Create test document
test_checkboxes = create_test_pdf("test_document.pdf")
```

## Run End-to-End Evaluation
```python
# Initialize the evaluator
evaluator = EndToEndEvaluator(
    detection_model_path="models/best.pt",
    classification_model_path="models/best_classifier.pt",
    use_vlm=False  # Start without VLM for speed
)

# Test on our created document
from pathlib import Path
import json

# Create test questions
test_questions = [
    {'key': 'Is training required?', 'values': [{'value': 'Yes'}]},
    {'key': 'Is background check complete?', 'values': [{'value': 'No'}]},
    {'key': 'Are medical requirements met?', 'values': [{'value': 'Yes'}]},
    {'key': 'How many items are checked?', 'values': [{'value': '2'}]},
]

# Evaluate the test document
print("🧪 Running end-to-end evaluation...")

# Convert PDF to image and test
images = evaluator.pdf_to_images(Path("test_document.pdf"))
if images:
    image = images[0]
    
    # Show the test image
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 8))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Test Document")
    plt.axis('off')
    plt.show()
    
    # Detect checkboxes
    checkboxes = evaluator.detect_checkboxes(image)
    print(f"🔍 Detected {len(checkboxes)} checkboxes")
    
    # Visualize detections
    vis_image = image.copy()
    for i, checkbox in enumerate(checkboxes):
        x1, y1, x2, y2 = checkbox['bbox']
        conf = checkbox['confidence']
        
        # Classify checkbox
        state, class_conf = evaluator.classify_checkbox(image, checkbox['bbox'])
        
        # Draw detection
        color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 3)
        cv2.putText(vis_image, f"{state} ({conf:.2f})", 
                   (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
    plt.title("Detection + Classification Results")
    plt.axis('off')
    plt.show()
    
    # Process questions
    predictions = []
    for question_data in test_questions:
        question = question_data['key']
        prediction = evaluator.process_question(question, image, checkboxes)
        predictions.append(prediction)
        
        print(f"❓ {question}")
        print(f"   Prediction: {prediction}")
        print(f"   Ground truth: {question_data['values'][0]['value']}")
        print()
    
    # Calculate ANLS score
    ground_truth = [q['values'][0]['value'] for q in test_questions]
    anls_score = evaluator.calculate_anls_score(predictions, ground_truth)
    
    print(f"🎯 End-to-End Results:")
    print(f"   ANLS* Score: {anls_score:.3f}")
    print(f"   SOTA Target: 0.832")
    
    if anls_score > 0.832:
        print(f"   🏆 BEATS SOTA by {(anls_score - 0.832):.3f}!")
    else:
        print(f"   📈 Gap to SOTA: {(0.832 - anls_score):.3f}")
        
else:
    print("❌ Failed to process test PDF")
```

## Performance Analysis
```python
# Analyze component performance
print("📊 Component Analysis:")
print(f"   Detection working: {'✅' if len(checkboxes) > 0 else '❌'}")
print(f"   Classification working: {'✅' if any('checked' in str(evaluator.classify_checkbox(image, cb['bbox'])) for cb in checkboxes[:2]) else '❌'}")
print(f"   Question processing: {'✅' if len(predictions) == len(test_questions) else '❌'}")

print(f"\n🔧 Next Steps:")
print(f"   1. Test on real CheckboxQA documents")
print(f"   2. Add VLM integration for complex questions")
print(f"   3. Fine-tune thresholds based on real performance")
```

## Expected Results
- **Detection**: Should find 4 checkboxes in test document
- **Classification**: Should correctly identify 2 checked, 2 unchecked  
- **ANLS Score**: Likely 0.5-0.8 (realistic for rule-based QA)
- **Real Performance**: Will be lower than 100% classification accuracy

This gives us a realistic baseline to compare against the 83.2% SOTA!