# CheckboxQA Visual Grounding Strategy

## Critical Discovery from Paper Sample

The CheckboxQA dataset **DOES contain visual grounding** between Q&A pairs and actual checkboxes in PDFs!

### Example from Paper:
- **Question**: "Does the company disclose grants exceeding $5K?"
- **Answer**: "No"
- **Visual**: Checkbox with "X" mark in the "No" column (row 21)

## Updated Understanding

### What CheckboxQA Actually Provides:
1. **PDF Documents**: With real checkboxes in forms
2. **Questions**: Natural language queries about checkbox states
3. **Answers**: Ground truth values (Yes/No/specific values)
4. **Implicit Visual Grounding**: Answers correspond to actual checkbox states in PDFs

### The Missing Link:
- CheckboxQA provides **semantic labels** but not **bounding box coordinates**
- We need to:
  1. Download the PDFs 
  2. Use our detection model to find checkboxes
  3. Match detected checkboxes to Q&A ground truth
  4. Create training data with known labels

## Implementation Strategy

### Step 1: Download CheckboxQA PDFs
```python
# Use the document_url_map.json to download PDFs
document_urls = load_json("data/raw/checkboxqa/data/document_url_map.json")
download_pdfs(document_urls)
```

### Step 2: Detect Checkboxes in PDFs
```python
# Use our trained YOLOv12 model
model = YOLO("models/best.pt")
for pdf in checkboxqa_pdfs:
    images = pdf_to_images(pdf)
    for image in images:
        detections = model(image)
        # Extract checkbox crops
```

### Step 3: Match Detections to Q&A Labels
```python
# For each Q&A pair:
qa_pair = {
    "question": "Does the company disclose grants exceeding $5K?",
    "answer": "No"
}

# Find the corresponding checkbox:
# - Use question text to identify form section
# - Match answer to checkbox state
# - Create labeled training example
```

### Step 4: Create Classification Training Data
```python
classification_data = []
for detection in checkboxes:
    label = match_detection_to_qa(detection, qa_pairs)
    classification_data.append({
        "image": detection.crop,
        "label": label,  # checked/unchecked/unclear
        "question_context": qa_pair.question
    })
```

## Validation Strategy

### Use CheckboxQA as End-to-End Test Set:
1. **Input**: PDF document
2. **Detection**: Find all checkboxes with YOLOv12
3. **Classification**: Classify each checkbox state
4. **Validation**: Compare against Q&A ground truth
5. **Metric**: Accuracy of answering questions correctly

### Example Validation:
```python
# Ground Truth from CheckboxQA
question = "Did the organization report more than $5,000 of grants?"
expected_answer = "No"

# Our Pipeline
checkboxes = detect_checkboxes(pdf)
checkbox_21 = find_checkbox_for_question(checkboxes, question)
predicted_state = classify_checkbox(checkbox_21)  # Returns: "unchecked"
predicted_answer = state_to_answer(predicted_state)  # Returns: "No"

# Validate
assert predicted_answer == expected_answer  # Success!
```

## Action Plan

### 1. Download CheckboxQA PDFs
```bash
python download_checkboxqa_pdfs.py
# Downloads 88 PDFs with known checkbox states
```

### 2. Process PDFs Through Detection Pipeline
```python
# Extract all checkbox crops with detection confidence
detected_checkboxes = process_checkboxqa_pdfs()
# Returns: ~1000+ checkbox crops from real documents
```

### 3. Create Labeled Classification Dataset
```python
# Match detections to Q&A labels
labeled_crops = match_detections_to_labels(detected_checkboxes, qa_data)
# Returns: Checkbox images with ground truth labels
```

### 4. Train Classification Model
```python
# Train EfficientNet on real checkbox crops
model = train_classifier(
    synthetic_data + labeled_crops,  # Combined dataset
    validation_set=checkboxqa_test_set
)
```

## Benefits of This Approach

### ✅ Real Document Checkboxes
- Actual checkboxes from government/legal forms
- Diverse styles, fonts, and layouts
- Real-world noise and quality issues

### ✅ Ground Truth Labels
- 755 Q&A pairs provide semantic labels
- Can validate entire pipeline accuracy
- End-to-end performance metrics

### ✅ Domain-Specific Training
- Checkboxes from target domain (forms)
- Improves generalization to similar documents
- Reduces synthetic-to-real domain gap

## Summary

CheckboxQA is **much more valuable** than initially thought:
- Not just Q&A pairs, but **visually grounded** answers
- Provides **real checkbox images** with **semantic labels**
- Perfect for **classification training** and **pipeline validation**
- Addresses the synthetic-to-real domain gap

This transforms our classification training from purely synthetic to **real-world grounded**!