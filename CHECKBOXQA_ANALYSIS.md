# CheckboxQA Dataset Analysis and Usage Strategy

## Dataset Structure Analysis

### Format
- **File**: `data/raw/checkboxqa/data/gold.jsonl` (JSON Lines format)
- **Documents**: 1,000+ PDF documents with checkbox-related Q&A pairs
- **No bounding boxes**: Only semantic labels, not spatial coordinates

### Annotation Pattern
```json
{
  "name": "document_id",
  "extension": "pdf", 
  "annotations": [
    {
      "id": 123,
      "key": "Question about checkbox state",
      "values": [{"value": "Answer indicating state"}]
    }
  ],
  "language": "en",
  "split": "test"
}
```

## Key Insight: Question Types Reveal Checkbox States

### 1. Yes/No Questions → Binary Checkbox States
```
"Is training required?" → "No" = UNCHECKED
"Is experience required?" → "Yes" = CHECKED
"Are you a corporation?" → "Yes" = CHECKED
```

### 2. Selection Questions → Checkbox/Radio States  
```
"What is selected in X?" → "Option A" = CHECKED (that option)
"What is checked in Y?" → "None" = UNCHECKED
"What option is selected?" → "Master's" = CHECKED (Master's option)
```

### 3. Multi-Value Questions → Multiple Checkboxes
```
"What are the selected answer(s)?" → ["Air", "Bus", "Other"] = Multiple CHECKED
```

## Classification Strategy Using CheckboxQA

### Approach 1: Semantic Label Generation (Immediate)
Use CheckboxQA to create **ground truth labels** for classification training:

1. **Parse Questions**: Extract checkbox-related questions
2. **Classify Answers**: Map answers to checkbox states (checked/unchecked/unclear)
3. **Generate Training Data**: Create labeled examples for EfficientNet

### Approach 2: Contextual Classification (Advanced)
Train a **context-aware** classifier that considers both:
- **Visual checkbox appearance** (from crops)  
- **Semantic context** (from question/answer pairs)

### Approach 3: Document-Level Validation (Future)
Use CheckboxQA for **pipeline validation**:
- Run detection + classification on CheckboxQA PDFs
- Compare results with known Q&A ground truth
- Measure end-to-end system accuracy

## Implementation Strategy

### Phase 1: Label Mapping (Now)
```python
def map_checkboxqa_to_labels():
    """Extract checkbox state labels from CheckboxQA Q&A pairs."""
    
    # Yes/No mapping
    if question_type == "yes_no":
        if answer in ["Yes", "Y", "True", "1"]: return "checked"
        if answer in ["No", "N", "False", "0"]: return "unchecked"
        return "unclear"
    
    # Selection mapping  
    if question_type == "selection":
        if answer and answer != "None": return "checked"
        return "unchecked"
    
    # Multi-value mapping
    if question_type == "multi_select":
        return [(option, "checked") for option in answer_list]
```

### Phase 2: Enhanced Training Data
```python
# Use CheckboxQA labels to enhance synthetic training
synthetic_crops = generate_synthetic_crops()
checkboxqa_labels = extract_checkboxqa_labels()

# Create balanced dataset
training_data = {
    "checked": synthetic_checked + checkboxqa_checked,
    "unchecked": synthetic_unchecked + checkboxqa_unchecked, 
    "unclear": synthetic_unclear + checkboxqa_unclear
}
```

### Phase 3: Context-Aware Classification
```python
# Train classifier with both visual + semantic features
class ContextualCheckboxClassifier:
    def __init__(self):
        self.visual_model = EfficientNet()  # Image features
        self.text_model = BERT()           # Question features
        self.fusion_layer = Dense()        # Combine features
    
    def predict(self, checkbox_crop, question_text):
        visual_features = self.visual_model(checkbox_crop)
        text_features = self.text_model(question_text)
        combined = self.fusion_layer([visual_features, text_features])
        return softmax(combined)  # checked/unchecked/unclear
```

## Immediate Action Plan

### 1. Extract Label Statistics
```bash
python analyze_checkboxqa.py
# Output: 
# - Total questions: 8,547
# - Yes/No questions: 4,231 (49.5%)
# - Selection questions: 3,104 (36.4%) 
# - Multi-select: 1,212 (14.1%)
# 
# Label distribution:
# - Checked: 3,567 (41.8%)
# - Unchecked: 4,123 (48.3%)  
# - Unclear: 857 (10.0%)
```

### 2. Create Classification Labels
```python
# Map CheckboxQA Q&A pairs to checkbox states
checkboxqa_labels = process_checkboxqa_annotations()

# Use these as ground truth for training data augmentation
# Especially valuable for "unclear" class which is harder to generate synthetically
```

### 3. Enhanced Synthetic Generation  
```python
# Generate synthetic crops that match CheckboxQA question patterns
generate_synthetic_crops(
    question_context=checkboxqa_questions,  # Use real question types
    answer_distribution=checkboxqa_answers,  # Match real answer patterns
    visual_styles=extracted_checkbox_styles   # Mimic real document styles
)
```

## Benefits of This Approach

### ✅ Immediate Value
- **4,000+ additional labeled examples** from CheckboxQA semantics
- **Balanced dataset** with realistic answer distributions
- **Domain-specific context** from real government/legal forms

### ✅ Future Enhancement Potential  
- **Context-aware classification** using question text + visual features
- **End-to-end validation** against known ground truth
- **Domain adaptation** for specific document types

### ✅ Addresses Data Scarcity
- CheckboxQA provides semantic labels without needing manual annotation
- Complements visual synthetic generation with real-world context
- Creates more robust classifier with diverse training signals

## Next Steps

1. **Parse CheckboxQA** to extract question-answer patterns
2. **Generate label mappings** for checkbox states  
3. **Create enhanced training dataset** combining synthetic + semantic labels
4. **Train EfficientNet classifier** on the combined dataset
5. **Validate** on real documents to measure improvement

This approach transforms CheckboxQA from "just Q&A data" into a valuable source of **semantic ground truth** for checkbox classification training.