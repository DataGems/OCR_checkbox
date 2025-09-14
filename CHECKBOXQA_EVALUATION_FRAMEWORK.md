# CheckboxQA Evaluation Framework Analysis

## Key Insights from evaluate.py

### Evaluation Metric: ANLS* Score
- **ANLS*** = Approximate Normalized Levenshtein Similarity (modified)
- Measures similarity between predicted answers and ground truth
- Score range: 0.0 to 1.0 (higher is better)
- Accounts for minor variations in text answers

### Expected Input/Output Format

#### Prediction Format (JSONL)
```json
{
  "name": "document_id",
  "annotations": [
    {
      "id": 1,
      "key": "Is training required?",
      "values": [{"value": "No"}]
    }
  ]
}
```

#### Key Observations:
1. **Document-level predictions**: Each line is one document
2. **Question matching**: Uses annotation `id` to match predictions to ground truth
3. **Multiple values supported**: Can have multiple answers per question
4. **None handling**: "None" is a valid answer for empty/unchecked

### Answer Processing Logic

```python
# The evaluation script normalizes answers:
- Single values → List format
- "None" → None (Python None)
- Multiple values → Maintained as list
- Case-sensitive comparison
```

### What This Means for Our Pipeline

#### Our Pipeline Must:
1. **Process entire documents** (not individual checkboxes)
2. **Answer specific questions** about checkbox states
3. **Return structured predictions** matching the JSONL format
4. **Handle multi-value questions** (e.g., "Which options are selected?")

#### Pipeline Architecture:
```
PDF Document
    ↓
[Detection] Find all checkboxes
    ↓
[Classification] Determine states (checked/unchecked)
    ↓
[Question Answering] Map checkboxes to questions
    ↓
JSONL Output (matching evaluation format)
```

## Implementation Strategy

### Step 1: Question-Checkbox Mapping
```python
def map_question_to_checkboxes(question, detected_checkboxes, page_layout):
    """
    Map natural language questions to specific checkboxes.
    
    Approaches:
    1. Spatial proximity (checkbox near question text)
    2. Text matching (question keywords → nearby text)
    3. Pattern matching (e.g., "Yes/No" questions → paired checkboxes)
    """
```

### Step 2: Answer Generation
```python
def generate_answer(question, checkbox_states):
    """
    Convert checkbox states to natural language answers.
    
    Examples:
    - Yes/No question + checked "No" box → "No"
    - Selection question + checked options → ["Option A", "Option C"]
    - No checked boxes → "None"
    """
```

### Step 3: JSONL Output Creation
```python
def create_evaluation_output(document_id, qa_predictions):
    """
    Format predictions for CheckboxQA evaluation.
    """
    return {
        "name": document_id,
        "annotations": [
            {
                "id": qa["id"],
                "key": qa["question"],
                "values": [{"value": answer} for answer in qa["answers"]]
            }
            for qa in qa_predictions
        ]
    }
```

## Evaluation Workflow

### 1. Run Complete Pipeline
```bash
python run_checkboxqa_pipeline.py \
    --input data/checkboxqa/pdfs/ \
    --output predictions.jsonl \
    --detection-model models/best.pt \
    --classification-model models/classifier.pth
```

### 2. Evaluate Results
```bash
python evaluate.py \
    --pred predictions.jsonl \
    --gold data/checkboxqa/data/gold.jsonl
```

### 3. Expected Output
```
ANLS* Score: 0.8234
```

## Critical Realization

CheckboxQA is **NOT** just about detecting and classifying individual checkboxes. It's about:

1. **Understanding document structure**
2. **Linking questions to visual elements**
3. **Generating natural language answers**
4. **Handling complex multi-checkbox scenarios**

This requires a more sophisticated pipeline than simple detection + classification!

## Enhanced Pipeline Components

### 1. Layout Understanding
- Detect text regions near checkboxes
- Associate labels with checkboxes
- Understand form structure (sections, groups)

### 2. Question Understanding
- Parse question intent (Yes/No, selection, multi-select)
- Extract key entities from questions
- Match question context to document regions

### 3. Answer Synthesis
- Map checkbox states to appropriate answers
- Handle multi-checkbox questions
- Generate "None" for unanswered questions

## Next Steps

1. **Implement question-checkbox mapping** logic
2. **Add OCR capability** to read text near checkboxes
3. **Create answer generation** module
4. **Test end-to-end** on CheckboxQA samples
5. **Optimize for ANLS* score** >0.85

The evaluation script reveals that CheckboxQA requires a **complete document understanding system**, not just checkbox detection!