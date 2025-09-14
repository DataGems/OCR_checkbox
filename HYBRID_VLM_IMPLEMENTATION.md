# Hybrid VLM Implementation Strategy

## Data Composition for Hybrid Approach

### Detection Model (Already Trained)
- **Real Data**: FUNSD (~150 images)
- **Synthetic**: 300 forms (~1,500 labels)  
- **Synthetic %**: ~90% (detection works well with synthetic)
- **Status**: ✅ Trained, mAP@0.5 = 0.881

### Classification Training (Next)
**Target**: 30-40% synthetic for better generalization

```python
classification_data = {
    # Real data sources (60-70%)
    'checkboxqa_crops': 200,      # From downloaded PDFs
    'funsd_crops': 150,           # Extracted from FUNSD
    'manual_crops': 50,           # High-quality samples
    
    # Synthetic data (30-40%)  
    'synthetic_crops': 400,       # Generated checkboxes
    
    # Total: ~800 classification samples
    # Synthetic %: 50% (at our guideline limit)
}
```

### VLM Integration Strategy
**Use VLM for**: Complex spatial reasoning, not raw detection/classification

```python
def hybrid_vlm_decision(question, checkboxes, confidence_scores):
    """Decide when to use VLM vs rule-based processing."""
    
    # Use VLM if:
    if (
        any(conf < 0.7 for conf in confidence_scores) or  # Low confidence
        is_complex_spatial_question(question) or          # Complex layout
        has_table_structure(checkboxes) or               # Table reasoning needed
        len(checkboxes) > 10                             # Many checkboxes
    ):
        return "use_vlm"
    else:
        return "use_rules"
```

## Implementation Plan

### Phase 1: Enhanced Classification (This Week)
```python
# 1. Generate balanced classification dataset
generate_classification_dataset(
    synthetic_ratio=0.4,  # 40% synthetic
    real_sources=['checkboxqa', 'funsd'],
    total_samples=1000
)

# 2. Train EfficientNet with real+synthetic
train_classifier(
    model='efficientnet-b0',
    data_augmentation=True,
    class_weights='balanced'  # Handle class imbalance
)
```

### Phase 2: VLM Integration (Next Week)
```python
# 3. Add VLM fallback for complex cases
class HybridCheckboxQA:
    def process_document(self, image, questions):
        # Always use our detection
        checkboxes = self.detector.detect(image)
        
        answers = []
        for question in questions:
            # Classify question complexity
            if self.needs_vlm(question, checkboxes):
                # Extract region with context
                region = self.extract_question_region(image, question, checkboxes)
                
                # Query VLM with focused prompt
                answer = self.query_vlm(region, question)
            else:
                # Use our rule-based processing
                answer = self.process_with_rules(question, checkboxes)
            
            answers.append(answer)
        
        return answers
```

### Phase 3: VLM Selection
**Option A: GPT-4V** (Most capable)
- Best spatial reasoning
- Expensive (~$0.01-0.03 per document)
- API dependency

**Option B: Open VLM** (Cost-effective)
- LLaVA-1.6, InternVL, Qwen-VL
- Can run locally/on-premise
- Lower capability but faster

**Option C: Gemini Vision** (Balanced)
- Good performance/cost ratio
- Fast inference
- Google ecosystem

## Synthetic Data Optimization

### Smart Synthetic Generation
```python
# Generate synthetic data matching real patterns
def generate_targeted_synthetic(real_samples, target_ratio=0.4):
    """Generate synthetic data that complements real data."""
    
    # Analyze real data patterns
    real_patterns = analyze_checkbox_styles(real_samples)
    
    # Generate synthetic samples filling gaps
    synthetic_samples = []
    for pattern in underrepresented_patterns:
        samples = generate_with_style(
            style=pattern,
            count=calculate_needed_count(pattern, target_ratio)
        )
        synthetic_samples.extend(samples)
    
    return synthetic_samples
```

### Quality Control
```python
# Validate synthetic data quality
def validate_synthetic_quality(synthetic_samples, real_samples):
    """Ensure synthetic data doesn't hurt performance."""
    
    # Train on real-only
    real_model = train_classifier(real_samples)
    real_accuracy = evaluate(real_model, test_set)
    
    # Train on real+synthetic
    hybrid_model = train_classifier(real_samples + synthetic_samples)
    hybrid_accuracy = evaluate(hybrid_model, test_set)
    
    # Synthetic is good if it improves performance
    return hybrid_accuracy > real_accuracy
```

## Performance Targets

### With Hybrid VLM Approach:
- **Detection**: mAP@0.5 > 0.85 ✅ (achieved: 0.881)
- **Classification**: Accuracy > 95% (target with real+synthetic)
- **End-to-End**: ANLS* > 85% (vs 83.2% SOTA)
- **Processing**: < 2 minutes per document
- **Cost**: < $0.05 per document (if using VLM APIs)

## Risk Mitigation

### Synthetic Data Risks:
- **Domain gap**: Synthetic doesn't match real documents
- **Overfitting**: Model learns synthetic artifacts
- **Class imbalance**: Synthetic skews label distribution

### Mitigation Strategies:
- **Limited ratio**: 30-40% synthetic maximum
- **Style matching**: Generate synthetic based on real patterns  
- **Validation**: A/B test real-only vs real+synthetic
- **Progressive training**: Start real-only, add synthetic gradually

## Next Steps

1. **Download CheckboxQA PDFs** for real classification data
2. **Extract checkbox crops** using our detection model
3. **Generate complementary synthetic** data (30-40% ratio)
4. **Train classification model** on balanced dataset
5. **Implement VLM integration** for complex cases
6. **Test end-to-end pipeline** on CheckboxQA

**Estimated Timeline**: 1-2 weeks for full hybrid implementation