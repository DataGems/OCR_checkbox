#!/usr/bin/env python3
"""
Analyze CheckboxQA dataset to understand label patterns and extract
checkbox state information from Q&A pairs.
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

def classify_question_type(question: str) -> str:
    """Classify question type based on text patterns."""
    question = question.lower().strip()
    
    # Yes/No questions
    yes_no_patterns = [
        r'^is\s', r'^are\s', r'^do\s', r'^did\s', r'^have\s', 
        r'^has\s', r'^was\s', r'^were\s', r'^will\s', r'^does\s'
    ]
    
    if any(re.search(pattern, question) for pattern in yes_no_patterns):
        return "yes_no"
    
    # Selection questions
    selection_patterns = [
        r'what\s+is\s+selected', r'what\s+is\s+checked', r'what\s+option',
        r'what\s+are\s+the\s+selected', r'what\s+is\s+the\s+selected'
    ]
    
    if any(re.search(pattern, question) for pattern in selection_patterns):
        return "selection"
    
    # Multiple choice
    if 'what are' in question and ('selected' in question or 'checked' in question):
        return "multi_select"
    
    return "other"

def classify_checkbox_state(question: str, answer: str) -> str:
    """Classify checkbox state based on question and answer."""
    question = question.lower().strip()
    answer = answer.strip()
    
    if not answer:
        return "unclear"
    
    answer_lower = answer.lower()
    
    # Yes/No questions
    if classify_question_type(question) == "yes_no":
        if answer_lower in ['yes', 'y', 'true', '1', 'checked']:
            return "checked"
        elif answer_lower in ['no', 'n', 'false', '0', 'unchecked', '']:
            return "unchecked"
        elif answer_lower in ['na', 'n/a', 'not applicable', 'unclear']:
            return "unclear"
    
    # Selection questions
    elif classify_question_type(question) in ["selection", "multi_select"]:
        if answer_lower in ['none', '', 'not selected', 'unchecked']:
            return "unchecked"
        elif answer_lower in ['na', 'n/a', 'not applicable']:
            return "unclear"
        else:
            return "checked"  # Has a specific value
    
    # Default classification
    if answer and len(answer.strip()) > 0:
        return "checked"
    else:
        return "unchecked"

def analyze_checkboxqa_data(data_file: Path) -> Dict:
    """Analyze CheckboxQA data and extract statistics."""
    
    if not data_file.exists():
        print(f"❌ CheckboxQA file not found: {data_file}")
        return {}
    
    print(f"📊 Analyzing CheckboxQA data from {data_file}")
    
    stats = {
        'total_documents': 0,
        'total_questions': 0,
        'question_types': Counter(),
        'checkbox_states': Counter(),
        'answer_patterns': Counter(),
        'documents_by_split': Counter(),
        'examples_by_state': defaultdict(list)
    }
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                stats['total_documents'] += 1
                stats['documents_by_split'][doc.get('split', 'unknown')] += 1
                
                for annotation in doc.get('annotations', []):
                    question = annotation.get('key', '')
                    values = annotation.get('values', [])
                    
                    if not question or not values:
                        continue
                    
                    stats['total_questions'] += 1
                    
                    # Classify question type
                    q_type = classify_question_type(question)
                    stats['question_types'][q_type] += 1
                    
                    # Process each answer
                    for value_obj in values:
                        answer = value_obj.get('value', '')
                        
                        # Classify checkbox state
                        state = classify_checkbox_state(question, answer)
                        stats['checkbox_states'][state] += 1
                        
                        # Track answer patterns
                        if answer:
                            stats['answer_patterns'][answer.lower()] += 1
                        
                        # Store examples for each state
                        if len(stats['examples_by_state'][state]) < 10:  # Keep first 10 examples
                            stats['examples_by_state'][state].append({
                                'document': doc.get('name', 'unknown'),
                                'question': question,
                                'answer': answer,
                                'question_type': q_type
                            })
            
            except json.JSONDecodeError as e:
                print(f"⚠️ Error parsing line: {e}")
                continue
    
    return stats

def print_analysis_results(stats: Dict):
    """Print comprehensive analysis results."""
    
    if not stats:
        print("❌ No data to analyze")
        return
    
    print("\n" + "="*60)
    print("📊 CHECKBOXQA DATASET ANALYSIS")
    print("="*60)
    
    # Basic statistics
    print(f"Total documents: {stats['total_documents']:,}")
    print(f"Total questions: {stats['total_questions']:,}")
    
    # Document splits
    print(f"\nDocument splits:")
    for split, count in stats['documents_by_split'].most_common():
        pct = (count / stats['total_documents']) * 100
        print(f"  {split}: {count:,} ({pct:.1f}%)")
    
    # Question types
    print(f"\nQuestion types:")
    for q_type, count in stats['question_types'].most_common():
        pct = (count / stats['total_questions']) * 100
        print(f"  {q_type}: {count:,} ({pct:.1f}%)")
    
    # Checkbox states (our labels!)
    print(f"\n🎯 EXTRACTED CHECKBOX STATES:")
    total_states = sum(stats['checkbox_states'].values())
    for state, count in stats['checkbox_states'].most_common():
        pct = (count / total_states) * 100
        print(f"  {state.upper()}: {count:,} ({pct:.1f}%)")
    
    # Most common answers
    print(f"\nMost common answers:")
    for answer, count in stats['answer_patterns'].most_common(10):
        print(f"  '{answer}': {count:,}")
    
    # Example questions for each state
    print(f"\n📝 EXAMPLE QUESTIONS BY STATE:")
    for state in ['checked', 'unchecked', 'unclear']:
        examples = stats['examples_by_state'][state]
        print(f"\n{state.upper()} examples ({len(examples)}):")
        for i, example in enumerate(examples[:3], 1):  # Show first 3
            print(f"  {i}. Q: {example['question'][:80]}{'...' if len(example['question']) > 80 else ''}")
            print(f"     A: {example['answer']}")
            print(f"     Type: {example['question_type']}")
    
    print("="*60)

def extract_classification_labels(data_file: Path, output_file: Path = None) -> List[Dict]:
    """Extract classification labels from CheckboxQA for training."""
    
    labels = []
    
    if not data_file.exists():
        print(f"❌ CheckboxQA file not found: {data_file}")
        return labels
    
    print(f"🏷️ Extracting classification labels...")
    
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                doc = json.loads(line.strip())
                doc_name = doc.get('name', 'unknown')
                
                for annotation in doc.get('annotations', []):
                    question = annotation.get('key', '')
                    values = annotation.get('values', [])
                    
                    for value_obj in values:
                        answer = value_obj.get('value', '')
                        state = classify_checkbox_state(question, answer)
                        
                        labels.append({
                            'document': doc_name,
                            'question': question,
                            'answer': answer,
                            'checkbox_state': state,
                            'question_type': classify_question_type(question),
                            'annotation_id': annotation.get('id', -1)
                        })
            
            except json.JSONDecodeError:
                continue
    
    print(f"✅ Extracted {len(labels)} classification labels")
    
    # Save labels if output path provided
    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(labels, f, indent=2, ensure_ascii=False)
        print(f"💾 Saved labels to: {output_file}")
    
    return labels

def main():
    """Main analysis function."""
    checkboxqa_file = Path("data/raw/checkboxqa/data/gold.jsonl")
    
    # Analyze the data
    stats = analyze_checkboxqa_data(checkboxqa_file)
    print_analysis_results(stats)
    
    # Extract labels for classification training
    labels_output = Path("data/checkboxqa_classification_labels.json")
    labels = extract_classification_labels(checkboxqa_file, labels_output)
    
    print(f"\n🎯 SUMMARY FOR CLASSIFICATION:")
    print(f"✅ {len(labels):,} labeled examples extracted from CheckboxQA")
    print(f"✅ Labels saved to: {labels_output}")
    print(f"✅ Ready to enhance classification training dataset")
    
    return 0

if __name__ == "__main__":
    exit(main())