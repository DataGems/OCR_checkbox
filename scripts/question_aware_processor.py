#!/usr/bin/env python3
"""
Question-aware processing for CheckboxQA.
Maps natural language questions to checkbox detections and generates answers.
"""

import re
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
from dataclasses import dataclass
import numpy as np

@dataclass
class Checkbox:
    """Represents a detected checkbox with its properties."""
    bbox: List[float]  # [x1, y1, x2, y2]
    state: str  # 'checked', 'unchecked', 'unclear'
    confidence: float
    page: int
    nearby_text: str = ""
    row_index: Optional[int] = None
    col_index: Optional[int] = None

@dataclass
class Question:
    """Represents a CheckboxQA question."""
    id: int
    text: str
    question_type: str  # 'binary', 'single_select', 'multi_select'
    
class QuestionAwareProcessor:
    """Process checkbox questions with context awareness."""
    
    def __init__(self):
        # Question type patterns
        self.binary_patterns = [
            r'^(is|are|do|does|did|have|has|was|were|will|would|should|can|could)\s',
            r'\?$',  # Questions ending with ?
        ]
        
        self.selection_patterns = [
            r'what\s+(is|are)\s+(selected|checked|chosen)',
            r'which\s+.+\s+(selected|checked)',
            r'what\s+(option|choice|answer)',
        ]
        
        self.multi_patterns = [
            r'what\s+are\s+the\s+selected',
            r'which\s+.+\s+are\s+(selected|checked)',
            r'select\s+all',
        ]
        
        # Common checkbox text patterns
        self.yes_patterns = ['yes', 'y', 'true', '✓', 'x', '✔', 'checked']
        self.no_patterns = ['no', 'n', 'false', 'unchecked', 'not checked']
        
    def classify_question_type(self, question: str) -> str:
        """Classify question into binary, single_select, or multi_select."""
        question_lower = question.lower().strip()
        
        # Check for multi-select first (most specific)
        for pattern in self.multi_patterns:
            if re.search(pattern, question_lower):
                return 'multi_select'
        
        # Check for single selection
        for pattern in self.selection_patterns:
            if re.search(pattern, question_lower):
                return 'single_select'
        
        # Check for binary (yes/no)
        for pattern in self.binary_patterns:
            if re.search(pattern, question_lower):
                return 'binary'
        
        # Default to single select for other questions
        return 'single_select'
    
    def extract_question_context(self, question: str) -> Dict[str, any]:
        """Extract key information from the question text."""
        context = {
            'type': self.classify_question_type(question),
            'keywords': [],
            'looking_for': None,
            'negation': False
        }
        
        # Extract quoted text (often indicates what to look for)
        quoted = re.findall(r'"([^"]*)"', question)
        if quoted:
            context['looking_for'] = quoted[0]
        
        # Check for negation
        negation_words = ['not', 'no', "n't", 'without', 'except']
        question_lower = question.lower()
        context['negation'] = any(word in question_lower for word in negation_words)
        
        # Extract key terms (nouns and important words)
        # Simple approach - could be enhanced with NLP
        important_words = re.findall(r'\b[A-Z][a-z]+\b', question)  # Capitalized words
        context['keywords'] = important_words
        
        return context
    
    def find_binary_checkboxes(self, checkboxes: List[Checkbox]) -> Tuple[Optional[Checkbox], Optional[Checkbox]]:
        """Find Yes/No checkbox pairs for binary questions."""
        yes_box = None
        no_box = None
        
        for checkbox in checkboxes:
            text_lower = checkbox.nearby_text.lower().strip()
            
            # Check for Yes checkbox
            if any(pattern in text_lower for pattern in self.yes_patterns):
                if yes_box is None or checkbox.confidence > yes_box.confidence:
                    yes_box = checkbox
            
            # Check for No checkbox
            elif any(pattern in text_lower for pattern in self.no_patterns):
                if no_box is None or checkbox.confidence > no_box.confidence:
                    no_box = checkbox
        
        # If we have both, check which is checked
        if yes_box and no_box:
            return yes_box, no_box
        
        # Try spatial heuristic: often Yes/No are horizontally aligned
        if yes_box and not no_box:
            # Look for checkbox at similar Y coordinate
            y_center = (yes_box.bbox[1] + yes_box.bbox[3]) / 2
            for checkbox in checkboxes:
                if checkbox != yes_box:
                    other_y = (checkbox.bbox[1] + checkbox.bbox[3]) / 2
                    if abs(y_center - other_y) < 20:  # Within 20 pixels vertically
                        no_box = checkbox
                        break
        
        return yes_box, no_box
    
    def find_selection_checkboxes(self, checkboxes: List[Checkbox], keywords: List[str]) -> List[Checkbox]:
        """Find checkboxes that match selection criteria."""
        matching_boxes = []
        
        for checkbox in checkboxes:
            text_lower = checkbox.nearby_text.lower()
            
            # Check if any keyword matches
            if keywords:
                for keyword in keywords:
                    if keyword.lower() in text_lower:
                        matching_boxes.append(checkbox)
                        break
            else:
                # If no specific keywords, consider all checked boxes
                if checkbox.state == 'checked':
                    matching_boxes.append(checkbox)
        
        return matching_boxes
    
    def group_checkboxes_by_row(self, checkboxes: List[Checkbox]) -> Dict[int, List[Checkbox]]:
        """Group checkboxes by their vertical position (row)."""
        if not checkboxes:
            return {}
        
        # Sort by Y coordinate
        sorted_boxes = sorted(checkboxes, key=lambda b: b.bbox[1])
        
        rows = {}
        current_row = 0
        current_y = sorted_boxes[0].bbox[1]
        row_threshold = 20  # Pixels threshold for same row
        
        for checkbox in sorted_boxes:
            y_pos = checkbox.bbox[1]
            
            # Check if this is a new row
            if abs(y_pos - current_y) > row_threshold:
                current_row += 1
                current_y = y_pos
            
            if current_row not in rows:
                rows[current_row] = []
            rows[current_row].append(checkbox)
        
        return rows
    
    def answer_binary_question(self, question: str, checkboxes: List[Checkbox]) -> str:
        """Answer a yes/no question based on checkbox states."""
        yes_box, no_box = self.find_binary_checkboxes(checkboxes)
        
        # Check which is checked
        if yes_box and yes_box.state == 'checked':
            return "Yes"
        elif no_box and no_box.state == 'checked':
            return "No"
        
        # If neither is clearly checked, check context
        context = self.extract_question_context(question)
        if context['negation']:
            # Questions with negation often default to "No"
            return "No"
        
        # Default to None if unclear
        return "None"
    
    def answer_selection_question(self, question: str, checkboxes: List[Checkbox]) -> Union[str, List[str]]:
        """Answer a selection question based on checked options."""
        context = self.extract_question_context(question)
        
        # Find all checked checkboxes
        checked_boxes = [cb for cb in checkboxes if cb.state == 'checked']
        
        if not checked_boxes:
            return "None"
        
        # If looking for specific text
        if context['looking_for']:
            for checkbox in checked_boxes:
                if context['looking_for'].lower() in checkbox.nearby_text.lower():
                    return checkbox.nearby_text.strip()
        
        # For multi-select, return all checked options
        if context['type'] == 'multi_select':
            answers = [cb.nearby_text.strip() for cb in checked_boxes if cb.nearby_text]
            return answers if answers else "None"
        
        # For single select, return the most confident checked option
        if checked_boxes:
            best_box = max(checked_boxes, key=lambda cb: cb.confidence)
            return best_box.nearby_text.strip() if best_box.nearby_text else "None"
        
        return "None"
    
    def process_question(self, question: Question, checkboxes: List[Checkbox]) -> Dict[str, any]:
        """Process a question and return the answer."""
        # Classify question type
        q_type = self.classify_question_type(question.text)
        
        # Answer based on type
        if q_type == 'binary':
            answer = self.answer_binary_question(question.text, checkboxes)
        else:
            answer = self.answer_selection_question(question.text, checkboxes)
        
        # Format for CheckboxQA evaluation
        return {
            "id": question.id,
            "key": question.text,
            "values": [{"value": answer}] if isinstance(answer, str) else [{"value": v} for v in answer]
        }
    
    def process_document(self, document_id: str, questions: List[Question], 
                        checkboxes: List[Checkbox]) -> Dict[str, any]:
        """Process all questions for a document."""
        annotations = []
        
        for question in questions:
            result = self.process_question(question, checkboxes)
            annotations.append(result)
        
        return {
            "name": document_id,
            "annotations": annotations
        }
    
    def spatial_proximity_score(self, checkbox: Checkbox, text_bbox: List[float]) -> float:
        """Calculate proximity score between checkbox and text."""
        # Calculate centers
        cb_cx = (checkbox.bbox[0] + checkbox.bbox[2]) / 2
        cb_cy = (checkbox.bbox[1] + checkbox.bbox[3]) / 2
        
        text_cx = (text_bbox[0] + text_bbox[2]) / 2
        text_cy = (text_bbox[1] + text_bbox[3]) / 2
        
        # Euclidean distance
        distance = np.sqrt((cb_cx - text_cx)**2 + (cb_cy - text_cy)**2)
        
        # Check alignment
        horizontal_aligned = abs(cb_cy - text_cy) < 10
        vertical_aligned = abs(cb_cx - text_cx) < 10
        
        # Text usually to the left or right of checkbox
        text_left = text_cx < cb_cx
        text_right = text_cx > cb_cx
        
        # Calculate score (higher is better)
        score = 1.0 / (1.0 + distance / 100.0)  # Distance factor
        
        if horizontal_aligned:
            score *= 2.0  # Boost for horizontal alignment
        
        if text_left or text_right:
            score *= 1.5  # Boost for expected positions
        
        return score

def demo_usage():
    """Demonstrate question-aware processing."""
    processor = QuestionAwareProcessor()
    
    # Example checkboxes
    checkboxes = [
        Checkbox(bbox=[100, 200, 120, 220], state='unchecked', confidence=0.95, 
                page=0, nearby_text="Yes"),
        Checkbox(bbox=[150, 200, 170, 220], state='checked', confidence=0.98, 
                page=0, nearby_text="No"),
        Checkbox(bbox=[100, 250, 120, 270], state='checked', confidence=0.92, 
                page=0, nearby_text="Master's"),
        Checkbox(bbox=[100, 280, 120, 300], state='unchecked', confidence=0.91, 
                page=0, nearby_text="Bachelor's"),
    ]
    
    # Example questions
    questions = [
        Question(id=1, text="Is training required in the job opportunity?", question_type='binary'),
        Question(id=2, text="What option is selected in \"Education: minimum level required:\"?", 
                question_type='single_select'),
    ]
    
    # Process questions
    for question in questions:
        result = processor.process_question(question, checkboxes)
        print(f"\nQuestion: {question.text}")
        print(f"Answer: {result['values'][0]['value']}")
    
    # Process full document
    doc_result = processor.process_document("example_doc", questions, checkboxes)
    print(f"\nFull document result:")
    print(json.dumps(doc_result, indent=2))

if __name__ == "__main__":
    demo_usage()