#!/usr/bin/env python3
"""
VLM Integration for CheckboxQA hybrid approach.
Combines our specialized detection with general-purpose VLM reasoning.
"""

import base64
import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import requests
from PIL import Image
import io

@dataclass
class VLMConfig:
    """Configuration for VLM integration."""
    provider: str = "openai"  # openai, anthropic, google, local
    model: str = "gpt-4-vision-preview"
    api_key: Optional[str] = None
    max_tokens: int = 1000
    temperature: float = 0.1
    cost_per_request: float = 0.01  # Estimated cost

class VLMIntegrator:
    """Integrates VLM capabilities with our checkbox pipeline."""
    
    def __init__(self, config: VLMConfig):
        self.config = config
        self.total_cost = 0.0
        self.vlm_calls = 0
        
    def should_use_vlm(self, 
                       question: str, 
                       checkboxes: List[Dict], 
                       confidence_scores: List[float]) -> bool:
        """Decide whether to use VLM or rule-based processing."""
        
        # Use VLM if detection/classification confidence is low
        if checkboxes and min(confidence_scores) < 0.7:
            return True
        
        # Use VLM for complex spatial questions
        complex_patterns = [
            'table', 'row', 'column', 'grid',
            'above', 'below', 'left of', 'right of',
            'section', 'part', 'area', 'region'
        ]
        
        if any(pattern in question.lower() for pattern in complex_patterns):
            return True
        
        # Use VLM for multi-checkbox scenarios
        if len(checkboxes) > 8:
            return True
        
        # Use VLM for unclear question types
        if self._classify_question_complexity(question) == 'complex':
            return True
        
        return False
    
    def _classify_question_complexity(self, question: str) -> str:
        """Classify question complexity for VLM routing."""
        question_lower = question.lower()
        
        # Simple binary questions
        simple_patterns = [
            r'^is\s+.+\s+required\?$',
            r'^do(es)?\s+.+\?$',
            r'^(was|were)\s+.+\?$'
        ]
        
        import re
        if any(re.search(pattern, question_lower) for pattern in simple_patterns):
            return 'simple'
        
        # Complex spatial/contextual questions
        complex_patterns = [
            'what.*selected.*in.*section',
            'which.*checked.*table',
            'what.*options.*are.*marked'
        ]
        
        if any(re.search(pattern, question_lower) for pattern in complex_patterns):
            return 'complex'
        
        return 'medium'
    
    def extract_question_region(self, 
                               image: np.ndarray, 
                               question: str,
                               checkboxes: List[Dict],
                               context_padding: int = 50) -> np.ndarray:
        """Extract relevant image region for VLM processing."""
        
        if not checkboxes:
            return image
        
        # Find bounding box of all relevant checkboxes
        all_boxes = [cb['bbox'] for cb in checkboxes]
        min_x = min(box[0] for box in all_boxes) - context_padding
        min_y = min(box[1] for box in all_boxes) - context_padding
        max_x = max(box[2] for box in all_boxes) + context_padding
        max_y = max(box[3] for box in all_boxes) + context_padding
        
        # Ensure bounds are within image
        h, w = image.shape[:2]
        min_x = max(0, min_x)
        min_y = max(0, min_y)
        max_x = min(w, max_x)
        max_y = min(h, max_y)
        
        # Extract region
        region = image[min_y:max_y, min_x:max_x]
        
        # Resize if too large (VLM input limits)
        if region.shape[0] > 1024 or region.shape[1] > 1024:
            scale = min(1024 / region.shape[0], 1024 / region.shape[1])
            new_h = int(region.shape[0] * scale)
            new_w = int(region.shape[1] * scale)
            region = cv2.resize(region, (new_w, new_h))
        
        return region
    
    def create_vlm_prompt(self, 
                         question: str,
                         detected_checkboxes: List[Dict],
                         context: Optional[str] = None) -> str:
        """Create optimized prompt for VLM."""
        
        prompt = f"""You are an expert at analyzing form documents and checkboxes.

TASK: Answer the following question based on the checkbox states visible in this image:

QUESTION: {question}

DETECTED CHECKBOXES: I've detected {len(detected_checkboxes)} potential checkboxes in this image region.

INSTRUCTIONS:
1. Look carefully at each checkbox to determine if it's checked (✓, X, filled) or unchecked (empty)
2. Read the text labels near each checkbox
3. Answer the question based ONLY on what you can see
4. If multiple checkboxes are relevant, list all that apply
5. If no checkboxes are clearly checked, answer "None" or "No"

RESPONSE FORMAT:
- For Yes/No questions: Answer "Yes" or "No"  
- For selection questions: Answer with the exact text near the checked checkbox(es)
- For multiple selections: Answer with a list like ["Option A", "Option B"]
- If unclear: Answer "None"

Be precise and only report what you can clearly see in the image."""

        if context:
            prompt += f"\n\nADDITIONAL CONTEXT: {context}"
        
        return prompt
    
    def encode_image(self, image: np.ndarray) -> str:
        """Encode image for VLM API."""
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL
        pil_image = Image.fromarray(image_rgb)
        
        # Encode to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return image_base64
    
    def query_openai_vlm(self, image: np.ndarray, prompt: str) -> str:
        """Query OpenAI GPT-4V."""
        image_base64 = self.encode_image(image)
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}"
        }
        
        payload = {
            "model": self.config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                answer = result['choices'][0]['message']['content'].strip()
                
                # Track usage
                self.vlm_calls += 1
                self.total_cost += self.config.cost_per_request
                
                return answer
            else:
                print(f"⚠️ VLM API error: {response.status_code}")
                return "Error"
                
        except Exception as e:
            print(f"⚠️ VLM query failed: {e}")
            return "Error"
    
    def query_anthropic_vlm(self, image: np.ndarray, prompt: str) -> str:
        """Query Anthropic Claude Vision (when available)."""
        # Implementation for Claude Vision API
        # Currently in limited beta
        return "Claude Vision not implemented yet"
    
    def query_local_vlm(self, image: np.ndarray, prompt: str) -> str:
        """Query local VLM (LLaVA, InternVL, etc.)."""
        # Implementation for local VLM inference
        # Would use transformers library
        return "Local VLM not implemented yet"
    
    def query_vlm(self, image: np.ndarray, prompt: str) -> str:
        """Query VLM based on configuration."""
        if self.config.provider == "openai":
            return self.query_openai_vlm(image, prompt)
        elif self.config.provider == "anthropic":
            return self.query_anthropic_vlm(image, prompt)
        elif self.config.provider == "local":
            return self.query_local_vlm(image, prompt)
        else:
            raise ValueError(f"Unsupported VLM provider: {self.config.provider}")
    
    def process_vlm_response(self, response: str, question: str) -> str:
        """Process and validate VLM response."""
        response = response.strip()
        
        # Handle common VLM response patterns
        if "answer:" in response.lower():
            # Extract answer after "Answer:"
            parts = response.lower().split("answer:")
            if len(parts) > 1:
                response = parts[1].strip()
        
        # Remove quotes if present
        response = response.strip('"\'')
        
        # Validate response format
        if not response or response.lower() in ['error', 'unclear', 'cannot determine']:
            return "None"
        
        return response
    
    def get_usage_stats(self) -> Dict:
        """Get VLM usage statistics."""
        return {
            'total_calls': self.vlm_calls,
            'total_cost': round(self.total_cost, 4),
            'average_cost_per_call': round(self.total_cost / max(self.vlm_calls, 1), 4)
        }

class HybridCheckboxProcessor:
    """Hybrid processor combining detection + VLM."""
    
    def __init__(self, 
                 detection_model_path: str = "models/best.pt",
                 vlm_config: Optional[VLMConfig] = None):
        
        # Load detection model
        from ultralytics import YOLO
        self.detector = YOLO(detection_model_path)
        
        # Initialize VLM
        if vlm_config is None:
            vlm_config = VLMConfig()
        self.vlm = VLMIntegrator(vlm_config)
        
        print(f"🔄 Initialized Hybrid Processor:")
        print(f"   Detection: {detection_model_path}")
        print(f"   VLM: {vlm_config.provider} ({vlm_config.model})")
    
    def process_document_hybrid(self, 
                               image: np.ndarray,
                               questions: List[Dict]) -> List[Dict]:
        """Process document using hybrid approach."""
        results = []
        
        # Step 1: Detect all checkboxes with our model
        detection_results = self.detector(image, conf=0.3)
        
        checkboxes = []
        confidence_scores = []
        
        for result in detection_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    
                    checkboxes.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf
                    })
                    confidence_scores.append(conf)
        
        print(f"🔍 Detected {len(checkboxes)} checkboxes")
        
        # Step 2: Process each question
        for question_data in questions:
            question_text = question_data['key']
            question_id = question_data['id']
            
            # Decide: VLM or rule-based?
            use_vlm = self.vlm.should_use_vlm(
                question_text, checkboxes, confidence_scores
            )
            
            if use_vlm:
                print(f"🤖 Using VLM for: {question_text[:50]}...")
                
                # Extract relevant region
                region = self.vlm.extract_question_region(
                    image, question_text, checkboxes
                )
                
                # Create prompt
                prompt = self.vlm.create_vlm_prompt(
                    question_text, checkboxes
                )
                
                # Query VLM
                raw_response = self.vlm.query_vlm(region, prompt)
                answer = self.vlm.process_vlm_response(raw_response, question_text)
                
            else:
                print(f"⚙️ Using rules for: {question_text[:50]}...")
                # Use our rule-based processing
                # (This would call our existing question_aware_processor)
                answer = self._process_with_rules(question_text, checkboxes, image)
            
            # Format response
            results.append({
                "id": question_id,
                "key": question_text,
                "values": [{"value": answer}],
                "method": "vlm" if use_vlm else "rules"
            })
        
        return results
    
    def _process_with_rules(self, question: str, checkboxes: List[Dict], image: np.ndarray) -> str:
        """Fallback rule-based processing."""
        # Simple heuristic for now
        # In practice, this would use our question_aware_processor
        
        if not checkboxes:
            return "None"
        
        # For binary questions, assume first checkbox is relevant
        if any(word in question.lower() for word in ['is ', 'are ', 'do ', 'did ']):
            # Rough heuristic based on checkbox position/size
            checkbox = checkboxes[0]
            x1, y1, x2, y2 = checkbox['bbox']
            
            # Extract crop and do simple analysis
            crop = image[y1:y2, x1:x2]
            if crop.size > 0:
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
                fill_ratio = np.sum(binary == 255) / binary.size
                
                return "Yes" if fill_ratio > 0.2 else "No"
        
        return "None"

def demo_hybrid_approach():
    """Demonstrate hybrid VLM approach."""
    # Configure VLM (you'll need to add your API key)
    vlm_config = VLMConfig(
        provider="openai",
        model="gpt-4-vision-preview",
        api_key="your-api-key-here"  # Add your key
    )
    
    # Create hybrid processor
    processor = HybridCheckboxProcessor(
        detection_model_path="models/best.pt",
        vlm_config=vlm_config
    )
    
    # Example usage (you'd load real CheckboxQA images)
    # image = cv2.imread("test_document.png")
    # questions = [{"id": 1, "key": "Is training required?"}]
    # results = processor.process_document_hybrid(image, questions)
    # print(results)
    
    print("Hybrid processor initialized!")
    print("Usage: processor.process_document_hybrid(image, questions)")

if __name__ == "__main__":
    demo_hybrid_approach()