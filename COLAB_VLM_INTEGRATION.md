# Colab VLM Integration Script

Test our hybrid detection + VLM approach in Google Colab.

```python
# 1. Setup
!pip install ultralytics openai requests pillow opencv-python matplotlib

import json
import cv2
import numpy as np
import base64
import io
from PIL import Image
from ultralytics import YOLO
import matplotlib.pyplot as plt

class HybridCheckboxProcessor:
    def __init__(self, model_path="best.pt", openai_api_key=None):
        """Initialize hybrid processor with detection + VLM."""
        self.detector = YOLO(model_path)
        self.api_key = openai_api_key
        self.vlm_calls = 0
        self.total_cost = 0.0
        
        print(f"🔄 Initialized Hybrid Processor")
        print(f"   Detection: {model_path}")
        print(f"   VLM: {'OpenAI GPT-4V' if openai_api_key else 'Mock VLM'}")
    
    def detect_checkboxes(self, image):
        """Detect checkboxes using our trained model."""
        results = self.detector(image, conf=0.3)
        
        checkboxes = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    
                    checkboxes.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf
                    })
        
        return checkboxes
    
    def should_use_vlm(self, question, checkboxes, confidence_scores):
        """Decide whether to use VLM or rule-based processing."""
        
        # Use VLM if low confidence
        if checkboxes and min(confidence_scores) < 0.7:
            return True, "Low detection confidence"
        
        # Use VLM for complex spatial questions
        complex_patterns = [
            'table', 'row', 'column', 'grid',
            'above', 'below', 'left of', 'right of',
            'section', 'part', 'area', 'region'
        ]
        
        if any(pattern in question.lower() for pattern in complex_patterns):
            return True, "Complex spatial question"
        
        # Use VLM for many checkboxes
        if len(checkboxes) > 8:
            return True, "Many checkboxes detected"
        
        return False, "Simple question"
    
    def create_vlm_prompt(self, question, detected_checkboxes):
        """Create optimized prompt for VLM."""
        
        prompt = f"""You are an expert at analyzing form documents and checkboxes.

TASK: Answer the following question based on the checkbox states in this image:

QUESTION: {question}

CONTEXT: I've detected {len(detected_checkboxes)} potential checkboxes in this image.

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

        return prompt
    
    def encode_image(self, image):
        """Encode image for VLM API."""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return image_base64
    
    def query_vlm_mock(self, image, prompt):
        """Mock VLM for testing (replace with real API call)."""
        self.vlm_calls += 1
        self.total_cost += 0.01  # Mock cost
        
        # Mock responses based on prompt content
        if "training required" in prompt.lower():
            return "Yes"
        elif "which options" in prompt.lower():
            return '["Option A", "Option C"]'
        else:
            return "Yes"
    
    def query_vlm_openai(self, image, prompt):
        """Query OpenAI GPT-4V (requires API key)."""
        if not self.api_key:
            return self.query_vlm_mock(image, prompt)
        
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        
        image_base64 = self.encode_image(image)
        
        try:
            response = client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user", 
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=500,
                temperature=0.1
            )
            
            self.vlm_calls += 1
            self.total_cost += 0.01  # Approximate cost
            
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            print(f"⚠️ VLM API error: {e}")
            return self.query_vlm_mock(image, prompt)
    
    def process_document_hybrid(self, image, questions):
        """Process document using hybrid approach."""
        print(f"🔄 Processing document with {len(questions)} questions...")
        
        # Step 1: Detect checkboxes
        checkboxes = self.detect_checkboxes(image)
        confidence_scores = [cb['confidence'] for cb in checkboxes]
        
        print(f"🔍 Detected {len(checkboxes)} checkboxes")
        if checkboxes:
            print(f"   Confidence range: {min(confidence_scores):.3f} - {max(confidence_scores):.3f}")
        
        # Step 2: Process each question
        results = []
        
        for i, question_data in enumerate(questions):
            question_text = question_data.get('key', question_data.get('question', ''))
            question_id = question_data.get('id', i)
            
            # Decide processing method
            use_vlm, reason = self.should_use_vlm(question_text, checkboxes, confidence_scores)
            
            if use_vlm:
                print(f"🤖 Using VLM: {question_text[:50]}... ({reason})")
                
                # Create prompt and query VLM
                prompt = self.create_vlm_prompt(question_text, checkboxes)
                answer = self.query_vlm_openai(image, prompt)
                
                # Clean response
                answer = answer.strip().strip('"\\'')
                
            else:
                print(f"⚙️ Using rules: {question_text[:50]}... ({reason})")
                # Simple rule-based processing
                if checkboxes and any(word in question_text.lower() for word in ['is ', 'are ', 'required']):
                    # Heuristic: if any checkbox detected with high confidence, answer "Yes"
                    high_conf_boxes = [cb for cb in checkboxes if cb['confidence'] > 0.8]
                    answer = "Yes" if high_conf_boxes else "No"
                else:
                    answer = "None"
            
            results.append({
                "id": question_id,
                "key": question_text,
                "values": [{"value": answer}],
                "method": "vlm" if use_vlm else "rules",
                "confidence": np.mean(confidence_scores) if confidence_scores else 0.0
            })
        
        return results
    
    def visualize_results(self, image, checkboxes, results):
        """Visualize detection and processing results."""
        vis_image = image.copy()
        
        # Draw detected checkboxes
        for checkbox in checkboxes:
            x1, y1, x2, y2 = checkbox['bbox']
            conf = checkbox['confidence']
            
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.5 else (0, 0, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(vis_image, f"{conf:.2f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Display results
        plt.figure(figsize=(15, 10))
        
        # Image with detections
        plt.subplot(2, 1, 1)
        plt.imshow(cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Hybrid Processing - {len(checkboxes)} checkboxes detected")
        plt.axis('off')
        
        # Results table
        plt.subplot(2, 1, 2)
        plt.axis('tight')
        plt.axis('off')
        
        table_data = []
        for result in results:
            table_data.append([
                result['key'][:40] + ('...' if len(result['key']) > 40 else ''),
                result['values'][0]['value'],
                result['method'].upper(),
                f"{result['confidence']:.3f}"
            ])
        
        table = plt.table(cellText=table_data,
                         colLabels=['Question', 'Answer', 'Method', 'Confidence'],
                         cellLoc='left',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        
        plt.title("Question Processing Results")
        plt.show()
        
        # Usage stats
        print(f"\n📊 Processing Stats:")
        print(f"   VLM calls: {self.vlm_calls}")
        print(f"   Estimated cost: ${self.total_cost:.3f}")
        print(f"   VLM usage: {len([r for r in results if r['method'] == 'vlm'])}/{len(results)} questions")

# Demo usage
def demo_hybrid_approach():
    """Demonstrate hybrid VLM approach with test data."""
    
    # Create test image with checkboxes
    test_image = np.ones((600, 800, 3), dtype=np.uint8) * 255
    
    # Draw checkboxes with labels
    cv2.rectangle(test_image, (100, 100), (120, 120), (0, 0, 0), 2)  # Empty
    cv2.putText(test_image, "Training Required", (130, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.rectangle(test_image, (100, 200), (120, 220), (0, 0, 0), -1)  # Filled  
    cv2.putText(test_image, "Background Check", (130, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    cv2.rectangle(test_image, (100, 300), (120, 320), (0, 0, 0), 2)   # Empty
    cv2.putText(test_image, "Medical Exam", (130, 315), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Test questions
    questions = [
        {"id": 1, "key": "Is training required?"},
        {"id": 2, "key": "Which options are selected in the medical section?"},
        {"id": 3, "key": "Are background checks completed?"}
    ]
    
    # Initialize processor (add your OpenAI API key for real VLM)
    processor = HybridCheckboxProcessor("best.pt", openai_api_key=None)  # Uses mock VLM
    
    # Process document
    results = processor.process_document_hybrid(test_image, questions)
    
    # Visualize
    checkboxes = processor.detect_checkboxes(test_image)
    processor.visualize_results(test_image, checkboxes, results)
    
    return results

# Run the demo
print("🚀 Starting Hybrid VLM Demo...")
demo_results = demo_hybrid_approach()

print("\n✅ Hybrid VLM integration test complete!")
print("🔧 To use real VLM: Add your OpenAI API key to the processor initialization")
```

## Instructions:
1. Upload your `best.pt` model to Colab
2. Run the script above
3. Optional: Add your OpenAI API key for real VLM testing
4. This demonstrates the hybrid approach: detection + VLM reasoning

The hybrid approach gives you the best of both worlds - reliable detection plus sophisticated reasoning!