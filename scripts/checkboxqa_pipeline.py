#!/usr/bin/env python3
"""
Complete CheckboxQA pipeline: Detection → Classification → Question Answering.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pytesseract
from ultralytics import YOLO
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse

# Import our modules
from question_aware_processor import QuestionAwareProcessor, Checkbox, Question

@dataclass
class DocumentResult:
    """Results for a single document."""
    document_id: str
    checkboxes: List[Checkbox]
    ocr_text: Dict[int, List[Dict]]  # Page -> text regions
    predictions: Dict  # CheckboxQA format predictions

class CheckboxQAPipeline:
    """End-to-end pipeline for CheckboxQA task."""
    
    def __init__(self, 
                 detection_model_path: str = "models/best.pt",
                 classification_model_path: Optional[str] = None,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        
        print(f"🚀 Initializing CheckboxQA Pipeline...")
        
        # Load detection model
        self.detector = YOLO(detection_model_path)
        print(f"✅ Loaded detection model: {detection_model_path}")
        
        # Load classification model (if provided)
        self.classifier = None
        if classification_model_path and Path(classification_model_path).exists():
            self.classifier = self._load_classifier(classification_model_path, device)
            print(f"✅ Loaded classification model: {classification_model_path}")
        else:
            print("⚠️ No classification model - using heuristics")
        
        # Initialize question processor
        self.question_processor = QuestionAwareProcessor()
        
        # Classification transform
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.device = device
    
    def _load_classifier(self, model_path: str, device: str):
        """Load EfficientNet classifier."""
        try:
            from efficientnet_pytorch import EfficientNet
            
            model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=3)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()
            return model
        except Exception as e:
            print(f"⚠️ Error loading classifier: {e}")
            return None
    
    def detect_checkboxes(self, image: np.ndarray, page_num: int = 0) -> List[Checkbox]:
        """Detect checkboxes in an image."""
        results = self.detector(image, conf=0.3)
        checkboxes = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    
                    checkbox = Checkbox(
                        bbox=[int(x1), int(y1), int(x2), int(y2)],
                        state='unknown',  # Will be classified
                        confidence=conf,
                        page=page_num
                    )
                    checkboxes.append(checkbox)
        
        return checkboxes
    
    def classify_checkbox_state(self, image: np.ndarray, checkbox: Checkbox) -> str:
        """Classify checkbox state as checked/unchecked/unclear."""
        if self.classifier is None:
            # Simple heuristic if no classifier
            return self._classify_with_heuristics(image, checkbox)
        
        # Extract crop
        x1, y1, x2, y2 = checkbox.bbox
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return 'unclear'
        
        # Prepare for classification
        crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
        crop_tensor = self.transform(crop_pil).unsqueeze(0).to(self.device)
        
        # Classify
        with torch.no_grad():
            outputs = self.classifier(crop_tensor)
            _, predicted = outputs.max(1)
        
        classes = ['checked', 'unchecked', 'unclear']
        return classes[predicted.item()]
    
    def _classify_with_heuristics(self, image: np.ndarray, checkbox: Checkbox) -> str:
        """Simple heuristic classification based on pixel density."""
        x1, y1, x2, y2 = checkbox.bbox
        crop = image[y1:y2, x1:x2]
        
        if crop.size == 0:
            return 'unclear'
        
        # Convert to grayscale
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        
        # Calculate fill ratio
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        fill_ratio = np.sum(binary == 255) / binary.size
        
        # Classify based on fill ratio
        if fill_ratio > 0.3:
            return 'checked'
        elif fill_ratio < 0.1:
            return 'unchecked'
        else:
            return 'unclear'
    
    def extract_nearby_text(self, image: np.ndarray, checkbox: Checkbox, 
                           search_radius: int = 100) -> str:
        """Extract text near a checkbox using OCR."""
        x1, y1, x2, y2 = checkbox.bbox
        h, w = image.shape[:2]
        
        # Expand search region
        search_x1 = max(0, x1 - search_radius)
        search_y1 = max(0, y1 - 20)  # Less vertical expansion
        search_x2 = min(w, x2 + search_radius)
        search_y2 = min(h, y2 + 20)
        
        # Extract region
        region = image[search_y1:search_y2, search_x1:search_x2]
        
        if region.size == 0:
            return ""
        
        # OCR
        try:
            text = pytesseract.image_to_string(region, config='--psm 8').strip()
            return text
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
            return ""
    
    def perform_ocr(self, image: np.ndarray) -> List[Dict]:
        """Perform OCR on entire image to get text regions."""
        try:
            # Get detailed OCR data
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            text_regions = []
            n_boxes = len(ocr_data['text'])
            
            for i in range(n_boxes):
                if ocr_data['text'][i].strip():
                    text_regions.append({
                        'text': ocr_data['text'][i],
                        'bbox': [
                            ocr_data['left'][i],
                            ocr_data['top'][i],
                            ocr_data['left'][i] + ocr_data['width'][i],
                            ocr_data['top'][i] + ocr_data['height'][i]
                        ],
                        'confidence': ocr_data['conf'][i]
                    })
            
            return text_regions
        except Exception as e:
            print(f"⚠️ OCR error: {e}")
            return []
    
    def associate_text_with_checkboxes(self, checkboxes: List[Checkbox], 
                                     text_regions: List[Dict]) -> None:
        """Associate nearby text with each checkbox."""
        for checkbox in checkboxes:
            best_score = 0
            best_text = ""
            
            for text_region in text_regions:
                score = self.question_processor.spatial_proximity_score(
                    checkbox, text_region['bbox']
                )
                
                if score > best_score:
                    best_score = score
                    best_text = text_region['text']
            
            checkbox.nearby_text = best_text
    
    def process_image(self, image: np.ndarray, page_num: int = 0) -> Tuple[List[Checkbox], List[Dict]]:
        """Process a single image: detect, classify, and extract text."""
        # Detect checkboxes
        checkboxes = self.detect_checkboxes(image, page_num)
        
        # Classify states
        for checkbox in checkboxes:
            checkbox.state = self.classify_checkbox_state(image, checkbox)
        
        # Perform OCR
        text_regions = self.perform_ocr(image)
        
        # Associate text with checkboxes
        self.associate_text_with_checkboxes(checkboxes, text_regions)
        
        # Alternative: extract nearby text directly
        for checkbox in checkboxes:
            if not checkbox.nearby_text:
                checkbox.nearby_text = self.extract_nearby_text(image, checkbox)
        
        return checkboxes, text_regions
    
    def process_document(self, document_path: Path, questions: List[Question]) -> DocumentResult:
        """Process a complete document."""
        doc_id = document_path.stem
        all_checkboxes = []
        all_ocr_text = {}
        
        # Handle PDF vs image
        if document_path.suffix.lower() == '.pdf':
            # Convert PDF to images
            import fitz
            pdf_document = fitz.open(str(document_path))
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                mat = fitz.Matrix(150/72.0, 150/72.0)  # 150 DPI
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                
                # Process page
                checkboxes, text_regions = self.process_image(img_array, page_num)
                all_checkboxes.extend(checkboxes)
                all_ocr_text[page_num] = text_regions
            
            pdf_document.close()
        else:
            # Process single image
            image = cv2.imread(str(document_path))
            checkboxes, text_regions = self.process_image(image, 0)
            all_checkboxes.extend(checkboxes)
            all_ocr_text[0] = text_regions
        
        # Answer questions
        predictions = self.question_processor.process_document(doc_id, questions, all_checkboxes)
        
        return DocumentResult(
            document_id=doc_id,
            checkboxes=all_checkboxes,
            ocr_text=all_ocr_text,
            predictions=predictions
        )
    
    def process_checkboxqa_dataset(self, 
                                  documents_dir: Path,
                                  questions_file: Path,
                                  output_file: Path) -> Dict:
        """Process entire CheckboxQA dataset."""
        # Load questions
        questions_by_doc = {}
        with open(questions_file, 'r') as f:
            for line in f:
                data = json.loads(line.strip())
                doc_name = data['name']
                questions = [
                    Question(
                        id=ann['id'],
                        text=ann['key'],
                        question_type=self.question_processor.classify_question_type(ann['key'])
                    )
                    for ann in data['annotations']
                ]
                questions_by_doc[doc_name] = questions
        
        # Process documents
        results = []
        doc_files = list(documents_dir.glob("*.pdf")) + list(documents_dir.glob("*.png"))
        
        for doc_path in tqdm(doc_files, desc="Processing documents"):
            doc_name = doc_path.stem
            
            if doc_name in questions_by_doc:
                try:
                    result = self.process_document(doc_path, questions_by_doc[doc_name])
                    results.append(result.predictions)
                    
                    print(f"✅ Processed {doc_name}: {len(result.checkboxes)} checkboxes found")
                except Exception as e:
                    print(f"❌ Error processing {doc_name}: {e}")
        
        # Save results
        with open(output_file, 'w') as f:
            for result in results:
                f.write(json.dumps(result) + '\n')
        
        print(f"\n✅ Saved predictions to {output_file}")
        print(f"📊 Processed {len(results)} documents")
        
        return {
            'documents_processed': len(results),
            'output_file': str(output_file)
        }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="CheckboxQA Pipeline")
    parser.add_argument("--documents", type=str, required=True,
                       help="Directory containing CheckboxQA documents")
    parser.add_argument("--questions", type=str, required=True,
                       help="Path to questions JSONL file")
    parser.add_argument("--output", type=str, default="predictions.jsonl",
                       help="Output predictions file")
    parser.add_argument("--detection-model", type=str, default="models/best.pt",
                       help="Detection model path")
    parser.add_argument("--classification-model", type=str, default=None,
                       help="Classification model path (optional)")
    
    args = parser.parse_args()
    
    # Create pipeline
    pipeline = CheckboxQAPipeline(
        detection_model_path=args.detection_model,
        classification_model_path=args.classification_model
    )
    
    # Process dataset
    results = pipeline.process_checkboxqa_dataset(
        documents_dir=Path(args.documents),
        questions_file=Path(args.questions),
        output_file=Path(args.output)
    )
    
    # Run evaluation
    print("\n📊 Running evaluation...")
    import subprocess
    try:
        subprocess.run([
            "python", "evaluate.py",
            "--pred", args.output,
            "--gold", args.questions
        ], check=True)
    except Exception as e:
        print(f"⚠️ Could not run automatic evaluation: {e}")
        print("Run manually: python evaluate.py --pred predictions.jsonl --gold gold.jsonl")

if __name__ == "__main__":
    main()