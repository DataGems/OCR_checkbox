#!/usr/bin/env python3
"""
Process CheckboxQA PDFs to create labeled classification dataset.

1. Download PDFs from CheckboxQA
2. Run detection model to find checkboxes
3. Match detections to Q&A labels
4. Create classification training dataset
"""

import json
import requests
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import fitz  # PyMuPDF
from ultralytics import YOLO
import argparse

class CheckboxQAProcessor:
    """Process CheckboxQA PDFs to extract labeled checkbox crops."""
    
    def __init__(self, data_dir: str = "data/raw/checkboxqa"):
        self.data_dir = Path(data_dir)
        self.pdfs_dir = self.data_dir / "pdfs"
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        
        self.crops_dir = Path("data/checkboxqa_crops")
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        
        # Load data files
        self.url_map = self._load_url_map()
        self.qa_data = self._load_qa_data()
        
    def _load_url_map(self) -> Dict:
        """Load document URL mapping."""
        url_file = self.data_dir / "data" / "document_url_map.json"
        if url_file.exists():
            with open(url_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _load_qa_data(self) -> Dict[str, List]:
        """Load Q&A annotations by document."""
        qa_file = self.data_dir / "data" / "gold.jsonl"
        qa_by_doc = {}
        
        if qa_file.exists():
            with open(qa_file, 'r') as f:
                for line in f:
                    data = json.loads(line.strip())
                    doc_name = data['name']
                    qa_by_doc[doc_name] = data['annotations']
        
        return qa_by_doc
    
    def download_pdfs(self, limit: Optional[int] = None) -> List[Path]:
        """Download PDFs from DocumentCloud."""
        print(f"📥 Downloading CheckboxQA PDFs...")
        
        downloaded = []
        items = list(self.url_map.items())[:limit] if limit else self.url_map.items()
        
        for doc_id, doc_info in tqdm(items, desc="Downloading PDFs"):
            pdf_path = self.pdfs_dir / f"{doc_id}.pdf"
            
            # Skip if already downloaded
            if pdf_path.exists():
                downloaded.append(pdf_path)
                continue
            
            # Download PDF
            pdf_url = doc_info.get('pdf_url')
            if pdf_url:
                try:
                    response = requests.get(pdf_url, timeout=30)
                    if response.status_code == 200:
                        with open(pdf_path, 'wb') as f:
                            f.write(response.content)
                        downloaded.append(pdf_path)
                except Exception as e:
                    print(f"⚠️ Error downloading {doc_id}: {e}")
        
        print(f"✅ Downloaded {len(downloaded)} PDFs")
        return downloaded
    
    def pdf_to_images(self, pdf_path: Path, dpi: int = 150) -> List[np.ndarray]:
        """Convert PDF pages to images."""
        images = []
        
        try:
            pdf_document = fitz.open(str(pdf_path))
            
            for page_num in range(len(pdf_document)):
                page = pdf_document[page_num]
                
                # Render page to image
                mat = fitz.Matrix(dpi/72.0, dpi/72.0)
                pix = page.get_pixmap(matrix=mat)
                
                # Convert to numpy array
                img_data = pix.tobytes("png")
                img_array = cv2.imdecode(
                    np.frombuffer(img_data, np.uint8), 
                    cv2.IMREAD_COLOR
                )
                
                images.append(img_array)
            
            pdf_document.close()
            
        except Exception as e:
            print(f"⚠️ Error converting PDF {pdf_path}: {e}")
        
        return images
    
    def detect_checkboxes(self, 
                         model_path: str = "models/best.pt",
                         conf_threshold: float = 0.3) -> Dict[str, List]:
        """Detect checkboxes in all PDFs using trained model."""
        print(f"🔍 Detecting checkboxes in PDFs...")
        
        if not Path(model_path).exists():
            print(f"❌ Model not found: {model_path}")
            return {}
        
        model = YOLO(model_path)
        detections_by_doc = {}
        
        pdf_files = list(self.pdfs_dir.glob("*.pdf"))
        
        for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
            doc_id = pdf_path.stem
            doc_detections = []
            
            # Convert PDF to images
            images = self.pdf_to_images(pdf_path)
            
            # Detect checkboxes in each page
            for page_num, image in enumerate(images):
                results = model(image, conf=conf_threshold)
                
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for i, box in enumerate(boxes):
                            x1, y1, x2, y2 = box.xyxy[0].tolist()
                            conf = box.conf[0].item()
                            
                            # Extract checkbox crop
                            crop = image[int(y1):int(y2), int(x1):int(x2)]
                            
                            if crop.size > 0:
                                doc_detections.append({
                                    'page': page_num,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': conf,
                                    'crop': crop,
                                    'crop_id': f"{doc_id}_p{page_num}_cb{i}"
                                })
            
            if doc_detections:
                detections_by_doc[doc_id] = doc_detections
        
        print(f"✅ Detected checkboxes in {len(detections_by_doc)} documents")
        return detections_by_doc
    
    def classify_answer_to_state(self, answer: str) -> str:
        """Convert Q&A answer to checkbox state."""
        answer_lower = answer.lower().strip()
        
        # Clear mappings
        if answer_lower in ['yes', 'y', 'true', '1', 'checked', 'x']:
            return 'checked'
        elif answer_lower in ['no', 'n', 'false', '0', 'unchecked', '']:
            return 'unchecked'
        elif answer_lower in ['na', 'n/a', 'not applicable', 'unclear']:
            return 'unclear'
        
        # For selection questions with specific values
        if answer and len(answer) > 0:
            return 'checked'
        
        return 'unclear'
    
    def create_classification_dataset(self, detections: Dict[str, List]) -> Dict:
        """Create labeled classification dataset from detections and Q&A."""
        print(f"🏷️ Creating labeled classification dataset...")
        
        labeled_crops = {
            'checked': [],
            'unchecked': [],
            'unclear': []
        }
        
        stats = {
            'total_documents': 0,
            'total_checkboxes': 0,
            'labeled_checkboxes': 0,
            'unlabeled_checkboxes': 0
        }
        
        for doc_id, doc_detections in detections.items():
            stats['total_documents'] += 1
            
            # Get Q&A annotations for this document
            qa_annotations = self.qa_data.get(doc_id, [])
            
            # For now, apply labels heuristically
            # In a full implementation, we'd match specific checkboxes to questions
            for detection in doc_detections:
                stats['total_checkboxes'] += 1
                
                # Simple heuristic: distribute based on Q&A answer patterns
                # This is a simplification - ideally we'd match spatially
                if qa_annotations:
                    # Use first few Q&A pairs to label checkboxes
                    for i, qa in enumerate(qa_annotations[:len(doc_detections)]):
                        if i == doc_detections.index(detection):
                            answer = qa['values'][0]['value'] if qa['values'] else ''
                            state = self.classify_answer_to_state(answer)
                            
                            labeled_crops[state].append({
                                'crop': detection['crop'],
                                'crop_id': detection['crop_id'],
                                'document': doc_id,
                                'page': detection['page'],
                                'confidence': detection['confidence'],
                                'question': qa.get('key', ''),
                                'answer': answer
                            })
                            stats['labeled_checkboxes'] += 1
                            break
                else:
                    stats['unlabeled_checkboxes'] += 1
        
        # Save crops to disk
        for state, crops in labeled_crops.items():
            state_dir = self.crops_dir / state
            state_dir.mkdir(exist_ok=True)
            
            for crop_data in crops:
                crop_path = state_dir / f"{crop_data['crop_id']}.png"
                cv2.imwrite(str(crop_path), crop_data['crop'])
        
        print(f"\n📊 Classification Dataset Statistics:")
        print(f"Documents processed: {stats['total_documents']}")
        print(f"Total checkboxes detected: {stats['total_checkboxes']}")
        print(f"Labeled checkboxes: {stats['labeled_checkboxes']}")
        print(f"Unlabeled checkboxes: {stats['unlabeled_checkboxes']}")
        print(f"\nLabeled crops by class:")
        for state, crops in labeled_crops.items():
            print(f"  {state}: {len(crops)}")
        
        return labeled_crops
    
    def run_full_pipeline(self, 
                         download_limit: Optional[int] = None,
                         model_path: str = "models/best.pt"):
        """Run complete CheckboxQA processing pipeline."""
        print("🚀 Processing CheckboxQA PDFs for classification dataset\n")
        
        # Step 1: Download PDFs
        pdfs = self.download_pdfs(limit=download_limit)
        
        # Step 2: Detect checkboxes
        detections = self.detect_checkboxes(model_path=model_path)
        
        # Step 3: Create labeled dataset
        labeled_data = self.create_classification_dataset(detections)
        
        print(f"\n✅ CheckboxQA processing complete!")
        print(f"📁 Labeled crops saved to: {self.crops_dir}")
        
        # Save metadata
        metadata = {
            'num_documents': len(pdfs),
            'detections_by_doc': {
                doc_id: len(dets) for doc_id, dets in detections.items()
            },
            'labeled_crops': {
                state: len(crops) for state, crops in labeled_data.items()
            }
        }
        
        metadata_path = self.crops_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return labeled_data

def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description="Process CheckboxQA PDFs")
    parser.add_argument("--data-dir", type=str, default="data/raw/checkboxqa",
                       help="CheckboxQA data directory")
    parser.add_argument("--model", type=str, default="models/best.pt",
                       help="Detection model path")
    parser.add_argument("--download-limit", type=int, default=10,
                       help="Limit number of PDFs to download (None for all)")
    
    args = parser.parse_args()
    
    # Process CheckboxQA
    processor = CheckboxQAProcessor(data_dir=args.data_dir)
    processor.run_full_pipeline(
        download_limit=args.download_limit,
        model_path=args.model
    )
    
    return 0

if __name__ == "__main__":
    exit(main())