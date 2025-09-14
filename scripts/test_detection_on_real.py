#!/usr/bin/env python3
"""
Test our detection model (90% synthetic trained) on real CheckboxQA documents.
This will show us how well synthetic training generalizes.
"""

import json
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import requests
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict

class DetectionTester:
    """Test detection model on real CheckboxQA documents."""
    
    def __init__(self, model_path: str = "models/best.pt"):
        print(f"📦 Loading detection model: {model_path}")
        self.model = YOLO(model_path)
        self.results = []
        
    def download_sample_pdfs(self, num_docs: int = 5) -> List[Path]:
        """Download a few CheckboxQA PDFs for testing."""
        url_map_file = Path("data/raw/checkboxqa/data/document_url_map.json")
        pdfs_dir = Path("data/checkboxqa_test_pdfs")
        pdfs_dir.mkdir(exist_ok=True)
        
        if not url_map_file.exists():
            print(f"❌ URL map not found: {url_map_file}")
            return []
        
        # Load URL map
        with open(url_map_file, 'r') as f:
            url_map = json.load(f)
        
        downloaded = []
        items = list(url_map.items())[:num_docs]
        
        for doc_id, doc_info in tqdm(items, desc="Downloading test PDFs"):
            pdf_path = pdfs_dir / f"{doc_id}.pdf"
            
            # Skip if exists
            if pdf_path.exists() and pdf_path.stat().st_size > 1000:
                downloaded.append(pdf_path)
                continue
            
            # Download
            pdf_url = doc_info.get('pdf_url')
            if pdf_url:
                try:
                    response = requests.get(pdf_url, timeout=30)
                    if response.status_code == 200:
                        with open(pdf_path, 'wb') as f:
                            f.write(response.content)
                        downloaded.append(pdf_path)
                        print(f"✅ Downloaded {doc_id}")
                except Exception as e:
                    print(f"⚠️ Failed to download {doc_id}: {e}")
        
        return downloaded
    
    def pdf_to_images(self, pdf_path: Path) -> List[np.ndarray]:
        """Convert PDF to images."""
        try:
            import fitz
            images = []
            pdf_document = fitz.open(str(pdf_path))
            
            for page_num in range(min(3, len(pdf_document))):  # Max 3 pages
                page = pdf_document[page_num]
                mat = fitz.Matrix(150/72.0, 150/72.0)  # 150 DPI
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                img_array = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                if img_array is not None:
                    images.append(img_array)
            
            pdf_document.close()
            return images
            
        except Exception as e:
            print(f"⚠️ Error converting PDF {pdf_path}: {e}")
            return []
    
    def test_detection_on_image(self, image: np.ndarray, image_name: str) -> Dict:
        """Test detection on a single image."""
        results = self.model(image, conf=0.3)
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = box.conf[0].item()
                    
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': conf,
                        'area': (x2 - x1) * (y2 - y1)
                    })
        
        return {
            'image_name': image_name,
            'num_detections': len(detections),
            'detections': detections,
            'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0,
            'min_confidence': min([d['confidence'] for d in detections]) if detections else 0.0,
            'max_confidence': max([d['confidence'] for d in detections]) if detections else 0.0
        }
    
    def visualize_detections(self, image: np.ndarray, detections: List[Dict], output_path: Path):
        """Save image with detection visualizations."""
        vis_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            conf = detection['confidence']
            
            # Draw bounding box
            color = (0, 255, 0) if conf > 0.7 else (0, 165, 255) if conf > 0.5 else (0, 0, 255)
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)
            
            # Draw confidence
            cv2.putText(vis_image, f"{conf:.2f}", (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imwrite(str(output_path), vis_image)
    
    def run_detection_test(self, num_test_docs: int = 5):
        """Run detection test on CheckboxQA documents."""
        print(f"🧪 Testing detection model on {num_test_docs} real documents...")
        
        # Download test PDFs
        pdf_paths = self.download_sample_pdfs(num_test_docs)
        
        if not pdf_paths:
            print("❌ No PDFs downloaded for testing")
            return
        
        # Create output directory
        output_dir = Path("detection_test_results")
        output_dir.mkdir(exist_ok=True)
        
        # Test each document
        all_results = []
        
        for pdf_path in pdf_paths:
            print(f"\n📄 Testing {pdf_path.name}...")
            
            # Convert to images
            images = self.pdf_to_images(pdf_path)
            
            for page_num, image in enumerate(images):
                image_name = f"{pdf_path.stem}_page_{page_num}"
                
                # Run detection
                result = self.test_detection_on_image(image, image_name)
                all_results.append(result)
                
                # Save visualization
                if result['detections']:
                    vis_path = output_dir / f"{image_name}_detections.png"
                    self.visualize_detections(image, result['detections'], vis_path)
                    print(f"   📊 Page {page_num}: {result['num_detections']} detections (avg conf: {result['avg_confidence']:.2f})")
                else:
                    print(f"   ⚠️ Page {page_num}: No detections")
        
        # Analyze results
        self.analyze_detection_results(all_results, output_dir)
        
        return all_results
    
    def analyze_detection_results(self, results: List[Dict], output_dir: Path):
        """Analyze and report detection performance."""
        if not results:
            return
        
        # Calculate statistics
        total_detections = sum(r['num_detections'] for r in results)
        total_images = len(results)
        images_with_detections = len([r for r in results if r['num_detections'] > 0])
        
        all_confidences = []
        for r in results:
            all_confidences.extend([d['confidence'] for d in r['detections']])
        
        print(f"\n" + "="*50)
        print("🔍 DETECTION TEST RESULTS")
        print("="*50)
        print(f"Images tested: {total_images}")
        print(f"Images with detections: {images_with_detections} ({images_with_detections/total_images:.1%})")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per image: {total_detections/total_images:.1f}")
        
        if all_confidences:
            print(f"\nConfidence Statistics:")
            print(f"  Average: {np.mean(all_confidences):.3f}")
            print(f"  Minimum: {np.min(all_confidences):.3f}")  
            print(f"  Maximum: {np.max(all_confidences):.3f}")
            print(f"  High confidence (>0.7): {len([c for c in all_confidences if c > 0.7])/len(all_confidences):.1%}")
            print(f"  Low confidence (<0.5): {len([c for c in all_confidences if c < 0.5])/len(all_confidences):.1%}")
        
        # Distribution by document
        print(f"\nDetections by Document:")
        for r in results:
            if r['num_detections'] > 0:
                print(f"  {r['image_name']}: {r['num_detections']} (avg conf: {r['avg_confidence']:.2f})")
        
        print("="*50)
        print(f"📁 Visualizations saved to: {output_dir}")
        
        # Assessment
        if images_with_detections / total_images > 0.7 and np.mean(all_confidences or [0]) > 0.6:
            print("✅ GOOD: Detection model generalizes well to real documents")
        elif images_with_detections / total_images > 0.5:
            print("⚠️ MODERATE: Some detection issues, but workable")
        else:
            print("❌ POOR: Detection model struggles on real documents")
            print("   Consider: More real training data, domain adaptation")
        
        # Save detailed results
        results_file = output_dir / "detection_test_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'total_images': total_images,
                    'images_with_detections': images_with_detections,
                    'total_detections': total_detections,
                    'avg_detections_per_image': total_detections/total_images,
                    'avg_confidence': np.mean(all_confidences) if all_confidences else 0.0
                },
                'detailed_results': results
            }, f, indent=2)
        
        print(f"💾 Detailed results saved to: {results_file}")

def main():
    """Main test function."""
    tester = DetectionTester("models/best.pt")
    results = tester.run_detection_test(num_test_docs=3)  # Start with 3 documents
    
    print(f"\n🎯 Next Steps:")
    print(f"1. Review visualizations in detection_test_results/")
    print(f"2. If performance is good, proceed with VLM integration")
    print(f"3. If performance is poor, consider domain adaptation")

if __name__ == "__main__":
    main()