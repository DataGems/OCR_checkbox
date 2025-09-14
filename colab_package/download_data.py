#!/usr/bin/env python3
"""
Data download script for checkbox detection datasets.
Downloads CheckboxQA, FUNSD, and other required datasets.
"""

import os
import sys
import requests
import zipfile
import tarfile
from pathlib import Path
import shutil
from typing import Optional
from tqdm import tqdm

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """
    Download file from URL with progress bar.
    
    Args:
        url: Download URL
        output_path: Local path to save file
        description: Description for progress bar
        
    Returns:
        True if successful, False otherwise
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(output_path, 'wb') as f, tqdm(
            desc=description,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        return True
        
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> bool:
    """
    Extract archive file.
    
    Args:
        archive_path: Path to archive file
        extract_to: Directory to extract to
        
    Returns:
        True if successful, False otherwise
    """
    try:
        extract_to.mkdir(parents=True, exist_ok=True)
        
        if archive_path.suffix.lower() == '.zip':
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        elif archive_path.suffix.lower() in ['.tar', '.gz']:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_to)
        else:
            print(f"Unsupported archive format: {archive_path.suffix}")
            return False
        
        return True
        
    except Exception as e:
        print(f"Error extracting {archive_path}: {str(e)}")
        return False


def download_checkboxqa(data_dir: Path) -> bool:
    """
    Download CheckboxQA dataset.
    
    Args:
        data_dir: Data directory
        
    Returns:
        True if successful, False otherwise
    """
    print("Downloading CheckboxQA dataset...")
    
    # CheckboxQA GitHub repository
    checkboxqa_url = "https://github.com/Snowflake-Labs/CheckboxQA/archive/refs/heads/main.zip"
    checkboxqa_zip = data_dir / "checkboxqa.zip"
    checkboxqa_dir = data_dir / "CheckboxQA"
    
    # Download
    success = download_file(
        checkboxqa_url,
        checkboxqa_zip,
        "CheckboxQA dataset"
    )
    
    if not success:
        return False
    
    # Extract
    if extract_archive(checkboxqa_zip, data_dir):
        # Move extracted files to proper location
        extracted_dir = data_dir / "CheckboxQA-main"
        if extracted_dir.exists():
            if checkboxqa_dir.exists():
                shutil.rmtree(checkboxqa_dir)
            extracted_dir.rename(checkboxqa_dir)
        
        # Clean up
        checkboxqa_zip.unlink()
        print(f"CheckboxQA downloaded and extracted to: {checkboxqa_dir}")
        return True
    
    return False


def download_funsd(data_dir: Path) -> bool:
    """
    Download FUNSD dataset.
    
    Args:
        data_dir: Data directory
        
    Returns:
        True if successful, False otherwise
    """
    print("Downloading FUNSD dataset...")
    
    funsd_base_url = "https://guillaumejaume.github.io/FUNSD"
    funsd_dir = data_dir / "FUNSD"
    
    files_to_download = [
        ("dataset.zip", "FUNSD dataset"),
    ]
    
    success = True
    for filename, description in files_to_download:
        url = f"{funsd_base_url}/{filename}"
        output_path = data_dir / filename
        
        if not download_file(url, output_path, description):
            success = False
            continue
        
        # Extract
        if not extract_archive(output_path, funsd_dir):
            success = False
        else:
            output_path.unlink()  # Clean up zip file
    
    if success:
        print(f"FUNSD downloaded and extracted to: {funsd_dir}")
    
    return success


def create_sample_data(data_dir: Path) -> bool:
    """
    Create sample synthetic data for testing.
    
    Args:
        data_dir: Data directory
        
    Returns:
        True if successful, False otherwise
    """
    print("Creating sample synthetic data...")
    
    try:
        import cv2
        import numpy as np
        
        sample_dir = data_dir / "sample"
        sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple sample image with checkboxes
        img = np.ones((600, 800, 3), dtype=np.uint8) * 255  # White background
        
        # Draw some sample checkboxes
        checkboxes = [
            (50, 50, 30, 30, False),   # Unchecked
            (50, 100, 30, 30, True),   # Checked
            (50, 150, 30, 30, False),  # Unchecked
        ]
        
        for x, y, w, h, checked in checkboxes:
            # Draw checkbox border
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            
            # Draw check mark if checked
            if checked:
                cv2.line(img, (x + 5, y + 15), (x + 12, y + 22), (0, 0, 0), 2)
                cv2.line(img, (x + 12, y + 22), (x + 25, y + 8), (0, 0, 0), 2)
        
        # Add some text
        cv2.putText(img, "Sample Form", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(img, "Option A", (90, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "Option B", (90, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "Option C", (90, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Save sample image
        sample_image_path = sample_dir / "sample_form.png"
        cv2.imwrite(str(sample_image_path), img)
        
        # Create a simple PDF using PIL (basic approach)
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        sample_pdf_path = sample_dir / "sample_form.pdf"
        pil_img.save(sample_pdf_path, "PDF", resolution=300.0)
        
        print(f"Sample data created in: {sample_dir}")
        return True
        
    except Exception as e:
        print(f"Error creating sample data: {str(e)}")
        return False


def setup_data_structure(data_dir: Path) -> None:
    """
    Setup standard data directory structure.
    
    Args:
        data_dir: Base data directory
    """
    subdirs = [
        "raw/checkboxqa",
        "raw/funsd", 
        "raw/rvl-cdip",
        "processed/train/images",
        "processed/train/labels", 
        "processed/val/images",
        "processed/val/labels",
        "processed/test/images",
        "processed/test/labels",
        "synthetic",
        "sample"
    ]
    
    for subdir in subdirs:
        (data_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print(f"Data directory structure created in: {data_dir}")


def main():
    """Main data download function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download checkbox detection datasets")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Data directory (default: data)"
    )
    parser.add_argument(
        "--checkboxqa",
        action="store_true",
        help="Download CheckboxQA dataset"
    )
    parser.add_argument(
        "--funsd", 
        action="store_true",
        help="Download FUNSD dataset"
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Create sample synthetic data"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all datasets and create samples"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    
    # Setup directory structure
    setup_data_structure(data_dir)
    
    # Determine what to download
    download_checkboxqa_flag = args.checkboxqa or args.all
    download_funsd_flag = args.funsd or args.all
    create_sample_flag = args.sample or args.all
    
    success_count = 0
    total_tasks = sum([download_checkboxqa_flag, download_funsd_flag, create_sample_flag])
    
    if total_tasks == 0:
        print("No datasets selected. Use --all or specify individual datasets.")
        print("Available options: --checkboxqa, --funsd, --sample, --all")
        return 1
    
    print(f"Starting download of {total_tasks} dataset(s)...")
    print()
    
    # Download datasets
    if download_checkboxqa_flag:
        if download_checkboxqa(data_dir / "raw"):
            success_count += 1
        print()
    
    if download_funsd_flag:
        if download_funsd(data_dir / "raw"):
            success_count += 1
        print()
    
    if create_sample_flag:
        if create_sample_data(data_dir):
            success_count += 1
        print()
    
    print("="*50)
    print(f"Download Summary: {success_count}/{total_tasks} successful")
    
    if success_count == total_tasks:
        print("All datasets downloaded successfully!")
        print(f"Data directory: {data_dir.absolute()}")
        return 0
    else:
        print("Some downloads failed. Check error messages above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
