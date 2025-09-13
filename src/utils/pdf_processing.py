"""
PDF Processing utilities for checkbox detection pipeline.
Handles PDF to image conversion, preprocessing, and basic image operations.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union
from pdf2image import convert_from_path
import fitz  # PyMuPDF
from PIL import Image


def pdf_to_images(
    pdf_path: Union[str, Path], 
    dpi: int = 300,
    first_page: Optional[int] = None,
    last_page: Optional[int] = None
) -> List[np.ndarray]:
    """
    Convert PDF pages to images using pdf2image.
    
    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (300 recommended for quality/speed balance)
        first_page: First page to convert (1-indexed)
        last_page: Last page to convert (1-indexed)
    
    Returns:
        List of images as numpy arrays (BGR format for OpenCV)
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Convert PDF to PIL images
        pil_images = convert_from_path(
            pdf_path, 
            dpi=dpi,
            first_page=first_page,
            last_page=last_page
        )
        
        # Convert PIL images to OpenCV format (BGR)
        cv_images = []
        for pil_img in pil_images:
            # Convert RGB to BGR for OpenCV
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            cv_images.append(cv_img)
            
        return cv_images
        
    except Exception as e:
        raise RuntimeError(f"Failed to convert PDF to images: {str(e)}")


def deskew_image(image: np.ndarray, max_angle: float = 10.0) -> np.ndarray:
    """
    Deskew an image using Hough line detection.
    
    Args:
        image: Input image (BGR)
        max_angle: Maximum rotation angle to consider (degrees)
    
    Returns:
        Deskewed image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Edge detection
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Hough line detection
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
    
    if lines is None:
        return image  # No lines detected, return original
    
    # Calculate rotation angles
    angles = []
    for line in lines[:20]:  # Use only first 20 lines for efficiency
        rho, theta = line[0]
        angle = theta * 180 / np.pi - 90
        
        # Only consider small angles (likely text lines)
        if abs(angle) <= max_angle:
            angles.append(angle)
    
    if not angles:
        return image  # No suitable angles found
    
    # Use median angle to avoid outliers
    rotation_angle = np.median(angles)
    
    # Rotate image
    if abs(rotation_angle) > 0.5:  # Only rotate if angle is significant
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Calculate new image size to avoid cropping
        cos_theta = abs(rotation_matrix[0, 0])
        sin_theta = abs(rotation_matrix[0, 1])
        new_w = int(h * sin_theta + w * cos_theta)
        new_h = int(h * cos_theta + w * sin_theta)
        
        # Adjust translation
        rotation_matrix[0, 2] += (new_w - w) / 2
        rotation_matrix[1, 2] += (new_h - h) / 2
        
        # Apply rotation
        rotated = cv2.warpAffine(
            image, rotation_matrix, (new_w, new_h), 
            flags=cv2.INTER_CUBIC, borderValue=(255, 255, 255)
        )
        return rotated
    
    return image


def denoise_image(image: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """
    Apply Gaussian denoising to image.
    
    Args:
        image: Input image
        sigma: Gaussian kernel standard deviation
    
    Returns:
        Denoised image
    """
    # Calculate kernel size based on sigma
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    kernel_size = max(3, kernel_size)  # Ensure minimum size of 3
    
    # Make kernel size odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)


def preprocess_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    apply_deskew: bool = True,
    apply_denoise: bool = True,
    denoise_sigma: float = 0.5
) -> np.ndarray:
    """
    Apply standard preprocessing pipeline to an image.
    
    Args:
        image: Input image
        target_size: Target size (width, height) for resizing
        apply_deskew: Whether to apply deskewing
        apply_denoise: Whether to apply denoising
        denoise_sigma: Sigma for Gaussian denoising
    
    Returns:
        Preprocessed image
    """
    processed = image.copy()
    
    # Deskew if requested
    if apply_deskew:
        processed = deskew_image(processed)
    
    # Denoise if requested
    if apply_denoise:
        processed = denoise_image(processed, sigma=denoise_sigma)
    
    # Resize if target size specified
    if target_size is not None:
        processed = cv2.resize(
            processed, target_size, interpolation=cv2.INTER_CUBIC
        )
    
    return processed


def save_images(
    images: List[np.ndarray], 
    output_dir: Union[str, Path], 
    prefix: str = "page"
) -> List[Path]:
    """
    Save list of images to directory.
    
    Args:
        images: List of images to save
        output_dir: Output directory
        prefix: Filename prefix
    
    Returns:
        List of saved file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    for i, image in enumerate(images):
        filename = f"{prefix}_{i+1:03d}.jpg"
        filepath = output_dir / filename
        
        # Save with high quality
        cv2.imwrite(str(filepath), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved_paths.append(filepath)
    
    return saved_paths


def extract_page_region(
    image: np.ndarray, 
    bbox: Tuple[int, int, int, int],
    padding: int = 5
) -> np.ndarray:
    """
    Extract a region from an image with optional padding.
    
    Args:
        image: Source image
        bbox: Bounding box (x1, y1, x2, y2)
        padding: Padding around the region
    
    Returns:
        Extracted region
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = bbox
    
    # Add padding and ensure bounds
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image[y1:y2, x1:x2]


def normalize_image_for_model(image: np.ndarray) -> np.ndarray:
    """
    Normalize image for model input (0-1 range, float32).
    
    Args:
        image: Input image (BGR, uint8)
    
    Returns:
        Normalized image (RGB, float32, 0-1 range)
    """
    # Convert BGR to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Normalize to 0-1 range
    normalized = rgb_image.astype(np.float32) / 255.0
    
    return normalized
