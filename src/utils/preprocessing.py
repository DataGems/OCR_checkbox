"""
Advanced preprocessing utilities for checkbox detection.
Implements PDF processing, deskewing, denoising, and augmentation.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
import logging
from scipy import ndimage
from skimage import filters, morphology, measure
import albumentations as A

logger = logging.getLogger(__name__)


class AdvancedPreprocessor:
    """Advanced preprocessing for document images."""
    
    def __init__(self, target_dpi: int = 300):
        """
        Initialize preprocessor.
        
        Args:
            target_dpi: Target DPI for PDF conversion
        """
        self.target_dpi = target_dpi
        
        # Define augmentation pipeline
        self.augmentation_pipeline = self._create_augmentation_pipeline()
    
    def _create_augmentation_pipeline(self) -> A.Compose:
        """Create augmentation pipeline for training."""
        return A.Compose([
            # Rotation augmentation (-10° to +10°)
            A.Rotate(
                limit=10,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_CONSTANT,
                value=255,  # White background
                p=0.5
            ),
            
            # Noise injection
            A.OneOf([
                # Gaussian noise
                A.GaussNoise(var_limit=(10.0, 50.0), mean=0, p=1.0),
                # Salt and pepper noise
                A.MultiplicativeNoise(
                    multiplier=(0.9, 1.1),
                    elementwise=True,
                    p=1.0
                ),
                # ISO noise (simulates camera sensor noise)
                A.ISONoise(
                    color_shift=(0.01, 0.05),
                    intensity=(0.1, 0.5),
                    p=1.0
                ),
            ], p=0.5),
            
            # Contrast and brightness variations
            A.OneOf([
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    brightness_by_max=True,
                    p=1.0
                ),
                A.CLAHE(
                    clip_limit=4.0,
                    tile_grid_size=(8, 8),
                    p=1.0
                ),
                A.RandomGamma(
                    gamma_limit=(80, 120),
                    p=1.0
                ),
            ], p=0.5),
            
            # Blur (simulates scan quality variations)
            A.OneOf([
                A.Blur(blur_limit=3, p=1.0),
                A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0),
            ], p=0.3),
            
            # Optical distortions
            A.OneOf([
                A.OpticalDistortion(
                    distort_limit=0.05,
                    shift_limit=0.05,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255,
                    p=1.0
                ),
                A.GridDistortion(
                    num_steps=5,
                    distort_limit=0.1,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=255,
                    p=1.0
                ),
            ], p=0.3),
        ])
    
    def deskew_advanced(self, image: np.ndarray, angle_limit: float = 10) -> np.ndarray:
        """
        Advanced deskewing using multiple methods.
        
        Args:
            image: Input image
            angle_limit: Maximum rotation angle to consider
            
        Returns:
            Deskewed image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Method 1: Hough Line Transform
        angle_hough = self._detect_skew_hough(gray, angle_limit)
        
        # Method 2: Projection Profile
        angle_projection = self._detect_skew_projection(gray, angle_limit)
        
        # Method 3: Moments-based
        angle_moments = self._detect_skew_moments(gray)
        
        # Combine methods (weighted average)
        weights = [0.5, 0.3, 0.2]  # Hough is most reliable for documents
        angles = [angle_hough, angle_projection, angle_moments]
        
        # Filter out None values and extreme angles
        valid_angles = [(a, w) for a, w in zip(angles, weights) 
                       if a is not None and abs(a) < angle_limit]
        
        if not valid_angles:
            return image
        
        # Weighted average
        final_angle = sum(a * w for a, w in valid_angles) / sum(w for _, w in valid_angles)
        
        # Rotate image
        return self._rotate_image(image, final_angle)
    
    def _detect_skew_hough(self, gray: np.ndarray, angle_limit: float) -> Optional[float]:
        """Detect skew using Hough line transform."""
        try:
            # Edge detection
            edges = cv2.Canny(gray, 50, 150, apertureSize=3)
            
            # Hough transform
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi/180,
                threshold=100,
                minLineLength=100,
                maxLineGap=10
            )
            
            if lines is None:
                return None
            
            # Calculate angles
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Normalize to [-angle_limit, angle_limit]
                if -angle_limit <= angle <= angle_limit:
                    angles.append(angle)
            
            if not angles:
                return None
            
            # Return median angle
            return np.median(angles)
            
        except Exception as e:
            logger.error(f"Error in Hough skew detection: {e}")
            return None
    
    def _detect_skew_projection(self, gray: np.ndarray, angle_limit: float) -> Optional[float]:
        """Detect skew using projection profile variance."""
        try:
            best_angle = 0
            max_variance = 0
            
            # Test different angles
            for angle in np.linspace(-angle_limit, angle_limit, 41):
                # Rotate image
                rotated = ndimage.rotate(gray, angle, reshape=False, order=1)
                
                # Calculate horizontal projection
                projection = np.sum(rotated, axis=1)
                
                # Calculate variance
                variance = np.var(projection)
                
                if variance > max_variance:
                    max_variance = variance
                    best_angle = angle
            
            return best_angle
            
        except Exception as e:
            logger.error(f"Error in projection skew detection: {e}")
            return None
    
    def _detect_skew_moments(self, gray: np.ndarray) -> Optional[float]:
        """Detect skew using image moments."""
        try:
            # Threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            
            # Calculate minimum area rectangle
            rect = cv2.minAreaRect(largest)
            angle = rect[2]
            
            # Normalize angle
            if angle < -45:
                angle += 90
            elif angle > 45:
                angle -= 90
            
            return angle
            
        except Exception as e:
            logger.error(f"Error in moments skew detection: {e}")
            return None
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by given angle."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new dimensions
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotate with white background
        return cv2.warpAffine(
            image, M, (new_w, new_h),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255) if len(image.shape) == 3 else 255
        )
    
    def denoise_advanced(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced denoising using multiple techniques.
        
        Args:
            image: Input image
            
        Returns:
            Denoised image
        """
        # Convert to grayscale for processing
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = image.copy()
            is_color = False
        
        # 1. Morphological denoising (remove small dots)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        morph = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # 2. Non-local means denoising
        denoised = cv2.fastNlMeansDenoising(morph, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 3. Bilateral filter (preserves edges)
        bilateral = cv2.bilateralFilter(denoised, 5, 50, 50)
        
        # 4. Remove salt and pepper noise
        median = cv2.medianBlur(bilateral, 3)
        
        # Apply back to color image if needed
        if is_color:
            # Create mask of changes
            diff = cv2.absdiff(gray, median)
            _, mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)
            
            # Apply denoising to color channels
            result = image.copy()
            for i in range(3):
                result[:, :, i] = np.where(mask > 0, median, image[:, :, i])
            return result
        else:
            return median
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast for better checkbox detection.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        # Convert to LAB color space if color
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            enhanced = cv2.merge([l, a, b])
            return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
        else:
            # Apply CLAHE to grayscale
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def remove_shadows(self, image: np.ndarray) -> np.ndarray:
        """
        Remove shadows from scanned documents.
        
        Args:
            image: Input image
            
        Returns:
            Shadow-removed image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Large kernel for background estimation
        dilated = cv2.dilate(gray, np.ones((7, 7), np.uint8))
        bg = cv2.medianBlur(dilated, 21)
        
        # Difference and normalize
        diff = 255 - cv2.absdiff(gray, bg)
        norm = cv2.normalize(diff, None, alpha=0, beta=255, 
                           norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        
        # Apply back to color if needed
        if len(image.shape) == 3:
            # Use normalized grayscale as correction factor
            factor = norm.astype(np.float32) / gray.astype(np.float32)
            factor = np.clip(factor, 0.5, 2.0)  # Limit correction
            
            result = image.astype(np.float32)
            for i in range(3):
                result[:, :, i] *= factor
            
            return np.clip(result, 0, 255).astype(np.uint8)
        else:
            return norm
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image for model input.
        
        Args:
            image: Input image
            
        Returns:
            Normalized image
        """
        # Ensure image is in [0, 255] range
        if image.dtype != np.uint8:
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image
    
    def preprocess_document(self, image: np.ndarray, 
                          apply_deskew: bool = True,
                          apply_denoise: bool = True,
                          apply_contrast: bool = True,
                          apply_shadows: bool = True) -> np.ndarray:
        """
        Apply full preprocessing pipeline to document image.
        
        Args:
            image: Input image
            apply_deskew: Whether to apply deskewing
            apply_denoise: Whether to apply denoising
            apply_contrast: Whether to enhance contrast
            apply_shadows: Whether to remove shadows
            
        Returns:
            Preprocessed image
        """
        result = image.copy()
        
        # 1. Deskew
        if apply_deskew:
            result = self.deskew_advanced(result)
        
        # 2. Remove shadows
        if apply_shadows:
            result = self.remove_shadows(result)
        
        # 3. Denoise
        if apply_denoise:
            result = self.denoise_advanced(result)
        
        # 4. Enhance contrast
        if apply_contrast:
            result = self.enhance_contrast(result)
        
        # 5. Normalize
        result = self.normalize_image(result)
        
        return result
    
    def augment_image(self, image: np.ndarray, bboxes: Optional[List[List[float]]] = None) -> Tuple[np.ndarray, Optional[List[List[float]]]]:
        """
        Apply augmentation to image and bounding boxes.
        
        Args:
            image: Input image
            bboxes: List of bounding boxes in format [x, y, width, height]
            
        Returns:
            Augmented image and transformed bboxes
        """
        if bboxes:
            # Convert to albumentations format
            transformed = self.augmentation_pipeline(
                image=image,
                bboxes=bboxes,
                bbox_params=A.BboxParams(format='coco', label_fields=[])
            )
            return transformed['image'], transformed['bboxes']
        else:
            transformed = self.augmentation_pipeline(image=image)
            return transformed['image'], None