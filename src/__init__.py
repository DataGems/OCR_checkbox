"""
OCR Checkbox Detection & Classification Pipeline

A production-ready system for detecting and classifying checkbox states 
in PDF documents using computer vision and machine learning.
"""

__version__ = "0.1.0"
__author__ = "David Pepper"

# Core modules
from . import detection
from . import classification  
from . import pipeline
from . import utils

__all__ = [
    "detection",
    "classification", 
    "pipeline",
    "utils",
]
