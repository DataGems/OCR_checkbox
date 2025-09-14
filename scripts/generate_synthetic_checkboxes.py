#!/usr/bin/env python3
"""
Generate synthetic checkbox data for training augmentation.
Creates checkboxes with various states, fonts, styles, and rotations.
"""

import os
import sys
import cv2
import numpy as np
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from tqdm import tqdm

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class CheckboxStyle:
    """Configuration for checkbox appearance."""
    size: int  # Size of checkbox in pixels
    border_width: int
    border_color: Tuple[int, int, int]  # RGB
    fill_color: Optional[Tuple[int, int, int]] = None
    check_style: str = "checkmark"  # checkmark, x, filled
    check_width: int = 2
    check_color: Tuple[int, int, int] = (0, 0, 0)
    rounded_corners: bool = False
    corner_radius: int = 3


class SyntheticCheckboxGenerator:
    """Generate synthetic checkboxes with various styles and states."""
    
    def __init__(self, output_dir: str = "data/synthetic"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        self.images_dir = self.output_dir / "images"
        self.labels_dir = self.output_dir / "labels"
        self.crops_dir = self.output_dir / "crops"
        
        for dir in [self.images_dir, self.labels_dir, self.crops_dir]:
            dir.mkdir(exist_ok=True)
        
        # Define checkbox variations
        self.sizes = [20, 25, 30, 35, 40]  # Common checkbox sizes
        self.border_widths = [1, 2, 3]
        self.check_styles = ["checkmark", "x", "filled", "dot"]
        self.rotations = [-10, -5, 0, 5, 10]  # Rotation angles
        
        # Common checkbox colors
        self.border_colors = [
            (0, 0, 0),      # Black
            (50, 50, 50),   # Dark gray
            (100, 100, 100), # Gray
            (0, 0, 255),    # Blue
        ]
        
        # Background patterns (to simulate different form backgrounds)
        self.bg_patterns = ["white", "light_gray", "grid", "noisy", "textured"]
        
    def generate_checkbox(self, style: CheckboxStyle, state: str) -> np.ndarray:
        """
        Generate a single checkbox image.
        
        Args:
            style: Checkbox appearance configuration
            state: "checked", "unchecked", or "unclear"
            
        Returns:
            Checkbox image as numpy array
        """
        # Create image with padding
        padding = 10
        img_size = style.size + 2 * padding
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
        
        # Convert to PIL for better drawing capabilities
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Draw checkbox border
        box_coords = [padding, padding, padding + style.size, padding + style.size]
        
        if style.rounded_corners:
            # Draw rounded rectangle
            draw.rounded_rectangle(box_coords, radius=style.corner_radius,
                                 outline=style.border_color, width=style.border_width)
        else:
            # Draw regular rectangle
            draw.rectangle(box_coords, outline=style.border_color, width=style.border_width)
        
        # Fill if specified
        if style.fill_color:
            inner_coords = [box_coords[0] + style.border_width,
                          box_coords[1] + style.border_width,
                          box_coords[2] - style.border_width,
                          box_coords[3] - style.border_width]
            draw.rectangle(inner_coords, fill=style.fill_color)
        
        # Draw check mark if checked
        if state == "checked":
            self._draw_check(draw, box_coords, style)
        elif state == "unclear":
            # Add some ambiguous marking
            self._draw_unclear(draw, box_coords, style)
        
        # Convert back to numpy
        img = np.array(pil_img)
        
        return img
    
    def _draw_check(self, draw: ImageDraw, box_coords: List[int], style: CheckboxStyle):
        """Draw check mark inside checkbox."""
        x1, y1, x2, y2 = box_coords
        margin = style.size // 5
        
        if style.check_style == "checkmark":
            # Draw traditional checkmark
            points = [
                (x1 + margin, y1 + style.size // 2),
                (x1 + style.size // 3, y2 - margin),
                (x2 - margin, y1 + margin)
            ]
            draw.line(points, fill=style.check_color, width=style.check_width)
            
        elif style.check_style == "x":
            # Draw X mark
            draw.line([(x1 + margin, y1 + margin), (x2 - margin, y2 - margin)],
                     fill=style.check_color, width=style.check_width)
            draw.line([(x2 - margin, y1 + margin), (x1 + margin, y2 - margin)],
                     fill=style.check_color, width=style.check_width)
            
        elif style.check_style == "filled":
            # Fill the entire checkbox
            inner_margin = style.border_width + 2
            draw.rectangle([x1 + inner_margin, y1 + inner_margin,
                          x2 - inner_margin, y2 - inner_margin],
                          fill=style.check_color)
            
        elif style.check_style == "dot":
            # Draw a dot/circle in center
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            radius = style.size // 4
            draw.ellipse([center_x - radius, center_y - radius,
                        center_x + radius, center_y + radius],
                        fill=style.check_color)
    
    def _draw_unclear(self, draw: ImageDraw, box_coords: List[int], style: CheckboxStyle):
        """Draw unclear/ambiguous marking."""
        x1, y1, x2, y2 = box_coords
        
        # Add partial or faded marking
        if random.choice([True, False]):
            # Partial checkmark
            margin = style.size // 5
            points = [
                (x1 + margin, y1 + style.size // 2),
                (x1 + style.size // 3, y2 - margin)
            ]
            # Use lighter color for unclear state
            unclear_color = tuple(min(255, c + 150) for c in style.check_color)
            draw.line(points, fill=unclear_color, width=style.check_width)
        else:
            # Scribble or smudge
            for _ in range(3):
                x_start = random.randint(x1 + 5, x2 - 5)
                y_start = random.randint(y1 + 5, y2 - 5)
                x_end = random.randint(x1 + 5, x2 - 5)
                y_end = random.randint(y1 + 5, y2 - 5)
                draw.line([(x_start, y_start), (x_end, y_end)],
                         fill=(200, 200, 200), width=1)
    
    def apply_transformations(self, img: np.ndarray, rotation_angle: float) -> np.ndarray:
        """Apply transformations like rotation, noise, blur."""
        # Rotate if needed
        if rotation_angle != 0:
            center = (img.shape[1] // 2, img.shape[0] // 2)
            matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            img = cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]),
                               borderValue=(255, 255, 255))
        
        # Add noise randomly
        if random.random() < 0.3:
            noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
            img = cv2.add(img, noise)
        
        # Add slight blur randomly
        if random.random() < 0.2:
            img = cv2.GaussianBlur(img, (3, 3), 0.5)
        
        return img
    
    def create_form_background(self, width: int, height: int, pattern: str) -> np.ndarray:
        """Create a form-like background."""
        if pattern == "white":
            bg = np.ones((height, width, 3), dtype=np.uint8) * 255
        elif pattern == "light_gray":
            bg = np.ones((height, width, 3), dtype=np.uint8) * 240
        elif pattern == "grid":
            bg = np.ones((height, width, 3), dtype=np.uint8) * 250
            # Add grid lines
            for i in range(0, width, 50):
                cv2.line(bg, (i, 0), (i, height), (220, 220, 220), 1)
            for i in range(0, height, 50):
                cv2.line(bg, (0, i), (width, i), (220, 220, 220), 1)
        elif pattern == "noisy":
            bg = np.ones((height, width, 3), dtype=np.uint8) * 255
            noise = np.random.normal(0, 5, bg.shape).astype(np.uint8)
            bg = cv2.add(bg, noise)
        else:  # textured
            bg = np.random.randint(240, 255, (height, width, 3), dtype=np.uint8)
        
        return bg
    
    def generate_form_with_checkboxes(self, form_id: int, num_checkboxes: int = 5) -> Dict:
        """
        Generate a complete form image with multiple checkboxes.
        
        Returns:
            Dictionary with image path and annotations
        """
        # Form dimensions
        width, height = 800, 1000
        
        # Create background
        bg_pattern = random.choice(self.bg_patterns)
        form_img = self.create_form_background(width, height, bg_pattern)
        
        # Convert to PIL for text rendering
        pil_img = Image.fromarray(form_img)
        draw = ImageDraw.Draw(pil_img)
        
        # Add form title
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
            title_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 32)
        except:
            font = ImageFont.load_default()
            title_font = font
        
        draw.text((50, 30), "Sample Form " + str(form_id), fill=(0, 0, 0), font=title_font)
        
        # Generate checkboxes with labels
        annotations = []
        y_position = 100
        
        for i in range(num_checkboxes):
            # Random checkbox style
            style = CheckboxStyle(
                size=random.choice(self.sizes),
                border_width=random.choice(self.border_widths),
                border_color=random.choice(self.border_colors),
                check_style=random.choice(self.check_styles),
                check_width=random.randint(2, 4),
                rounded_corners=random.choice([True, False])
            )
            
            # Random state
            state = random.choice(["checked", "unchecked", "unclear"])
            
            # Generate checkbox
            checkbox_img = self.generate_checkbox(style, state)
            rotation = random.choice(self.rotations)
            checkbox_img = self.apply_transformations(checkbox_img, rotation)
            
            # Place on form
            x_position = 50
            
            # Convert back to numpy to paste checkbox
            form_img = np.array(pil_img)
            
            # Calculate position and size
            cb_h, cb_w = checkbox_img.shape[:2]
            y_end = min(y_position + cb_h, height)
            x_end = min(x_position + cb_w, width)
            
            # Paste checkbox (handling boundaries)
            form_img[y_position:y_end, x_position:x_end] = checkbox_img[:y_end-y_position, :x_end-x_position]
            
            # Add to PIL image
            pil_img = Image.fromarray(form_img)
            draw = ImageDraw.Draw(pil_img)
            
            # Add label text
            label = f"Option {chr(65 + i)}"
            draw.text((x_position + cb_w + 10, y_position + cb_h // 4), 
                     label, fill=(0, 0, 0), font=font)
            
            # Save annotation (YOLO format: class x_center y_center width height)
            # Normalize coordinates
            x_center = (x_position + cb_w // 2) / width
            y_center = (y_position + cb_h // 2) / height
            norm_width = cb_w / width
            norm_height = cb_h / height
            
            annotations.append({
                'class': 0,  # checkbox class
                'x_center': x_center,
                'y_center': y_center, 
                'width': norm_width,
                'height': norm_height,
                'state': state
            })
            
            # Move to next position
            y_position += cb_h + 30
        
        # Save form image
        form_path = self.images_dir / f"form_{form_id:04d}.png"
        final_img = np.array(pil_img)
        cv2.imwrite(str(form_path), final_img)
        
        return {
            'image_path': form_path,
            'annotations': annotations,
            'form_id': form_id
        }
    
    def generate_form_with_checkboxes(self, num_checkboxes: int = 5, 
                                    image_size: Tuple[int, int] = (800, 600),
                                    output_path: Optional[Path] = None) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate a single form with checkboxes for unified pipeline.
        
        Args:
            num_checkboxes: Number of checkboxes to generate
            image_size: (width, height) of the output image
            output_path: Where to save the image (optional)
            
        Returns:
            Tuple of (image_array, annotations_list)
        """
        width, height = image_size
        
        # Create background
        bg_pattern = random.choice(self.bg_patterns)
        form_img = self.create_form_background(width, height, bg_pattern)
        
        # Convert to PIL for text rendering
        pil_img = Image.fromarray(form_img)
        draw = ImageDraw.Draw(pil_img)
        
        # Add form elements
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 20)
        except:
            font = ImageFont.load_default()
        
        # Generate checkboxes
        annotations = []
        y_start = 80
        y_spacing = max(50, height // (num_checkboxes + 2))
        
        for i in range(num_checkboxes):
            # Random checkbox style
            style = CheckboxStyle(
                size=random.choice(self.sizes),
                border_width=random.choice(self.border_widths),
                border_color=random.choice(self.border_colors),
                check_style=random.choice(self.check_styles),
                rounded_corners=random.choice([True, False])
            )
            
            # Random state
            state = random.choice(["checked", "unchecked", "unclear"])
            
            # Generate checkbox
            checkbox_img = self.generate_checkbox(style, state)
            
            # Position
            x_pos = 50 + random.randint(-10, 10)  # Add some variation
            y_pos = y_start + i * y_spacing + random.randint(-5, 5)
            
            # Convert back to numpy to paste checkbox
            form_img = np.array(pil_img)
            
            # Calculate dimensions
            cb_h, cb_w = checkbox_img.shape[:2]
            
            # Ensure within bounds
            if y_pos + cb_h < height and x_pos + cb_w < width:
                # Paste checkbox
                form_img[y_pos:y_pos + cb_h, x_pos:x_pos + cb_w] = checkbox_img
                
                # Back to PIL
                pil_img = Image.fromarray(form_img)
                draw = ImageDraw.Draw(pil_img)
                
                # Add label
                label = f"Option {chr(65 + i)}: {random.choice(['Accept', 'Decline', 'Agree', 'Disagree', 'Yes', 'No'])}"
                draw.text((x_pos + cb_w + 15, y_pos + cb_h // 3), 
                         label, fill=(0, 0, 0), font=font)
                
                # Create annotation
                annotations.append({
                    'x_center': x_pos + cb_w // 2,
                    'y_center': y_pos + cb_h // 2,
                    'width': cb_w,
                    'height': cb_h,
                    'state': state
                })
        
        # Final image
        final_img = np.array(pil_img)
        
        # Save if path provided
        if output_path:
            cv2.imwrite(str(output_path), final_img)
        
        return final_img, annotations
    
    def generate_dataset(self, num_forms: int = 100):
        """Generate complete synthetic dataset."""
        print(f"Generating {num_forms} synthetic forms...")
        
        metadata = {
            "num_forms": num_forms,
            "classes": {"0": "unchecked", "1": "checked", "2": "unclear"},
            "forms": []
        }
        
        for i in tqdm(range(num_forms)):
            num_checkboxes = random.randint(3, 10)
            form_data = self.generate_form_with_checkboxes(i, num_checkboxes)
            metadata["forms"].append(form_data)
        
        # Save metadata
        metadata_path = self.output_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Generated {num_forms} forms with synthetic checkboxes")
        print(f"Images saved to: {self.images_dir}")
        print(f"Labels saved to: {self.labels_dir}")
        print(f"Metadata saved to: {metadata_path}")
        
        # Generate summary statistics
        total_checkboxes = sum(len(form["annotations"]) for form in metadata["forms"])
        state_counts = {"unchecked": 0, "checked": 0, "unclear": 0}
        
        for form in metadata["forms"]:
            for ann in form["annotations"]:
                state_counts[ann["state"]] += 1
        
        print(f"\nDataset Statistics:")
        print(f"Total checkboxes: {total_checkboxes}")
        print(f"Average checkboxes per form: {total_checkboxes / num_forms:.1f}")
        print(f"State distribution:")
        for state, count in state_counts.items():
            print(f"  - {state}: {count} ({count/total_checkboxes*100:.1f}%)")


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate synthetic checkbox data")
    parser.add_argument("--num-forms", type=int, default=100,
                       help="Number of forms to generate")
    parser.add_argument("--output-dir", type=str, default="data/synthetic",
                       help="Output directory for synthetic data")
    
    args = parser.parse_args()
    
    generator = SyntheticCheckboxGenerator(args.output_dir)
    generator.generate_dataset(args.num_forms)


if __name__ == "__main__":
    main()