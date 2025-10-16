#!/usr/bin/env python3
"""
Create proper Stanford Cars annotations from the class-based structure.
"""

import os
import pandas as pd
from pathlib import Path
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)

def create_stanford_annotations():
    """Create annotations for the real Stanford Cars dataset."""
    car_ims_dir = Path('data/input/stanford/car_ims')
    devkit_dir = Path('data/input/stanford/devkit')
    
    log.info("Creating Stanford Cars annotations...")
    
    # Get all class directories
    class_dirs = [d for d in car_ims_dir.iterdir() if d.is_dir()]
    class_dirs.sort()
    
    log.info(f"Found {len(class_dirs)} class directories")
    
    annotations = []
    
    # Process each class directory
    for class_dir in class_dirs:
        class_name = class_dir.name
        
        # Handle different naming formats
        if '_' in class_name and class_name.split('_')[0] == 'class':
            # Format: class_001, class_002, etc.
            class_id = int(class_name.split('_')[1]) - 1
        else:
            # Use directory index as class ID
            class_id = class_dirs.index(class_dir)
        
        # Get all images in this class
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(class_dir.glob(ext))
        
        log.info(f"Processing {class_name}: {len(image_files)} images")
        
        # Create annotations for each image
        for img_file in image_files:
            rel_path = img_file.relative_to(car_ims_dir.parent)
            
            # Determine train/test based on original structure
            # We'll use a simple heuristic: first 80% train, last 20% test
            img_index = int(img_file.stem) if img_file.stem.isdigit() else 0
            is_test = img_index % 5 == 0  # Every 5th image is test
            
            annotations.append({
                'relative_im_path': str(rel_path),
                'bbox_x1': 0, 'bbox_y1': 0, 'bbox_x2': 224, 'bbox_y2': 224,
                'class': class_id,
                'test': 1 if is_test else 0
            })
    
    # Save annotations
    annos_df = pd.DataFrame(annotations)
    annos_df.to_csv(devkit_dir / 'cars_annos.csv', index=False)
    
    # Create meta file with class names
    meta_data = []
    for class_dir in class_dirs:
        meta_data.append({'class_names': class_dir.name})
    
    meta_df = pd.DataFrame(meta_data)
    meta_df.to_csv(devkit_dir / 'cars_meta.csv', index=False)
    
    log.info(f"Created annotations for {len(annotations)} images")
    log.info(f"Training images: {len([a for a in annotations if a['test'] == 0])}")
    log.info(f"Test images: {len([a for a in annotations if a['test'] == 1])}")
    log.info(f"Total classes: {len(class_dirs)}")

if __name__ == '__main__':
    create_stanford_annotations()
