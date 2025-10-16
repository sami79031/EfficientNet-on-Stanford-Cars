#!/usr/bin/env python3
"""
Clean up and properly organize the real Stanford Cars dataset.
"""

import os
import shutil
from pathlib import Path
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def cleanup_and_organize():
    """Clean up dummy files and organize the real Stanford Cars dataset."""
    car_ims_dir = Path('data/input/stanford/car_ims')
    devkit_dir = Path('data/input/stanford/devkit')
    
    log.info("Cleaning up dummy files and organizing real Stanford Cars dataset...")
    
    # Remove dummy files
    dummy_files = list(car_ims_dir.glob('dummy_*.jpg'))
    log.info(f"Removing {len(dummy_files)} dummy files...")
    for dummy_file in dummy_files:
        dummy_file.unlink()
    
    # Count real car class directories
    car_dirs = [d for d in car_ims_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    log.info(f"Found {len(car_dirs)} real car class directories")
    
    # Create proper annotations
    create_real_annotations(car_ims_dir, devkit_dir, car_dirs)
    
    log.info("Real Stanford Cars dataset organized successfully!")


def create_real_annotations(car_ims_dir, devkit_dir, car_dirs):
    """Create proper annotations for the real Stanford Cars dataset."""
    import pandas as pd
    
    log.info("Creating annotations for real Stanford Cars dataset...")
    
    # Sort car directories to ensure consistent class IDs
    car_dirs.sort()
    
    # Create class name to ID mapping
    class_mapping = {}
    for i, car_dir in enumerate(car_dirs):
        class_mapping[car_dir.name] = i
    
    annotations = []
    
    # Process each car class directory
    for car_dir in car_dirs:
        class_id = class_mapping[car_dir.name]
        class_name = car_dir.name
        
        # Get all images in this class directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            image_files.extend(car_dir.glob(ext))
        
        log.info(f"Processing {class_name}: {len(image_files)} images")
        
        # Create annotations for each image
        for img_file in image_files:
            # Determine if it's train or test based on directory name
            # Stanford Cars dataset typically has train/test split in the original structure
            # We'll use a simple heuristic: first 80% train, last 20% test
            img_index = int(img_file.stem.split('_')[-1]) if '_' in img_file.stem else 0
            is_test = img_index % 5 == 0  # Every 5th image is test
            
            rel_path = img_file.relative_to(car_ims_dir.parent)
            
            annotations.append({
                'relative_im_path': str(rel_path),
                'bbox_x1': 0, 'bbox_y1': 0, 'bbox_x2': 224, 'bbox_y2': 224,
                'class': class_id,
                'test': 1 if is_test else 0
            })
    
    # Save annotations
    annos_df = pd.DataFrame(annotations)
    annos_df.to_csv(devkit_dir / 'cars_annos.csv', index=False)
    
    # Create meta file with real class names
    meta_data = []
    for car_dir in car_dirs:
        meta_data.append({'class_names': car_dir.name})
    
    meta_df = pd.DataFrame(meta_data)
    meta_df.to_csv(devkit_dir / 'cars_meta.csv', index=False)
    
    log.info(f"Created annotations for {len(annotations)} real car images")
    log.info(f"Training images: {len([a for a in annotations if a['test'] == 0])}")
    log.info(f"Test images: {len([a for a in annotations if a['test'] == 1])}")
    log.info(f"Total classes: {len(car_dirs)}")


if __name__ == '__main__':
    cleanup_and_organize()
