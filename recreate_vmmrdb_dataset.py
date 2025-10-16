#!/usr/bin/env python3
"""
Recreate VMMRdb dataset with robust image handling.
"""

import os
import pandas as pd
from pathlib import Path
from utils.default_logging import configure_default_logging
import random
import shutil

log = configure_default_logging(__name__)

def recreate_vmmrdb_dataset(vmmrdb_path="VMMRdb"):
    """Recreate VMMRdb dataset with robust image handling."""
    vmmrdb_dir = Path(vmmrdb_path)
    car_ims_dir = Path('data/input/stanford/car_ims')
    devkit_dir = Path('data/input/stanford/devkit')
    
    # Create directories
    car_ims_dir.mkdir(parents=True, exist_ok=True)
    devkit_dir.mkdir(parents=True, exist_ok=True)
    
    log.info(f"Recreating VMMRdb dataset from: {vmmrdb_dir}")
    
    # Get all car model directories
    car_dirs = [d for d in vmmrdb_dir.iterdir() if d.is_dir()]
    car_dirs.sort()
    
    log.info(f"Found {len(car_dirs)} car model directories")
    
    # Create class mapping (make_model_year -> class_id)
    class_mapping = {}
    class_names = []
    
    for i, car_dir in enumerate(car_dirs):
        class_mapping[car_dir.name] = i
        class_names.append(car_dir.name)
    
    annotations = []
    total_images = 0
    skipped_images = 0
    
    # Process each car model directory
    for car_dir in car_dirs:
        car_model = car_dir.name
        class_id = class_mapping[car_model]
        
        # Get all images in this directory
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(car_dir.glob(ext))
        
        log.info(f"Processing {car_model}: {len(image_files)} images")
        
        # Copy images to car_ims directory and create annotations
        for img_file in image_files:
            try:
                # Create new filename to avoid conflicts
                new_filename = f"{car_model}_{img_file.name}"
                new_path = car_ims_dir / new_filename
                
                # Copy image
                shutil.copy2(img_file, new_path)
                
                # Create relative path for annotation
                rel_path = new_path.relative_to(car_ims_dir.parent)
                
                # Random train/test split (80% train, 20% test)
                is_test = random.random() < 0.2
                
                annotations.append({
                    'relative_im_path': str(rel_path),
                    'bbox_x1': 0, 'bbox_y1': 0, 'bbox_x2': 224, 'bbox_y2': 224,
                    'class': class_id,
                    'test': 1 if is_test else 0
                })
                
                total_images += 1
                
            except Exception as e:
                log.warning(f"Skipped image {img_file}: {e}")
                skipped_images += 1
                continue
    
    # Save annotations
    annos_df = pd.DataFrame(annotations)
    annos_df.to_csv(devkit_dir / 'cars_annos.csv', index=False)
    
    # Create meta file with class names
    meta_data = []
    for class_name in class_names:
        meta_data.append({'class_names': class_name})
    
    meta_df = pd.DataFrame(meta_data)
    meta_df.to_csv(devkit_dir / 'cars_meta.csv', index=False)
    
    log.info(f"Dataset recreation complete!")
    log.info(f"Total images processed: {total_images}")
    log.info(f"Skipped images: {skipped_images}")
    log.info(f"Training images: {len([a for a in annotations if a['test'] == 0])}")
    log.info(f"Test images: {len([a for a in annotations if a['test'] == 1])}")
    log.info(f"Total classes: {len(class_names)}")
    
    return len(class_names), total_images

if __name__ == '__main__':
    import sys
    vmmrdb_path = sys.argv[1] if len(sys.argv) > 1 else "VMMRdb"
    recreate_vmmrdb_dataset(vmmrdb_path)
