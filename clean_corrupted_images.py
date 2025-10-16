#!/usr/bin/env python3
"""
Clean up corrupted images from the VMMRdb dataset.
"""

import os
from pathlib import Path
from PIL import Image
import pandas as pd
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)

def clean_corrupted_images():
    """Remove corrupted images and update annotations."""
    car_ims_dir = Path('data/input/stanford/car_ims')
    devkit_dir = Path('data/input/stanford/devkit')
    
    log.info("Cleaning corrupted images from VMMRdb dataset...")
    
    # Read current annotations
    annos_df = pd.read_csv(devkit_dir / 'cars_annos.csv')
    log.info(f"Starting with {len(annos_df)} images")
    
    # Check each image and remove corrupted ones
    valid_annotations = []
    corrupted_count = 0
    
    for idx, row in annos_df.iterrows():
        image_path = car_ims_dir / row['relative_im_path']
        
        try:
            # Try to open and verify the image
            with Image.open(image_path) as img:
                img.verify()  # Verify the image is valid
            
            # If we get here, the image is valid
            valid_annotations.append(row)
            
        except Exception as e:
            log.warning(f"Corrupted image: {image_path} - {e}")
            corrupted_count += 1
            
            # Remove the corrupted file
            if image_path.exists():
                image_path.unlink()
                log.info(f"Removed corrupted file: {image_path}")
    
    # Update annotations with only valid images
    valid_df = pd.DataFrame(valid_annotations)
    valid_df.to_csv(devkit_dir / 'cars_annos.csv', index=False)
    
    log.info(f"Cleaning complete!")
    log.info(f"Removed {corrupted_count} corrupted images")
    log.info(f"Valid images remaining: {len(valid_df)}")
    log.info(f"Training images: {len(valid_df[valid_df['test']==0])}")
    log.info(f"Test images: {len(valid_df[valid_df['test']==1])}")
    log.info(f"Total classes: {valid_df['class'].nunique()}")

if __name__ == '__main__':
    clean_corrupted_images()
