#!/usr/bin/env python3
"""
Create a small test dataset from the full VMMRdb dataset for testing purposes.
"""

import pandas as pd
import random
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)

def create_small_test_dataset():
    """Create a small test dataset with balanced classes."""
    log.info("Creating small test dataset...")
    
    # Read the full dataset
    df = pd.read_csv('data/input/stanford/devkit/cars_annos.csv')
    log.info(f'Original dataset: {len(df)} images, {df["class"].nunique()} classes')
    
    # Create a small test dataset with balanced classes
    # Take up to 2 images per class
    small_data = []
    for class_id in df['class'].unique():
        class_images = df[df['class'] == class_id]
        # Take up to 2 images per class
        sample_size = min(2, len(class_images))
        small_data.append(class_images.sample(n=sample_size, random_state=42))
    
    small_df = pd.concat(small_data, ignore_index=True)
    small_df.to_csv('data/input/stanford/devkit/cars_annos_small.csv', index=False)
    
    log.info(f'Small test dataset created: {len(small_df)} images, {small_df["class"].nunique()} classes')
    log.info(f'Training images: {len(small_df[small_df["test"]==0])}')
    log.info(f'Test images: {len(small_df[small_df["test"]==1])}')
    
    return len(small_df), small_df["class"].nunique()

if __name__ == '__main__':
    create_small_test_dataset()
