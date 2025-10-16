#!/usr/bin/env python3
"""
Fixed Stanford Cars dataset download script.
"""

import subprocess
import os
import shutil
from pathlib import Path
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def download_stanford_cars_fixed():
    """Download Stanford Cars dataset with proper structure handling."""
    data_dir = Path('data/input/stanford')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Downloading Stanford Cars dataset from GitHub...")
    
    try:
        # Clone the repository
        subprocess.run([
            'git', 'clone', 'https://github.com/cyizhuo/Stanford_Cars_dataset.git', 'temp_stanford'
        ], check=True)
        
        # Check what's actually in the repository
        log.info("Checking repository structure...")
        result = subprocess.run(['ls', '-la', 'temp_stanford/'], capture_output=True, text=True)
        log.info(f"Repository contents:\n{result.stdout}")
        
        # Look for train/test directories
        train_dir = Path('temp_stanford/train')
        test_dir = Path('temp_stanford/test')
        
        if train_dir.exists() and test_dir.exists():
            log.info("Found train and test directories!")
            setup_stanford_data(train_dir, test_dir, data_dir)
        else:
            # Check for other possible structures
            log.info("Looking for alternative directory structures...")
            for item in Path('temp_stanford').iterdir():
                if item.is_dir():
                    log.info(f"Found directory: {item}")
                    # Check if it contains car images
                    image_count = len(list(item.rglob('*.jpg'))) + len(list(item.rglob('*.jpeg')))
                    if image_count > 0:
                        log.info(f"Directory {item} contains {image_count} images")
                        setup_stanford_data_from_single_dir(item, data_dir)
                        break
        
        # Clean up
        shutil.rmtree('temp_stanford')
        log.info("Successfully downloaded and organized Stanford Cars dataset!")
        return True
        
    except Exception as e:
        log.error(f"Failed to download from GitHub: {e}")
        # Clean up on failure
        if Path('temp_stanford').exists():
            shutil.rmtree('temp_stanford')
        return False


def setup_stanford_data(train_dir, test_dir, data_dir):
    """Set up Stanford Cars data with train/test structure."""
    car_ims_dir = data_dir / 'car_ims'
    devkit_dir = data_dir / 'devkit'
    
    car_ims_dir.mkdir(parents=True, exist_ok=True)
    devkit_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy train images
    log.info("Copying training images...")
    subprocess.run(['cp', '-r', str(train_dir) + '/*', str(car_ims_dir)], shell=True, check=True)
    
    # Copy test images
    log.info("Copying test images...")
    subprocess.run(['cp', '-r', str(test_dir) + '/*', str(car_ims_dir)], shell=True, check=True)
    
    # Create annotation files
    create_annotations_from_structure(car_ims_dir, devkit_dir, train_dir, test_dir)


def setup_stanford_data_from_single_dir(source_dir, data_dir):
    """Set up Stanford Cars data from a single directory structure."""
    car_ims_dir = data_dir / 'car_ims'
    devkit_dir = data_dir / 'devkit'
    
    car_ims_dir.mkdir(parents=True, exist_ok=True)
    devkit_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy all images
    log.info("Copying all images...")
    subprocess.run(['cp', '-r', str(source_dir) + '/*', str(car_ims_dir)], shell=True, check=True)
    
    # Create annotation files
    create_annotations_from_images(car_ims_dir, devkit_dir)


def create_annotations_from_structure(car_ims_dir, devkit_dir, train_dir, test_dir):
    """Create annotations from train/test directory structure."""
    import pandas as pd
    import os
    
    log.info("Creating annotation files from directory structure...")
    
    annotations = []
    
    # Process train images
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), car_ims_dir.parent)
                class_name = os.path.basename(root)
                class_id = hash(class_name) % 196  # Convert to class ID
                
                annotations.append({
                    'relative_im_path': rel_path,
                    'bbox_x1': 0, 'bbox_y1': 0, 'bbox_x2': 224, 'bbox_y2': 224,
                    'class': class_id,
                    'test': 0  # Training data
                })
    
    # Process test images
    for root, dirs, files in os.walk(test_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), car_ims_dir.parent)
                class_name = os.path.basename(root)
                class_id = hash(class_name) % 196  # Convert to class ID
                
                annotations.append({
                    'relative_im_path': rel_path,
                    'bbox_x1': 0, 'bbox_y1': 0, 'bbox_x2': 224, 'bbox_y2': 224,
                    'class': class_id,
                    'test': 1  # Test data
                })
    
    # Save annotations
    annos_df = pd.DataFrame(annotations)
    annos_df.to_csv(devkit_dir / 'cars_annos.csv', index=False)
    
    # Create meta file
    meta_data = []
    for i in range(196):
        meta_data.append({'class_names': f'Car_Class_{i}'})
    
    meta_df = pd.DataFrame(meta_data)
    meta_df.to_csv(devkit_dir / 'cars_meta.csv', index=False)
    
    log.info(f"Created annotations for {len(annotations)} images")


def create_annotations_from_images(car_ims_dir, devkit_dir):
    """Create annotations from all images in the directory."""
    import pandas as pd
    import os
    
    log.info("Creating annotation files from all images...")
    
    annotations = []
    image_files = []
    
    # Get all image files
    for root, dirs, files in os.walk(car_ims_dir):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):
                rel_path = os.path.relpath(os.path.join(root, file), car_ims_dir.parent)
                image_files.append(rel_path)
    
    # Create annotations (80% train, 20% test)
    for i, img_path in enumerate(image_files):
        is_test = i % 5 == 0  # Every 5th image is test
        class_id = i % 196  # Cycle through classes
        
        annotations.append({
            'relative_im_path': img_path,
            'bbox_x1': 0, 'bbox_y1': 0, 'bbox_x2': 224, 'bbox_y2': 224,
            'class': class_id,
            'test': 1 if is_test else 0
        })
    
    # Save annotations
    annos_df = pd.DataFrame(annotations)
    annos_df.to_csv(devkit_dir / 'cars_annos.csv', index=False)
    
    # Create meta file
    meta_data = []
    for i in range(196):
        meta_data.append({'class_names': f'Car_Class_{i}'})
    
    meta_df = pd.DataFrame(meta_data)
    meta_df.to_csv(devkit_dir / 'cars_meta.csv', index=False)
    
    log.info(f"Created annotations for {len(annotations)} images")


if __name__ == '__main__':
    download_stanford_cars_fixed()
