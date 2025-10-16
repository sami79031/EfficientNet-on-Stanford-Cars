#!/usr/bin/env python3
"""
Manual data download script for Stanford Cars dataset.
"""

import subprocess
from pathlib import Path
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def download_data():
    """Download Stanford Cars dataset manually."""
    data_dir = Path('data/input/stanford')
    data_dir.mkdir(parents=True, exist_ok=True)
    
    log.info("Downloading Stanford Cars dataset...")
    
    # Try different URLs
    urls = [
        'https://image-net.org/data/cars/car_ims.tgz',
        'https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset/download?datasetVersionNumber=1',
    ]
    
    for url in urls:
        try:
            log.info(f"Trying to download from: {url}")
            if 'kaggle' in url:
                log.info("Kaggle download requires authentication. Please download manually.")
                continue
            
            subprocess.run(['wget', '-O', 'car_ims.tgz', url], check=True)
            subprocess.run(['tar', '-xzf', 'car_ims.tgz', '-C', str(data_dir)], check=True)
            subprocess.run(['rm', 'car_ims.tgz'], check=True)
            log.info("Successfully downloaded car images!")
            break
        except subprocess.CalledProcessError as e:
            log.warning(f"Failed to download from {url}: {e}")
            continue
    
    # Download devkit
    try:
        log.info("Downloading devkit...")
        subprocess.run([
            'wget', '-O', 'car_devkit.tgz',
            'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
        ], check=True)
        subprocess.run(['tar', '-xzf', 'car_devkit.tgz', '-C', str(data_dir)], check=True)
        subprocess.run(['rm', 'car_devkit.tgz'], check=True)
        log.info("Successfully downloaded devkit!")
    except subprocess.CalledProcessError as e:
        log.warning(f"Failed to download devkit: {e}")


if __name__ == '__main__':
    download_data()
