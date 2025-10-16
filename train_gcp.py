#!/usr/bin/env python3
"""
Google Cloud Platform training script for EfficientNet on Stanford Cars dataset.
This script handles data download, preparation, and training on GCP.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append('.')

import torch
from training.train import perform_training
from training.trial_info import TrialInfo
from models.efficient_net.efficient_nets import EfficientNets
from data_preparation import prepare_raw_data
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def download_stanford_cars_data():
    """Download and extract Stanford Cars dataset if not present."""
    data_dir = Path('data/input/stanford')
    car_ims_dir = data_dir / 'car_ims'
    devkit_dir = data_dir / 'devkit'
    
    # Check if data already exists
    if car_ims_dir.exists() and devkit_dir.exists():
        log.info("Stanford Cars dataset already exists, skipping download")
        return
    
    log.info("Downloading Stanford Cars dataset...")
    
    # Create data directory
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download car images - using working URLs
    if not car_ims_dir.exists():
        log.info("Downloading car images...")
        # Try multiple URLs in case some are down
        urls = [
            'https://image-net.org/data/cars/car_ims.tgz',
            'https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset/download?datasetVersionNumber=1',
        ]
        
        success = False
        for url in urls:
            try:
                if 'kaggle' in url:
                    log.info("Note: Kaggle download requires authentication. Please download manually if needed.")
                    continue
                else:
                    subprocess.run(['wget', '-O', 'car_ims.tgz', url], check=True)
                
                subprocess.run(['tar', '-xzf', 'car_ims.tgz', '-C', str(data_dir)], check=True)
                subprocess.run(['rm', 'car_ims.tgz'], check=True)
                success = True
                break
            except subprocess.CalledProcessError as e:
                log.warning(f"Failed to download from {url}: {e}")
                continue
        
        if not success:
            log.error("Failed to download car images from all sources.")
            log.info("Please download manually from: https://www.kaggle.com/datasets/jessicali9530/stanford-cars-dataset")
            log.info("Or try: https://image-net.org/data/cars/car_ims.tgz")
            return
    
    # Download devkit (annotations)
    if not devkit_dir.exists():
        log.info("Downloading devkit...")
        try:
            subprocess.run([
                'wget', '-O', 'car_devkit.tgz',
                'https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz'
            ], check=True)
            subprocess.run(['tar', '-xzf', 'car_devkit.tgz', '-C', str(data_dir)], check=True)
            subprocess.run(['rm', 'car_devkit.tgz'], check=True)
        except subprocess.CalledProcessError:
            log.warning("Failed to download devkit. Creating minimal annotation file...")
            # Create a minimal annotation structure
            devkit_dir.mkdir(parents=True, exist_ok=True)
            # We'll create the annotation files manually if needed


def setup_gcp_environment():
    """Set up environment for GCP training."""
    # Check GPU availability more thoroughly
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        log.info("CUDA not available, using CPU")
        # Try to check if GPU is present but drivers not loaded
        try:
            result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
            if result.returncode == 0:
                log.info("GPU detected but CUDA not available. You may need to restart the VM.")
        except:
            pass
    
    # Create output directories
    Path('data/output/logs').mkdir(parents=True, exist_ok=True)
    Path('data/output/trials').mkdir(parents=True, exist_ok=True)
    Path('lightning_logs').mkdir(parents=True, exist_ok=True)


def main():
    """Main training function for GCP."""
    log.info("Starting GCP training setup...")
    
    # Setup environment
    setup_gcp_environment()
    
    # Download data if needed
    download_stanford_cars_data()
    
    # Prepare data (convert .mat to .csv)
    log.info("Preparing data...")
    prepare_raw_data()
    
    # Create trial info with GCP-optimized settings
    trial_info = TrialInfo(
        model_info=EfficientNets.b0.value,
        load_weights=True,
        advprop=False,
        freeze_pretrained_weights=False,
        epochs=150,
        batch_size=32,  # Larger batch size for GCP GPU
        initial_lr=1e-3,
        optimizer=torch.optim.AdamW,
        optimizer_settings=dict(),
        scheduler_settings=dict(patience=3),
        custom_dropout_rate=None,
        num_classes=196,
        in_channels=3,
    )
    
    log.info(f"Starting training with trial: {trial_info}")
    log.info(f"Batch size: {trial_info.batch_size}")
    log.info(f"Epochs: {trial_info.epochs}")
    log.info(f"GPU available: {torch.cuda.is_available()}")
    
    # Start training
    perform_training(trial_info)
    
    log.info("Training completed!")


if __name__ == '__main__':
    main()