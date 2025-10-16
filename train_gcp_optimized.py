#!/usr/bin/env python3
"""
GCP-Optimized training script for EfficientNet on Stanford Cars dataset.
This script is specifically optimized for Google Cloud Platform with GPU acceleration.
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to path
sys.path.append('.')

import torch
from training.train_gcp import perform_training_gcp
from training.trial_info import TrialInfo
from models.efficient_net.efficient_nets import EfficientNets
from datasets.stanford.stanford_cars_data_module_gcp import StanfordCarsDataModuleGCP
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


def setup_gcp_environment():
    """Set up environment for GCP training with optimizations."""
    # Check GPU availability more thoroughly
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic for speed
        
        # Set memory growth to avoid OOM
        torch.cuda.empty_cache()
        
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


def get_optimal_batch_size():
    """Determine optimal batch size based on available GPU memory."""
    if not torch.cuda.is_available():
        return 16  # CPU fallback
    
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    
    if gpu_memory >= 16:  # V100, A100
        return 128
    elif gpu_memory >= 8:  # T4, P100
        return 64
    elif gpu_memory >= 4:  # K80
        return 32
    else:
        return 16


def main():
    """Main GCP-optimized training function."""
    log.info("Starting GCP-optimized training setup...")
    
    # Setup environment
    setup_gcp_environment()
    
    # Download data if needed
    download_stanford_cars_data()
    
    # Prepare data (convert .mat to .csv)
    log.info("Preparing data...")
    prepare_raw_data()
    
    # Determine optimal batch size
    optimal_batch_size = get_optimal_batch_size()
    log.info(f"Optimal batch size: {optimal_batch_size}")
    
    # Create trial info with GCP-optimized settings
    trial_info = TrialInfo(
        model_info=EfficientNets.b0.value,
        load_weights=True,
        advprop=False,
        freeze_pretrained_weights=False,
        epochs=150,
        batch_size=optimal_batch_size,  # Dynamic batch size based on GPU
        initial_lr=1e-3,
        optimizer=torch.optim.AdamW,
        optimizer_settings=dict(weight_decay=0.01),  # Added weight decay
        scheduler_settings=dict(patience=5, factor=0.5),  # More aggressive scheduling
        custom_dropout_rate=None,
        num_classes=196,
        in_channels=3,
    )
    
    # Create GCP-optimized data module
    training_data = StanfordCarsDataModuleGCP(
        batch_size=trial_info.batch_size,
        in_channels=trial_info.in_channels,
        image_size=224  # EfficientNet-B0 image size
    )
    
    log.info(f"Starting GCP-optimized training with trial: {trial_info}")
    log.info(f"Batch size: {trial_info.batch_size}")
    log.info(f"Epochs: {trial_info.epochs}")
    log.info(f"GPU available: {torch.cuda.is_available()}")
    log.info(f"Data workers: {training_data.num_workers}")
    
    # Start training with GCP optimizations
    perform_training_gcp(trial_info, training_data=training_data)
    
    log.info("GCP-optimized training completed!")


if __name__ == '__main__':
    main()
