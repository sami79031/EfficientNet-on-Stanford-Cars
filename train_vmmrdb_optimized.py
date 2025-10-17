#!/usr/bin/env python3
"""
Optimized training script for VMMRdb dataset - 50 epochs in 3-7 days.
"""

import torch
import pytorch_lightning as pl
from pathlib import Path
import sys
import logging
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from training.train_optimized import perform_training_optimized
from training.trial_info import TrialInfo
from models.efficient_net.efficient_nets import EfficientNets
from datasets.stanford.stanford_cars_data_module_gcp import StanfordCarsDataModuleGCP
from utils.default_logging import configure_default_logging

# Set up logging
log = configure_default_logging(__name__)

def calculate_optimal_settings():
    """Calculate optimal training settings for 3-7 day completion."""
    
    # Check CUDA availability
    if torch.cuda.is_available():
        log.info("CUDA available, using GPU")
        accelerator = 'gpu'
        devices = 1
        batch_size = 64  # Large batch size for GPU
        num_workers = 8  # More workers for GPU
        precision = '16-mixed'  # Mixed precision for speed
    else:
        log.info("CUDA not available, using CPU")
        accelerator = 'cpu'
        devices = 1
        batch_size = 16  # Smaller batch size for CPU
        num_workers = 4  # Fewer workers for CPU
        precision = 'bf16-mixed'  # bfloat16 for CPU
    
    return accelerator, devices, batch_size, num_workers, precision

def main():
    log.info("Starting OPTIMIZED VMMRdb training for 50 epochs...")
    
    # Get optimal settings
    accelerator, devices, batch_size, num_workers, precision = calculate_optimal_settings()
    
    # Count classes and calculate training metrics
    try:
        import pandas as pd
        annos_df = pd.read_csv('data/input/stanford/devkit/cars_annos.csv')
        num_classes = annos_df['class'].nunique()
        train_images = len(annos_df[annos_df['test']==0])
        val_images = len(annos_df[annos_df['test']==1])
        
        batches_per_epoch = train_images // batch_size
        val_batches = val_images // batch_size
        
        log.info(f"Dataset: {num_classes} classes, {len(annos_df)} total images")
        log.info(f"Training: {train_images} images ({batches_per_epoch} batches)")
        log.info(f"Validation: {val_images} images ({val_batches} batches)")
        log.info(f"Batch size: {batch_size}")
        log.info(f"Workers: {num_workers}")
        log.info(f"Precision: {precision}")
        
        # Estimate training time
        # Assuming ~0.1 seconds per batch on GPU, ~0.5 seconds on CPU
        time_per_batch = 0.1 if accelerator == 'gpu' else 0.5
        time_per_epoch = batches_per_epoch * time_per_batch / 3600  # hours
        total_time_hours = time_per_epoch * 50
        total_time_days = total_time_hours / 24
        
        log.info(f"Estimated time per epoch: {time_per_epoch:.1f} hours")
        log.info(f"Estimated total time: {total_time_hours:.1f} hours ({total_time_days:.1f} days)")
        
    except Exception as e:
        log.error(f"Could not analyze dataset: {e}")
        num_classes = 9169
        batch_size = 32
        num_workers = 4
        precision = '16-mixed'
    
    # Create optimized trial info
    trial_info = TrialInfo(
        model_info=EfficientNets.b0.value,
        load_weights=True,
        advprop=False,
        freeze_pretrained_weights=False,
        epochs=50,  # Target: 50 epochs
        batch_size=batch_size,
        initial_lr=0.001,
        optimizer=torch.optim.AdamW,
        optimizer_settings={'weight_decay': 1e-4},
        scheduler_settings={'patience': 15, 'factor': 0.5, 'min_lr': 1e-6},  # More patience for long training
        custom_dropout_rate=None,
        num_classes=num_classes,
        in_channels=3
    )
    
    log.info(f"Starting optimized training with trial: {trial_info.trial_id}")
    log.info(f"Epochs: {trial_info.epochs}")
    log.info(f"Number of classes: {trial_info.num_classes}")
    
    # Create optimized data module
    log.info("Creating optimized data module...")
    training_data = StanfordCarsDataModuleGCP(
        image_size=224,
        batch_size=trial_info.batch_size,
        in_channels=3,
        root_path=Path('.')
    )
    
    # Override num_workers for optimization
    training_data.num_workers = num_workers
    
    log.info("Starting optimized training...")
    
    # Perform training
    try:
        perform_training_optimized(trial_info, training_data=training_data)
        log.info("Optimized training completed successfully!")
    except Exception as e:
        log.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
