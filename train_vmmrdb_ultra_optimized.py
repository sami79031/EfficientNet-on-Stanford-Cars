#!/usr/bin/env python3
"""
Ultra-optimized training script for VMMRdb dataset - 50 epochs in 2 weeks.
"""

import torch
import pytorch_lightning as pl
from pathlib import Path
import sys
import logging
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from training.train_ultra_optimized import perform_training_ultra_optimized
from training.trial_info import TrialInfo
from models.efficient_net.efficient_nets import EfficientNets
from datasets.stanford.stanford_cars_data_module_ultra import StanfordCarsDataModuleUltra
from utils.default_logging import configure_default_logging

# Set up logging
log = configure_default_logging(__name__)

def calculate_ultra_optimal_settings():
    """Calculate ultra-optimal settings for 2-week completion."""
    
    # Check CUDA availability
    if torch.cuda.is_available():
        log.info("CUDA available, using GPU")
        accelerator = 'gpu'
        devices = 1
        batch_size = 128  # Much larger batch size
        num_workers = 16  # More workers
        precision = '16-mixed'
    else:
        log.info("CUDA not available, using CPU")
        accelerator = 'cpu'
        devices = 1
        batch_size = 16  # Conservative batch size for macOS CPU
        num_workers = 4  # Conservative workers for macOS
        precision = 'bf16-mixed'
    
    return accelerator, devices, batch_size, num_workers, precision

def main():
    log.info("Starting ULTRA-OPTIMIZED VMMRdb training for 2-week completion...")
    
    # Get ultra-optimal settings
    accelerator, devices, batch_size, num_workers, precision = calculate_ultra_optimal_settings()
    
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
        log.info(f"Ultra batch size: {batch_size}")
        log.info(f"Ultra workers: {num_workers}")
        log.info(f"Precision: {precision}")
        
        # Estimate training time with optimizations
        # Much faster per batch with optimizations
        time_per_batch = 0.02 if accelerator == 'gpu' else 0.1  # 5x faster
        time_per_epoch = batches_per_epoch * time_per_batch / 3600  # hours
        total_time_hours = time_per_epoch * 50
        total_time_days = total_time_hours / 24
        
        log.info(f"Ultra-optimized time per epoch: {time_per_epoch:.1f} hours")
        log.info(f"Ultra-optimized total time: {total_time_hours:.1f} hours ({total_time_days:.1f} days)")
        
    except Exception as e:
        log.error(f"Could not analyze dataset: {e}")
        num_classes = 9169
        batch_size = 64
        num_workers = 8
        precision = '16-mixed'
    
    # Create ultra-optimized trial info
    trial_info = TrialInfo(
        model_info=EfficientNets.b0.value,
        load_weights=True,
        advprop=False,
        freeze_pretrained_weights=False,
        epochs=50,  # Keep 50 epochs
        batch_size=batch_size,
        initial_lr=0.002,  # Higher learning rate for faster convergence
        optimizer=torch.optim.AdamW,
        optimizer_settings={'weight_decay': 1e-4},
        scheduler_settings={'patience': 10, 'factor': 0.7, 'min_lr': 1e-6},  # Faster learning rate decay
        custom_dropout_rate=None,
        num_classes=num_classes,
        in_channels=3
    )
    
    log.info(f"Starting ultra-optimized training with trial: {trial_info.trial_id}")
    log.info(f"Epochs: {trial_info.epochs}")
    log.info(f"Number of classes: {trial_info.num_classes}")
    log.info(f"Learning rate: {trial_info.initial_lr}")
    
    # Create ultra-optimized data module
    log.info("Creating ultra-optimized data module...")
    training_data = StanfordCarsDataModuleUltra(
        image_size=224,
        batch_size=trial_info.batch_size,
        in_channels=3,
        root_path=Path('.')
    )
    
    # Override with ultra-optimized settings
    training_data.num_workers = num_workers
    
    log.info("Starting ultra-optimized training...")
    
    # Perform training
    try:
        perform_training_ultra_optimized(trial_info, training_data=training_data)
        log.info("Ultra-optimized training completed successfully!")
    except Exception as e:
        log.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
