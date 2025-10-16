#!/usr/bin/env python3
"""
Simple GCP training script that skips data download for testing.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

import torch
from training.train_gcp import perform_training_gcp
from training.trial_info import TrialInfo
from models.efficient_net.efficient_nets import EfficientNets
from datasets.stanford.stanford_cars_data_module_gcp import StanfordCarsDataModuleGCP
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def setup_gcp_environment():
    """Set up environment for GCP training with optimizations."""
    # Check GPU availability more thoroughly
    if torch.cuda.is_available():
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Set CUDA optimizations
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.cuda.empty_cache()
    else:
        log.info("CUDA not available, using CPU")
    
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
    """Simple GCP training function."""
    log.info("Starting simple GCP training...")
    
    # Setup environment
    setup_gcp_environment()
    
    # Check if data exists
    data_dir = Path('data/input/stanford')
    if not (data_dir / 'devkit' / 'cars_annos.csv').exists():
        log.error("Dataset not found! Please run download_dataset_manual.py first")
        log.info("Run: python download_dataset_manual.py")
        return
    
    # Determine optimal batch size
    optimal_batch_size = get_optimal_batch_size()
    log.info(f"Optimal batch size: {optimal_batch_size}")
    
    # Create trial info with GCP-optimized settings
    trial_info = TrialInfo(
        model_info=EfficientNets.b0.value,
        load_weights=True,
        advprop=False,
        freeze_pretrained_weights=False,
        epochs=5,  # Reduced for testing
        batch_size=optimal_batch_size,
        initial_lr=1e-3,
        optimizer=torch.optim.AdamW,
        optimizer_settings=dict(weight_decay=0.01),
        scheduler_settings=dict(patience=5, factor=0.5),
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
    
    log.info(f"Starting simple GCP training with trial: {trial_info}")
    log.info(f"Batch size: {trial_info.batch_size}")
    log.info(f"Epochs: {trial_info.epochs}")
    log.info(f"GPU available: {torch.cuda.is_available()}")
    log.info(f"Data workers: {training_data.num_workers}")
    
    # Start training with GCP optimizations
    perform_training_gcp(trial_info, training_data=training_data)
    
    log.info("Simple GCP training completed!")


if __name__ == '__main__':
    main()
