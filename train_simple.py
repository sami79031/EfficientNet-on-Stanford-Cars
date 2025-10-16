#!/usr/bin/env python3
"""
Simple training script that uses the original training code with minimal changes.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append('.')

import torch
from training.train import perform_training
from training.trial_info import TrialInfo
from models.efficient_net.efficient_nets import EfficientNets
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def main():
    """Simple training function."""
    log.info("Starting simple training...")
    
    # Check GPU
    if torch.cuda.is_available():
        log.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        log.info("Using CPU")
    
    # Create output directories
    Path('data/output/logs').mkdir(parents=True, exist_ok=True)
    Path('data/output/trials').mkdir(parents=True, exist_ok=True)
    Path('lightning_logs').mkdir(parents=True, exist_ok=True)
    
    # Create trial info
    trial_info = TrialInfo(
        model_info=EfficientNets.b0.value,
        load_weights=True,
        advprop=False,
        freeze_pretrained_weights=False,
        epochs=150,
        batch_size=16,  # Smaller batch size for CPU
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
