#!/usr/bin/env python3
"""
Test training script with small VMMRdb dataset.
"""

import torch
import pytorch_lightning as pl
from pathlib import Path
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from training.train_gcp import perform_training_gcp
from training.trial_info import TrialInfo
from models.efficient_net.efficient_nets import EfficientNets
from datasets.stanford.stanford_cars_data_module_gcp import StanfordCarsDataModuleGCP
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)

def main():
    log.info("Starting VMMRdb SMALL test training...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        log.info("CUDA available, using GPU")
        accelerator = 'gpu'
        devices = 1
    else:
        log.info("CUDA not available, using CPU")
        accelerator = 'cpu'
        devices = 1
    
    # Use smaller batch size for testing
    batch_size = 4
    
    # Count classes from small dataset
    try:
        import pandas as pd
        annos_df = pd.read_csv('data/input/stanford/devkit/cars_annos_small.csv')
        num_classes = annos_df['class'].nunique()
        log.info(f"Detected {num_classes} classes in small dataset")
    except Exception as e:
        log.warning(f"Could not detect number of classes: {e}")
        num_classes = 100  # Default for small test
    
    # Create trial info for small test
    trial_info = TrialInfo(
        model_info=EfficientNets.b0.value,
        load_weights=True,
        advprop=False,
        freeze_pretrained_weights=False,
        epochs=2,  # Just 2 epochs for testing
        batch_size=batch_size,
        initial_lr=0.001,
        optimizer=torch.optim.AdamW,
        optimizer_settings={'weight_decay': 1e-4},
        scheduler_settings={'patience': 5, 'factor': 0.5, 'min_lr': 1e-6},
        custom_dropout_rate=None,
        num_classes=num_classes,
        in_channels=3
    )
    
    log.info(f"Starting small test training with trial: {trial_info.trial_id}")
    log.info(f"Batch size: {trial_info.batch_size}")
    log.info(f"Epochs: {trial_info.epochs}")
    log.info(f"Number of classes: {trial_info.num_classes}")
    
    # Create data module
    training_data = StanfordCarsDataModuleGCP(
        image_size=224,
        batch_size=trial_info.batch_size,
        in_channels=3,
        root_path=Path('.')
    )
    
    # Perform training
    perform_training_gcp(trial_info, training_data=training_data)
    
    log.info("Small test training completed!")

if __name__ == '__main__':
    main()
