#!/usr/bin/env python3
"""
Training script optimized for VMMRdb dataset.
VMMRdb has many more classes than Stanford Cars (potentially 1000+ vs 196).
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
    log.info("Starting VMMRdb training setup...")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        log.info("CUDA available, using GPU")
        accelerator = 'gpu'
        devices = 1
    else:
        log.info("CUDA not available, using CPU")
        accelerator = 'cpu'
        devices = 1
    
    # Determine optimal batch size based on available memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        if gpu_memory >= 16:  # 16GB+ GPU
            batch_size = 32
        elif gpu_memory >= 8:  # 8GB+ GPU
            batch_size = 16
        else:  # < 8GB GPU
            batch_size = 8
    else:
        batch_size = 8  # CPU training
    
    log.info(f"Optimal batch size: {batch_size}")
    
    # Count actual number of classes from the dataset
    try:
        import pandas as pd
        annos_df = pd.read_csv('data/input/stanford/devkit/cars_annos.csv')
        num_classes = annos_df['class'].nunique()
        log.info(f"Detected {num_classes} classes in dataset")
    except Exception as e:
        log.warning(f"Could not detect number of classes: {e}")
        num_classes = 1000  # Default for VMMRdb
    
    # Create trial info optimized for VMMRdb
    trial_info = TrialInfo(
        model_info=EfficientNets.b0.value,
        load_weights=True,
        advprop=False,
        freeze_pretrained_weights=False,
        epochs=50,  # Reduced epochs for initial testing
        batch_size=batch_size,
        initial_lr=0.001,
        optimizer=torch.optim.AdamW,
        optimizer_settings={'weight_decay': 1e-4},
        scheduler_settings={'step_size': 10, 'gamma': 0.1},
        custom_dropout_rate=None,
        num_classes=num_classes,
        in_channels=3
    )
    
    log.info(f"Starting VMMRdb training with trial: {trial_info.trial_id}")
    log.info(f"Batch size: {trial_info.batch_size}")
    log.info(f"Epochs: {trial_info.epochs}")
    log.info(f"Number of classes: {trial_info.num_classes}")
    log.info(f"GPU available: {torch.cuda.is_available()}")
    log.info(f"Data workers: 4")
    
    # Create data module
    training_data = StanfordCarsDataModuleGCP(
        data_directory=Path('data/input/stanford'),
        image_size=224,
        batch_size=trial_info.batch_size,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Perform training
    perform_training_gcp(trial_info, training_data=training_data)
    
    log.info("VMMRdb training completed!")

if __name__ == '__main__':
    main()
