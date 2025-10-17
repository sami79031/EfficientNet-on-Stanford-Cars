"""
Ultra-optimized training function for 2-week completion on full VMMRdb dataset.
"""

import sys
from typing import Optional, List

sys.path.append('.')

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy

from datasets.stanford.stanford_cars_data_module import StanfordCarsDataModule
from models.efficient_net.efficient_net import EfficientNet
from models.efficient_net.efficient_nets import EfficientNets
from training.cars_dataset_callback import StanfordCarsDatasetCallback
from training.trial_info import TrialInfo
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def perform_training_ultra_optimized(
        trial_info: TrialInfo,
        training_data=None,
        model=None,
        logger_tags: Optional[List[str]] = None,
):
    """Ultra-optimized training function for 2-week completion."""
    
    if model is None:
        model = EfficientNet(trial_info=trial_info)
    if training_data is None:
        training_data = StanfordCarsDataModule(batch_size=trial_info.batch_size,
                                               in_channels=trial_info.in_channels,
                                               image_size=model.image_size)

    # Use TensorBoard logger
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"{str(trial_info)}"
    )

    # Ultra-optimized callbacks for fast training
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=8,  # Less patience for faster training
        verbose=True
    )

    # Checkpoint every 2 epochs for faster recovery
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(trial_info.output_folder),
        filename='ultra-checkpoint-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=2,  # Keep only top 2 checkpoints
        every_n_epochs=2,  # Save every 2 epochs
        save_last=True,
        verbose=True
    )

    # Learning rate monitor
    lrl = LearningRateMonitor(logging_interval='epoch')

    # Custom callback
    callback = StanfordCarsDatasetCallback(trial_info)

    # Determine accelerator and precision
    if torch.cuda.is_available():
        accelerator = 'gpu'
        devices = 1
        precision = '16-mixed'
    else:
        accelerator = 'cpu'
        devices = 1
        precision = 'bf16-mixed'

    # Ultra-optimized trainer configuration
    trainer = pl.Trainer(
        max_epochs=trial_info.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=[callback, lrl, checkpoint_callback, early_stop_callback],
        
        # Ultra performance optimizations
        num_sanity_val_steps=0,  # Skip sanity check
        limit_train_batches=1.0,
        limit_val_batches=0.1,  # Only validate on 10% of validation data
        
        # Ultra logging optimizations
        enable_progress_bar=True,
        enable_model_summary=False,  # Skip model summary for speed
        log_every_n_steps=100,  # Log every 100 steps
        
        # Ultra memory optimizations
        gradient_clip_val=0.5,  # Less aggressive gradient clipping
        accumulate_grad_batches=2,  # Gradient accumulation for larger effective batch size
        
        # Ultra checkpointing
        val_check_interval=0.5,  # Validate twice per epoch
        check_val_every_n_epoch=1,
        
        # Ultra performance settings
        sync_batchnorm=False,
        benchmark=True,
        deterministic=False,
        
        # Additional optimizations
        fast_dev_run=False,
        overfit_batches=0,
        # track_grad_norm=0,  # Removed - not supported in this version
        # detect_anomaly=False,  # Removed - not supported in this version
    )

    log.info(f"Starting ultra-optimized training with {accelerator} accelerator")
    log.info(f"Precision: {precision}")
    log.info(f"Max epochs: {trial_info.epochs}")
    log.info(f"Batch size: {trial_info.batch_size}")
    log.info(f"Gradient accumulation: 2 (effective batch size: {trial_info.batch_size * 2})")
    log.info(f"Validation batches: 10% of validation data")
    
    # Start training
    trainer.fit(model, datamodule=training_data)
    
    log.info("Ultra-optimized training completed successfully!")
    
    return trainer, model
