"""
Optimized training function for long-running VMMRdb training.
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


def perform_training_optimized(
        trial_info: TrialInfo,
        training_data=None,
        model=None,
        logger_tags: Optional[List[str]] = None,
):
    """Optimized training function for 50-epoch VMMRdb training."""
    
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

    # Optimized callbacks for long training
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=20,  # More patience for long training
        verbose=True
    )

    # Checkpoint every 5 epochs to avoid losing progress
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(trial_info.output_folder),
        filename='checkpoint-{epoch:02d}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,  # Keep top 3 checkpoints
        every_n_epochs=5,  # Save every 5 epochs
        save_last=True,  # Always save the last checkpoint
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
        precision = '16-mixed'  # Mixed precision for speed
    else:
        accelerator = 'cpu'
        devices = 1
        precision = 'bf16-mixed'  # bfloat16 for CPU

    # Optimized trainer configuration
    trainer = pl.Trainer(
        max_epochs=trial_info.epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        logger=logger,
        callbacks=[callback, lrl, checkpoint_callback, early_stop_callback],
        
        # Performance optimizations
        num_sanity_val_steps=0,  # Skip sanity check for faster startup
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        
        # Logging optimizations
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=50,  # Log every 50 steps instead of every step
        
        # Memory optimizations
        gradient_clip_val=1.0,  # Gradient clipping for stability
        accumulate_grad_batches=1,  # No gradient accumulation for now
        
        # Checkpointing
        val_check_interval=1.0,  # Validate every epoch
        check_val_every_n_epoch=1,
        
        # Performance settings
        sync_batchnorm=False,  # Don't sync batch norm for single GPU
        benchmark=True,  # Enable cuDNN benchmark for consistent input sizes
        deterministic=False,  # Allow non-deterministic for speed
    )

    log.info(f"Starting optimized training with {accelerator} accelerator")
    log.info(f"Precision: {precision}")
    log.info(f"Max epochs: {trial_info.epochs}")
    log.info(f"Batch size: {trial_info.batch_size}")
    
    # Start training
    trainer.fit(model, datamodule=training_data)
    
    log.info("Training completed successfully!")
    
    return trainer, model
