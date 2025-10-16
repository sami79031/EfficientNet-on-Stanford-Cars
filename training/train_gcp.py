import sys
from typing import Optional, List

sys.path.append('.')

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from datasets.stanford.stanford_cars_data_module import StanfordCarsDataModule
from models.efficient_net.efficient_net import EfficientNet
from models.efficient_net.efficient_nets import EfficientNets
from training.cars_dataset_callback import StanfordCarsDatasetCallback
from training.trial_info import TrialInfo
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


def perform_training_gcp(
        trial_info: TrialInfo,
        training_data=None,
        model=None,
        logger_tags: Optional[List[str]] = None,
):
    """GCP-optimized training function."""
    if model is None:
        model = EfficientNet(trial_info=trial_info)
    if training_data is None:
        training_data = StanfordCarsDataModule(batch_size=trial_info.batch_size,
                                               in_channels=trial_info.in_channels,
                                               image_size=model.image_size)

    # Use TensorBoard logger for GCP
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"{str(trial_info)}"
    )

    # GCP-optimized callbacks
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor='val_loss',
        min_delta=1e-3,
        patience=10
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(trial_info.output_folder),
        filename='best-checkpoint',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )

    callback = StanfordCarsDatasetCallback(trial_info)
    lrl = LearningRateMonitor()

    # GCP-optimized trainer settings
    trainer = pl.Trainer(
        max_epochs=trial_info.epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1 if torch.cuda.is_available() else 1,
        logger=logger,
        callbacks=[callback, lrl, checkpoint_callback, early_stop_callback],
        # GCP-optimized settings (removed Mac memory constraints)
        num_sanity_val_steps=2,  # Enable sanity checks for GCP
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        enable_progress_bar=True,
        enable_model_summary=True,
        # GCP-specific optimizations
        precision=16,  # Use mixed precision for faster training
        gradient_clip_val=1.0,  # Gradient clipping for stability
        accumulate_grad_batches=1,  # No gradient accumulation needed on GCP
        # Better logging for cloud environments
        log_every_n_steps=50,
        val_check_interval=1.0,
    )
    
    trainer.fit(model, datamodule=training_data)


if __name__ == '__main__':
    # GCP-optimized trial configuration
    trial_info = TrialInfo(
        model_info=EfficientNets.b0.value,
        load_weights=True,
        advprop=False,
        freeze_pretrained_weights=False,
        epochs=150,
        batch_size=64,  # Increased batch size for GCP GPU
        initial_lr=1e-3,
        optimizer=torch.optim.AdamW,
        optimizer_settings=dict(weight_decay=0.01),  # Added weight decay
        scheduler_settings=dict(patience=5, factor=0.5),  # More aggressive scheduling
        custom_dropout_rate=None,
        num_classes=196,
        in_channels=3,
    )
    
    log.info(f"Starting GCP training with batch size: {trial_info.batch_size}")
    log.info(f"GPU available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    perform_training_gcp(trial_info)
