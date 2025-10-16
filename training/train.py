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


def perform_training(
        trial_info: TrialInfo,
        training_data=None,
        model=None,
        logger_tags: Optional[List[str]] = None,

):
    if model is None:
        model = EfficientNet(trial_info=trial_info)
    if training_data is None:
        training_data = StanfordCarsDataModule(batch_size=trial_info.batch_size,
                                               in_channels=trial_info.in_channels,
                                               image_size=model.image_size)

    # Use TensorBoard logger instead of Neptune for now
    logger = TensorBoardLogger(
        save_dir="lightning_logs",
        name=f"{str(trial_info)}"
    )

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

    trainer = pl.Trainer(max_epochs=trial_info.epochs,
                         accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                         devices=1 if torch.cuda.is_available() else 1,
                         # fast_dev_run=True,
                         logger=logger,
                         callbacks=[callback, lrl, checkpoint_callback, early_stop_callback],
                         # Memory-safe settings for macOS
                         num_sanity_val_steps=0,
                         limit_train_batches=1.0,
                         limit_val_batches=1.0,
                         enable_progress_bar=True,
                         enable_model_summary=True
                         )
    trainer.fit(model, datamodule=training_data)


if __name__ == '__main__':
    trial_info = TrialInfo(model_info=EfficientNets.b0.value,
                           load_weights=True,
                           advprop=False,
                           freeze_pretrained_weights=False,
                           epochs=150,
                           batch_size=16,  # Reduced batch size to prevent memory issues
                           initial_lr=1e-3,
                           optimizer=torch.optim.AdamW,
                           optimizer_settings=dict(),
                           scheduler_settings=dict(patience=3),
                           custom_dropout_rate=None,
                           num_classes=196,
                           in_channels=3,
                           )
    perform_training(trial_info)
