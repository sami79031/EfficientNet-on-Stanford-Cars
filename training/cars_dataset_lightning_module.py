from abc import ABC

import pytorch_lightning as pl
from torch.optim.lr_scheduler import ReduceLROnPlateau

from training.label_smoothing_cross_entropy import LabelSmoothingCrossEntropy
from utils.default_logging import configure_default_logging

log = configure_default_logging(__name__)


class StanfordCarsDatasetLightningModule(pl.LightningModule, ABC):

    def __init__(self, trial_info):
        super().__init__()
        self.trial_info = trial_info
        self.loss = trial_info.loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)

        # Calculate accuracy
        acc = (pred.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.loss(pred, y)
        
        # Calculate accuracy
        acc = (pred.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', acc, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        optimizer = self.trial_info.optimizer(self.parameters(), lr=self.trial_info.initial_lr,
                                              **self.trial_info.optimizer_settings)
        
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, **self.trial_info.scheduler_settings),
            'reduce_on_plateau': True,
            'monitor': 'val_loss',
            'name': 'lr'
        }
        return [optimizer], [scheduler]
