import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
from pydantic import BaseModel, PositiveFloat, PositiveInt
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR, LambdaLR
from torchmetrics.functional import accuracy

from models.cnn import make_cnn
from datamodules.cifar10 import Cifar10DataModule
from const import PATH_DATASETS, BATCH_SIZE, NUM_WORKERS, INITIAL_LR, LR_UPDATE_STEPS

seed_everything(0)


def inverse_sqrt_schedule(step, update_step):
    prev_step = max(step - 1, 0)
    multiplier = np.sqrt(1.0 + np.floor(prev_step / update_step)) / np.sqrt(
        1.0 + np.floor(step / update_step)
    )
    return multiplier


class TrainingConfig(BaseModel):
    c: PositiveInt = 64
    num_classes: PositiveInt = 10
    lr: PositiveFloat = INITIAL_LR
    lr_update_steps: PositiveInt = LR_UPDATE_STEPS
    epochs: PositiveInt = 1
    batch_size: PositiveInt = BATCH_SIZE
    num_workers: PositiveInt = NUM_WORKERS
    label_noise: PositiveFloat = 0.0  # TODO
    augment: bool = True  # TODO


class LitCNN(pl.LightningModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()

        self.config = config
        self.save_hyperparameters()
        self.model = make_cnn(self.config.c, self.config.num_classes)

    def forward(self, x):
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, "multiclass", num_classes=self.config.num_classes)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        self.evaluate(batch, "val")

    def test_step(self, batch, batch_idx):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.config.lr,
        )
        scheduler_dict = {
            "scheduler": LambdaLR(
                optimizer,
                lambda step: inverse_sqrt_schedule(step, self.config.lr_update_steps),
            ),
            "interval": "step",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
