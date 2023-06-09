import os
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from pydantic import BaseModel, PositiveFloat, PositiveInt
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy

from models.cnn import make_cnn
from datamodules.cifar10 import Cifar10DataModule
from const import PATH_DATASETS, BATCH_SIZE, NUM_WORKERS

seed_everything(0)


class TrainingConfig(BaseModel):
    c: PositiveInt = 64
    num_classes: PositiveInt = 10
    lr: PositiveFloat = 0.05
    epochs: PositiveInt = 1
    batch_size: PositiveInt = BATCH_SIZE
    num_workers: PositiveInt = NUM_WORKERS
    label_noise: PositiveFloat = 0.0  # TODO
    augment: bool = False  # TODO


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
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
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
            momentum=0.9,
            weight_decay=5e-4,
        )
        return optimizer
