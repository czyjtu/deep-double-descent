import pytorch_lightning as pl
import torch as th
from torch.utils.data import DataLoader
import torchvision.transforms as tvt
import torchvision.datasets as datasets

from const import PATH_DATASETS, BATCH_SIZE, NUM_WORKERS


class Cifar10DataModule(pl.LightningDataModule):
    # TODO: augmentation and label noise (make it optional and deterministic)

    def __init__(self, batch_size: int = 256):
        super().__init__()
        self.batch_size = batch_size

        self.train_transforms = tvt.Compose(
            [
                tvt.ToTensor(),
                self.cifar10_normalization(),
            ]
        )

        self.test_transforms = tvt.Compose(
            [
                tvt.ToTensor(),
                self.cifar10_normalization(),
            ]
        )

    @staticmethod
    def cifar10_normalization():
        normalize = tvt.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        return normalize

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = datasets.CIFAR10(
                PATH_DATASETS,
                train=True,
                download=True,
                transform=self.train_transforms,
            )
            self.test_dataset = datasets.CIFAR10(
                PATH_DATASETS,
                train=False,
                download=True,
                transform=self.test_transforms,
            )
        elif stage == "test":
            self.test_dataset = datasets.CIFAR10(
                PATH_DATASETS,
                train=False,
                download=True,
                transform=self.test_transforms,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=NUM_WORKERS,
            shuffle=False,
            pin_memory=True,
        )
