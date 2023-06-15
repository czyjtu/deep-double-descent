import pytorch_lightning as pl
import torch as th
from torch.utils.data import DataLoader
import torchvision.transforms as tvt
import torchvision.datasets as datasets
from torch.utils.data import TensorDataset

from const import PATH_DATASETS, BATCH_SIZE, NUM_WORKERS, DEVICE


class Cifar10DataModule(pl.LightningDataModule):

    def __init__(
        self,
        batch_size: int = 256,
        augmentation: bool = False,
        label_noise: float = 0.0,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.label_noise = label_noise

        basic_transforms = [
            tvt.ToTensor(),
            self.cifar10_normalization(),
        ]
        train_transforms = basic_transforms.copy()
        if augmentation:
            train_transforms.extend(
                [
                    tvt.RandomCrop(32, padding=4),
                    tvt.RandomHorizontalFlip(),
                ]
            )

        self.train_transforms = tvt.Compose(train_transforms)
        self.test_transforms = tvt.Compose(basic_transforms)

    @staticmethod
    def cifar10_normalization():
        normalize = tvt.Normalize(
            mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
            std=[x / 255.0 for x in [63.0, 62.1, 66.7]],
        )
        return normalize

    def get_dataset_with_label_noise(self, label_noise):
        train_dataset = datasets.CIFAR10(
            PATH_DATASETS,
            train=True,
            download=True,
            transform=self.train_transforms,
        )
        if label_noise > 0.0:
            num_samples = len(train_dataset)
            num_noise_samples = int(num_samples * label_noise)
            noise_indices = th.randperm(num_samples)[:num_noise_samples]
            for idx in noise_indices:
                train_dataset.targets[idx] = th.randint(0, 10, size=(1,)).item() # TODO: make it deterministic

        images = th.stack([x[0] for x in train_dataset]).to(DEVICE)
        targets = th.tensor(train_dataset.targets).to(DEVICE)
        train_dataset = TensorDataset(images, targets)
        return train_dataset

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = self.get_dataset_with_label_noise(self.label_noise)
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
            # num_workers=2,#NUM_WORKERS,
            # shuffle=True,
            # pin_memory=True,
            # pin_memory_device=DEVICE
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            # num_workers=NUM_WORKERS,
            # shuffle=False,
            # pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            # num_workers=NUM_WORKERS,
            # shuffle=False,
            # pin_memory=True,
        )
