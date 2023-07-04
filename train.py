from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from tasks.cifar import LitCNN, TrainingConfig
from datamodules.cifar10 import Cifar10DataModule
from const import BATCH_SIZE
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    TQDMProgressBar,
    ModelCheckpoint,
)
import wandb
import click
import logging
import train
import os
from pathlib import Path
from const import BATCH_SIZE, NUM_WORKERS

logging.basicConfig(level=logging.INFO)

SCRATCH_DIR = Path(os.environ.get("SCRATCH_LOCAL", "."))
logging.info(f"Scratch DIR used: {SCRATCH_DIR}")


@click.command()
@click.option("--steps", required=True, type=click.INT)
@click.option("--lr", required=True, type=click.FLOAT)
@click.option("--c", required=True, type=click.INT)
@click.option("--batch_size", required=False, type=click.INT, default=BATCH_SIZE)
@click.option("--num_workers", required=False, type=click.INT, default=NUM_WORKERS)
@click.option("--label_noise", required=False, type=click.FLOAT, default=0.1)
def main(steps, batch_size, lr, c, num_workers, label_noise):
    config = TrainingConfig(
        steps=steps,
        lr=lr,
        c=c,
        batch_size=batch_size,
        num_workers=num_workers,
        label_noise=label_noise,
    )
    logging.info(f"config used: {config}")

    model = LitCNN(config)
    data_module = Cifar10DataModule(
        config.batch_size, augmentation=config.augment, label_noise=config.label_noise
    )

    logger = WandbLogger(
        project="deep-double-descent",
        log_model="all",
        entity="czyjtu",
        save_dir=SCRATCH_DIR,
    )
    wandb.log(config.dict())

    trainer = Trainer(
        max_steps=config.steps,
        max_epochs=-1,
        accelerator="auto",
        logger=logger,
        log_every_n_steps=3,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
            ModelCheckpoint(
                SCRATCH_DIR / "model_checkpoints", save_top_k=-1, every_n_epochs=3
            ),
        ],
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
