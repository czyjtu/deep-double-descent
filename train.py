from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer

from tasks.cifar import LitCNN, TrainingConfig
from datamodules.cifar10 import Cifar10DataModule
from const import BATCH_SIZE
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar
import wandb
import click
import logging

logging.basicConfig(level=logging.INFO)

from const import BATCH_SIZE, NUM_WORKERS


@click.command()
@click.option("--steps", required=True, type=click.INT)
@click.option("--lr", required=True, type=click.FLOAT)
@click.option("--c", required=True, type=click.INT)
@click.option("--batch_size", required=False, type=click.INT, default=BATCH_SIZE)
@click.option("--num_workers", required=False, type=click.INT, default=NUM_WORKERS)
def main(steps, batch_size, lr, c, num_workers):
    config = TrainingConfig(
        steps=steps, lr=lr, c=c, batch_size=batch_size, num_workers=num_workers
    )
    logging.info(f"config used: {config}")

    model = LitCNN(config)
    data_module = Cifar10DataModule(BATCH_SIZE)

    logger = WandbLogger(project="deep-double-descent", log_model=True, entity="czyjtu")
    wandb.log(config.dict())

    trainer = Trainer(
        max_steps=config.steps,
        max_epochs=-1,
        accelerator="auto",
        logger=logger,
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            TQDMProgressBar(refresh_rate=10),
        ],
    )
    trainer.fit(model, data_module)
    trainer.test(model, data_module)


if __name__ == "__main__":
    main()
