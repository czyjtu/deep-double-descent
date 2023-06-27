import os
import torch
from pathlib import Path
from torch.optim.lr_scheduler import StepLR

PATH_DATASETS = Path(__file__).parent.parent / "data"
PATH_DATASETS.mkdir(exist_ok=True)
PATH_CHECKPOINTS_DIR = Path(__file__).parent / "model_checkpoints_examples"

BATCH_SIZE = 128  # 128 was used in article
INITIAL_LR = 0.1
LR_UPDATE_STEPS = 512

NUM_WORKERS = int(os.cpu_count() / 2)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
