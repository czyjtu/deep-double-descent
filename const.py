import os
import torch
from pathlib import Path
from torch.optim.lr_scheduler import StepLR

PATH_DATASETS = Path(__file__).parent.parent / "data"
PATH_DATASETS.mkdir(exist_ok=True)

BATCH_SIZE = 128  # 128 was used in article
INITIAL_LR = 0.1
LR_UPDATE_STEPS = 512

NUM_WORKERS = 1  # int(os.cpu_count() / 2)
if torch.cuda.is_available():
    DEVICE = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
