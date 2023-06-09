import os 
import torch 
from pathlib import Path

PATH_DATASETS = (Path(__file__).parent.parent / "data")
PATH_DATASETS.mkdir(exist_ok=True)

BATCH_SIZE = 256 if torch.cuda.is_available() else 64
NUM_WORKERS = int(os.cpu_count() / 2)
