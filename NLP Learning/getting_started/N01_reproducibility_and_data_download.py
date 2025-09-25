import pandas as pd
import numpy as np
from pathlib import Path
import random
import torch

# For reproducibility across both CPU and GPU
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Additional seeds for CUDA operations to ensure reproducibility
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)  # for multi-GPU setups
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Set Pandas display options
pd.set_option('display.max_colwidth', 100)

# Data direction
data_path = Path('D:\Python Program\\NLP Learning\getting_started\data')

# Load the datasets
train_df = pd.read_csv(data_path / 'train.csv')
test_df = pd.read_csv(data_path / 'test.csv')