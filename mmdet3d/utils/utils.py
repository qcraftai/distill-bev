import random
import numpy as np
import torch

def set_up_deterministic_seed(seed=42, rank=0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True