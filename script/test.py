import gymnasium as gym  # 修改这里
from gymnasium import spaces  # 修改这里
import numpy as np

import torch
import torch.nn.functional as F


a= torch.tensor([[1,2,3],[4,5,6]])
print(a)
b= a.unsqueeze(-1)
print(b)
print(a.dim())