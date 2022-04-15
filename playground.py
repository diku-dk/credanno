# %% 
import os
import argparse
import json
from pathlib import Path
import sys
import random

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

import utils
import vision_transformer as vits
import data_LIDC_IDRI as data

# %%
output_dir = './logs'
df = pd.read_csv(f'{output_dir}/pred_results.csv', index_col=0)
f_gt = df.loc[:, 'gt_subtlety':'gt_texture']
f_pd = df.loc[:, 'pd_subtlety':'pd_texture']
pred_array = abs(f_gt.to_numpy() - f_pd.to_numpy()) <= 1
accs = pred_array.sum(axis=0) / len(pred_array)
correctFTRs = pred_array.sum(axis=1)

# %%
