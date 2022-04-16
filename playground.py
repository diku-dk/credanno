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
df_FTR = pd.read_csv(f'{output_dir}/pred_resultsFTR.csv', index_col=0)
df_CLS = pd.read_csv(f'{output_dir}/pred_resultsCLS.csv', index_col=0)
# df = df_CLS.merge(df_FTR, on='img_id', how='inner')
df = df_CLS.fillna(df_FTR)

gt_FTR = df_FTR.loc[:, 'gt_subtlety':'gt_texture']
pd_FTR = df_FTR.loc[:, 'pd_subtlety':'pd_texture']

gt_CLS = df_CLS.loc[:, 'gt_malignancy']
pd_CLS = df_CLS.loc[:, 'pd_malignancy']

preds_FTR = abs(gt_FTR.to_numpy() - pd_FTR.to_numpy()) <= 1
accs_FTR = preds_FTR.sum(axis=0) / len(preds_FTR)
correctFTRs = preds_FTR.sum(axis=1)

preds_CLS = gt_CLS.to_numpy() == pd_CLS.to_numpy()
accs_CLS = preds_CLS.sum(axis=0) / len(preds_CLS)
correctCLSs = preds_CLS.sum(axis=0)

# %%
