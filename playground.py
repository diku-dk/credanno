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
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
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
# joint predictions
output_dir = './logs'
df = pd.read_csv(f'{output_dir}/pred_results.csv', index_col=0)

gt_FTR = df.loc[:, 'gt_subtlety':'gt_texture']
pd_FTR = df.loc[:, 'pd_subtlety':'pd_texture']

gt_CLS = df.loc[:, 'gt_malignancy']
pd_CLS = df.loc[:, 'pd_malignancy']

preds_FTR = abs(gt_FTR.to_numpy() - pd_FTR.to_numpy()) <= 1
accs_FTR = preds_FTR.sum(axis=0) / len(preds_FTR)
correctFTRs = preds_FTR.sum(axis=1)

preds_CLS = gt_CLS.to_numpy() == pd_CLS.to_numpy()
accs_CLS = preds_CLS.sum(axis=0) / len(preds_CLS)
correctCLSs = preds_CLS.sum(axis=0)

embds = np.asarray(df.loc[:, '0':])
embds_pca = PCA(n_components=0.99, whiten=False).fit_transform(embds)
embds_tsne = TSNE(n_components=2, init='random', learning_rate='auto').fit_transform(embds)
df['embd_tsne_dim0'] = embds_tsne[:, 0]
df['embd_tsne_dim1'] = embds_tsne[:, 1]

# %%
# seperate correct/incorrect
df_right = df[df.pd_malignancy == df.gt_malignancy]
df_wrong = df[df.pd_malignancy != df.gt_malignancy]

gt_FTR_right = df_right.loc[:, 'gt_subtlety':'gt_texture']
pd_FTR_right = df_right.loc[:, 'pd_subtlety':'pd_texture']

preds_FTR_right = abs(gt_FTR_right.to_numpy() - pd_FTR_right.to_numpy()) <= 1
accs_FTR = preds_FTR.sum(axis=0) / len(preds_FTR)
correctFTRs_right = preds_FTR_right.sum(axis=1)

gt_FTR_wrong = df_wrong.loc[:, 'gt_subtlety':'gt_texture']
pd_FTR_wrong = df_wrong.loc[:, 'pd_subtlety':'pd_texture']

preds_FTR_wrong = abs(gt_FTR_wrong.to_numpy() - pd_FTR_wrong.to_numpy()) <= 1
accs_FTR = preds_FTR.sum(axis=0) / len(preds_FTR)
correctFTRs_wrong = preds_FTR_wrong.sum(axis=1)

# %%
# plot histogram of correctly predicted FTRs
plt.hist(correctFTRs_right, bins=np.arange(-0.5, 9.5), density=True, alpha=0.4, color='tab:green');
plt.hist(correctFTRs_wrong, bins=np.arange(-0.5, 9.5), density=True, alpha=0.4, color='tab:red');

# %%
# plot t-SNE of learned EMBDs
plt.figure(figsize=(8, 8))
plt.scatter(df_right['embd_tsne_dim0'], df_right['embd_tsne_dim1'], alpha=0.4, color='tab:green');
plt.scatter(df_wrong['embd_tsne_dim0'], df_wrong['embd_tsne_dim1'], alpha=0.4, color='tab:red');
