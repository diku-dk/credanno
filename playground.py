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
import seaborn as sns

import utils
import vision_transformer as vits
import data_LIDC_IDRI as data

# %%
# dataset setting
ftr_CLASSES = {
    "subtlety": [1, 2, 3, 4, 5],
    "internalStructure": [1, 2, 3, 4], # 2, 3, never appeared
    "calcification": [1, 2, 3, 4, 5, 6], # 1, never appeared
    "sphericity": [1, 2, 3, 4, 5], # 1, never appeared
    "margin": [1, 2, 3, 4, 5],
    "lobulation": [1, 2, 3, 4, 5], 
    "spiculation": [1, 2, 3, 4, 5],
    "texture": [1, 2, 3, 4, 5],
    # "malignancy": [1, 2, 3, 4, 5],
}

# # compute class counts
# df = pd.DataFrame(valset.img_ftr_ids)
# for fk in ftr_CLASSES.keys():
#     print(f"{fk}: {dict(zip(*np.unique(df[fk], return_counts=True)))}")

def calculate_correct_FTR(pd, gt:pd.DataFrame, allowed_range=1):
    if not isinstance(pd, np.ndarray):
        pd = pd.to_numpy()
    preds = abs(gt.to_numpy() - pd) <= allowed_range
    accs = preds.sum(axis=0) / len(preds)
    corrects = preds.sum(axis=1)
    return corrects, accs


# %%
# joint predictions
output_dir = './logs'
df = pd.read_csv(f'{output_dir}/pred_results.csv', index_col=0)

gt_FTR = df.loc[:, 'gt_subtlety':'gt_texture']
pd_FTR = df.loc[:, 'pd_subtlety':'pd_texture']

gt_CLS = df.loc[:, 'gt_malignancy']
pd_CLS = df.loc[:, 'pd_malignancy']

# preds_FTR = abs(gt_FTR.to_numpy() - pd_FTR.to_numpy()) <= 1
# accs_FTR = preds_FTR.sum(axis=0) / len(preds_FTR)
# correctFTRs = preds_FTR.sum(axis=1)

preds_CLS = gt_CLS.to_numpy() == pd_CLS.to_numpy()
accs_CLS = preds_CLS.sum(axis=0) / len(preds_CLS)
correctCLSs = preds_CLS.sum(axis=0)

# %%
# plot histogram of correctly predicted FTRs

allowed_range = 1

# trained model
correctFTRs, accs_FTR = calculate_correct_FTR(pd_FTR, gt_FTR, allowed_range)
plt.hist(correctFTRs, bins=np.arange(-0.5, 9.5), density=True, alpha=0.4, color='tab:green');

# random prediction
pd_FTR_rand = np.random.randint(low=0, high=[len(f) for f in ftr_CLASSES.values()], size=gt_FTR.shape)
correctFTRs, accs_FTR = calculate_correct_FTR(pd_FTR_rand, gt_FTR, allowed_range)
plt.hist(correctFTRs, bins=np.arange(-0.5, 9.5), density=True, alpha=0.4, color='tab:red');

# most probable combo
combos, counts = np.unique(gt_FTR, return_counts=True, axis=0)
pd_FTR_combo = np.tile(combos[np.argmax(counts)], [len(gt_FTR), 1])
correctFTRs, accs_FTR = calculate_correct_FTR(pd_FTR_combo, gt_FTR, allowed_range)
plt.hist(correctFTRs, bins=np.arange(-0.5, 9.5), density=True, alpha=0.4, color='tab:orange');

# most probable single
sing_pred = [gt_FTR[col].value_counts().idxmax() for col in gt_FTR.columns]
if allowed_range == 1:
    # smart naive to avoid edge
    for i in range(len(sing_pred)):
        if sing_pred[i] == max(list(ftr_CLASSES.values())[i]) - 1:
            sing_pred[i] -= allowed_range
        elif sing_pred[i] == 0:
            sing_pred[i] += allowed_range
pd_FTR_single = np.tile(sing_pred, [len(gt_FTR), 1])
correctFTRs, accs_FTR = calculate_correct_FTR(pd_FTR_single, gt_FTR, allowed_range)
plt.hist(correctFTRs, bins=np.arange(-0.5, 9.5), density=True, alpha=0.4, color='tab:blue');



# %%
# seperate correct/incorrect
df_right = df[df.pd_malignancy == df.gt_malignancy]
df_wrong = df[df.pd_malignancy != df.gt_malignancy]

gt_FTR_right = df_right.loc[:, 'gt_subtlety':'gt_texture']
pd_FTR_right = df_right.loc[:, 'pd_subtlety':'pd_texture']

preds_FTR_right = abs(gt_FTR_right.to_numpy() - pd_FTR_right.to_numpy()) <= 1
correctFTRs_right = preds_FTR_right.sum(axis=1)

gt_FTR_wrong = df_wrong.loc[:, 'gt_subtlety':'gt_texture']
pd_FTR_wrong = df_wrong.loc[:, 'pd_subtlety':'pd_texture']

preds_FTR_wrong = abs(gt_FTR_wrong.to_numpy() - pd_FTR_wrong.to_numpy()) <= 1
correctFTRs_wrong = preds_FTR_wrong.sum(axis=1)

# %%
# plot histogram of correctly predicted FTRs
plt.hist(correctFTRs_right, bins=np.arange(-0.5, 9.5), density=True, alpha=0.4, color='tab:green');
plt.hist(correctFTRs_wrong, bins=np.arange(-0.5, 9.5), density=True, alpha=0.4, color='tab:red');

# %%
# compute T-SNE
embds = np.asarray(df.loc[:, '0':])
embds_pca = PCA(n_components=0.99, whiten=False).fit_transform(embds)
embds_tsne = TSNE(n_components=2, init='random', learning_rate='auto').fit_transform(embds)
df['embd_tsne_dim0'] = embds_tsne[:, 0]
df['embd_tsne_dim1'] = embds_tsne[:, 1]

# %%
# plot t-SNE of learned EMBDs
plt.figure(figsize=(8, 8))
plt.scatter(df[df.pd_malignancy == 0]['embd_tsne_dim0'], df[df.pd_malignancy == 0]['embd_tsne_dim1'], alpha=0.4, color='tab:blue');
plt.scatter(df[df.pd_malignancy == 1]['embd_tsne_dim0'], df[df.pd_malignancy == 1]['embd_tsne_dim1'], alpha=0.4, color='tab:orange');

# %%
legend_dict = {
    'malignancy':['Unlikely', 'Suspicious'],
    'subtlety':['Extremely Subtle', 'Moderately Subtle', 'Fairly Subtle', 'Moderately Obvious', 'Obvious'],
    'internalStructure':['Soft Tissue', 'Fluid', 'Fat', 'Air'],
    'calcification':[#'Popcorn', 'Laminated', 
        'Solid', 'Non-central', 'Central', 'Absent',],
    'sphericity':[#'Linear', 
        'Ovoid/Linear', 'Ovoid', 'Ovoid/Round', 'Round'],
    'margin':['Poorly Defined', 'Near Poorly Defined', 'Medium Margin', 'Near Sharp', 'Sharp'],
    'lobulation':['No Lobulation', 'Nearly No Lobulation', 'Medium Lobulation', 'Near Marked Lobulation', 'Marked Lobulation'],
    'spiculation':['No Spiculation', 'Nearly No Spiculation', 'Medium Spiculation', 'Near Marked Spiculation', 'Marked Spiculation'],
    'texture':['Non-Solid/GGO', 'Non-Solid/Mixed', 'Part Solid/Mixed', 'Solid/Mixed', 'Solid'],
}
# %% Save plots
for task, d in legend_dict.items():
    sns.set_theme(style="white", palette=None) # palette='viridis'
    if task == 'malignancy':
        fig = sns.jointplot(data=df, x='embd_tsne_dim0', y='embd_tsne_dim1', hue=f'gt_{task}', kind='scatter', xlim=(-105, 85), ylim=(-85, 95), palette=[sns.color_palette("flare", as_cmap=True).colors[30], sns.color_palette("flare", as_cmap=True).colors[-30]], alpha=0.6, s=40)
    else:
        fig = sns.jointplot(data=df, x='embd_tsne_dim0', y='embd_tsne_dim1', hue=f'gt_{task}', kind='scatter', xlim=(-105, 85), ylim=(-85, 95), palette='flare', alpha=0.6, s=40)
    handles, labels = fig.ax_joint.get_legend_handles_labels()
    fig.ax_joint.legend(handles=handles, labels=d, title=task.capitalize(), loc='upper left')
    plt.savefig(f"{os.path.join(output_dir, f'embd_tsne_{task}.png')}", bbox_inches='tight', dpi=300)

# %% 
# Get transformed sample image for illustration
from main_dino import DataAugmentationDINO
from PIL import Image
from torchvision.utils import save_image

def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

img_path = '/DATA/lu/datasets/LIDC_IDRI/imagenet_2d_ann/Image/train/1/LIDC-IDRI-0014_s25_ann216_n00.png'
img = pil_loader(img_path)
transform = DataAugmentationDINO(global_crops_scale=(0.4, 1.), local_crops_scale=(0.05, 0.4), local_crops_number=8)
img_transformed = transform(img)

save_image(img_transformed[1], './logs/LIDC-IDRI-0014_s25_ann216_n00_aug1.png')
# save_image(img_transformed[:2], './logs/LIDC-IDRI-0014_s25_ann216_n00_augs.png')
# save_image(img_transformed[2:], './logs/LIDC-IDRI-0014_s25_ann216_n00_aug_local.png')

from data_LIDC_IDRI import RandomRotation
transform = pth_transforms.Compose([
                    pth_transforms.RandomResizedCrop(224),
                    pth_transforms.RandomHorizontalFlip(),
                    pth_transforms.RandomVerticalFlip(),
                    RandomRotation(angles=[0, 90, 180, 270]),
                    pth_transforms.GaussianBlur(kernel_size=1),
                    pth_transforms.ToTensor(),
])
img_transformed2 = transform(img)

save_image(img_transformed2, './logs/LIDC-IDRI-0014_s25_ann216_n00_aug2.png')


# %%
# Annotation reduction plots
dark = False
output_dir = './logs/vits16_pretrain_full_2d_ann'
df = pd.read_csv(f"{os.path.join(output_dir, 'results', 'anno_reduce.csv')}")

if dark:
    bg_color = '#181717'
    plt.style.use(['ggplot','dark_background'])
    plt.rcParams['axes.facecolor'] = '#212020'
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['grid.color'] = bg_color
    plt.rcParams['axes.edgecolor'] = bg_color
    label_color = 'white'
else:
    plt.style.use('ggplot')
    label_color = 'black'

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7), sharex='col', sharey='row')
line1, = ax.plot(df['Annotation percentage'], df['Baseline'], label='Baseline', marker='X', linewidth=5, markersize=20, alpha=0.8, c='#B84878')
line2, = ax.plot(df['Annotation percentage'], df['MinAnno'], label='MinAnno', marker='o', linewidth=5, markersize=20, alpha=0.8, c='#2F847C')
ax.set_xscale('log')
ax.invert_xaxis()
# ax.set_xlim(1, 1e-2)
ax.set_ylim(70, 90)
ax.legend(handles=[line2, line1], fontsize='xx-large', framealpha=0.4, loc='lower left')
ax.set_xlabel('Percentage of annotations used', fontsize='xx-large', color=label_color)
ax.set_ylabel('Accuracy of malignancy prediction [%]', fontsize='xx-large', color=label_color)
ax.tick_params(labelsize='x-large')

if dark:
    plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'anno_reduce.svg')}", 
                    format='svg', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    # plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'anno_reduce.png')}", format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'anno_reduce.pdf')}", format='pdf', bbox_inches='tight')