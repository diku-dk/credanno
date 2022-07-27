# %% 
import os
import argparse
import json
from pathlib import Path
import sys
import random
import itertools

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

def calculate_correct_FTR(pd, gt: pd.DataFrame, allowed_range=1):
    if not isinstance(pd, np.ndarray):
        pd = pd.to_numpy()
    preds = abs(gt.to_numpy() - pd) <= allowed_range
    accs = preds.sum(axis=0) / len(preds)
    corrects = preds.sum(axis=1)
    return corrects, accs

def prob_nFTRs_correct_dict(row_prob) -> dict:
    if not isinstance(row_prob, np.ndarray):
        row_prob = row_prob.to_numpy()
    n_total_ftrs = len(row_prob)
    n_ftrs = set(range(n_total_ftrs))
    row_prob_neg = 1 - row_prob
    prob_nFTRs_correct = lambda n: sum([row_prob[list(combo)].prod() * row_prob_neg[list(n_ftrs - set(combo))].prod() for combo in itertools.combinations(n_ftrs, n)])
    return {n: prob_nFTRs_correct(n) for n in range(n_total_ftrs + 1)}

def extract_results(method: str, output_dir: str = './logs'):
    """Extract joint prediction results

    Args:
        method (str): method str
        output_dir (str, optional): dir of result csv. Defaults to './logs'.

    Returns:
        DataFrames
    """
    df = pd.read_csv(f'{output_dir}/pred_results_{method}.csv', index_col=0)
    pd_FTR = df.loc[:, 'pd_subtlety':'pd_texture']
    gt_FTR = df.loc[:, 'gt_subtlety':'gt_texture']
    pd_CLS = df.loc[:, 'pd_malignancy']
    gt_CLS = df.loc[:, 'gt_malignancy']
    return pd_FTR, gt_FTR, pd_CLS, gt_CLS


# %%
# accuracy of competitors
header = ['method', 'subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']

method_list = [
    {'method':'HSCNN', 'subtlety':.719, 'calcification':.908, 'sphericity':.552, 'margin':.725, 'texture':.834, 'malignancy':.842},
    {'method':'X-Caps', 'subtlety':.9039, 'sphericity':.8544, 'margin':.8414, 'lobulation':.7069, 'spiculation':.7523, 'texture':.9310, 'malignancy':.8639},
    {'method':'MSN-JCN', 'subtlety':.7077, 'calcification':.9407, 'sphericity':.6863, 'margin':.7888, 'lobulation':.9475, 'spiculation':.9375, 'texture':.89, 'malignancy':.8707},
    {'method':'WeakSup (1:3)', 'subtlety':.668, 'internalStructure':.973, 'calcification':.915, 'sphericity':.664, 'margin':.796, 'lobulation':.743, 'spiculation':.814, 'texture':.822, 'malignancy':.891},
    {'method':'WeakSup (1:5)', 'subtlety':.431, 'internalStructure':.701, 'calcification':.639, 'sphericity':.424, 'margin':.585, 'lobulation':.406, 'spiculation':.387, 'texture':.512, 'malignancy':.824},
    # {'method':'cRedAnno(1%)', 'subtlety':.9181, 'calcification':.9337, 'sphericity':.9649, 'margin':.9077, 'lobulation':.8973, 'spiculation':.9233, 'texture':.9376, 'malignancy':.8596},
    # {'method':'cRedAnno(kNN)', 'subtlety':.96359, 'calcification':.92588, 'sphericity':.96229, 'margin':.94148, 'lobulation':.90897, 'spiculation':.92328, 'texture':.92718, 'malignancy':.88947},
]

df_acc = pd.DataFrame(method_list, columns=header)
df_acc = df_acc.fillna(value=1)
# df_acc.set_index('method')

df_probs = pd.DataFrame(columns=range(len(ftr_CLASSES) + 1))
for _, row in df_acc.iterrows():
    df_probs.loc[row['method']] = prob_nFTRs_correct_dict(row['subtlety':'texture'])
# df_probs.reset_index(inplace=True)
# df_probs = df_probs.rename(columns = {'index':'method'})

# random prediction
output_dir = './logs/vits16_pretrain_full_2d_ann'
gt_FTR = extract_results('kNN_250', os.path.join(output_dir, 'results'))[1]
pd_FTR_rand = np.random.randint(low=0, high=[len(f) for f in ftr_CLASSES.values()], size=gt_FTR.shape)
correctFTRs_rand, accs_FTR = calculate_correct_FTR(pd_FTR_rand, gt_FTR)

# %%
# Histogram of correctly predicted FTRs
dark=False

if dark:
    bg_color = '#181717'
    plt.style.use(['ggplot','dark_background'])
    plt.rcParams['axes.facecolor'] = '#212020'
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['grid.color'] = bg_color
    plt.rcParams['axes.edgecolor'] = bg_color
    plt.rcParams['savefig.facecolor'] = bg_color
    label_color = 'white'
else:
    sns.reset_orig()
    sns.set_theme()
    plt.style.use('ggplot')
    label_color = 'black'

plt.figure(figsize=(10, 7))

# plot reference (random prediction)
hist, _ = np.histogram(correctFTRs_rand, bins=np.arange(-0.5, 9.5), density=True)
c = '#4C72B0'
ax = sns.histplot(x=df_probs.columns, weights=hist, label='Random', discrete=True, common_norm=False, stat="probability", alpha=0.3, kde=True, kde_kws={'bw_adjust':.7}, line_kws={'linewidth': 3}, color=c, edgecolor=c);

# plot competitors
palette = itertools.cycle(sns.cubehelix_palette(light=0.7, dark=0.3))
for method in df_probs.index:
    c = next(palette)
    ax = sns.histplot(x=df_probs.columns, weights=df_probs.loc[method], label=method, discrete=True, common_norm=False, stat="probability", alpha=0.3, kde=True, kde_kws={'bw_adjust':.7}, line_kws={'linewidth': 3}, color=c, edgecolor=c);

# plot ours
label_dict = {'1p':'1%, trained', '100p':'100%, trained', 'kNN_150_10p':'10%, 150-NN'}
palette = itertools.cycle(sns.color_palette("ch:2,r=.2,d=.0,l=.7"))
for method, label in label_dict.items():
    pd_FTR, gt_FTR = extract_results(method, os.path.join(output_dir, 'results'))[:2]
    correctFTRs, accs_FTR = calculate_correct_FTR(pd_FTR, gt_FTR)    
    hist, _ = np.histogram(correctFTRs, bins=np.arange(-0.5, 9.5), density=True)
    # sns.histplot(data=correctFTRs, bins=df_probs.columns, label='cRedAnno', discrete=True, common_norm=False, stat="probability", alpha=0.3, kde=True, kde_kws={'bw_adjust':2.5}, color='green')
    c = next(palette)
    ax = sns.histplot(x=df_probs.columns, weights=hist, label=f'cRedAnno ({label})', discrete=True, common_norm=False, stat="probability", alpha=0.3, kde=True, kde_kws={'bw_adjust':.7}, line_kws={'linewidth': 3}, color=c, edgecolor=c);

plt.legend(fontsize=16.5, framealpha=0.4, loc='upper left')
plt.xlabel('Number of correctly predicted nodule attributes', fontsize=21, color=label_color)
plt.ylabel('Probability', fontsize=21, color=label_color)
plt.xticks(df_probs.columns)
plt.tick_params(labelsize=16.5)

if dark:
    plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'prob_ftr_prediction.svg')}", format='svg', bbox_inches='tight', edgecolor='none')
else:
    plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'prob_ftr_prediction.pdf')}", format='pdf', bbox_inches='tight')



# %%
# compute T-SNE
output_dir = './logs/vits16_pretrain_full_2d_ann'
df = pd.read_csv(f"{os.path.join(output_dir, 'results', 'pred_results_kNN_250.csv')}", index_col=0)

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
# %% Save t-SNE plots
dark = False
# sns.reset_orig()
if dark:
    bg_color = '#181717'
    plt.style.use(['ggplot','dark_background'])
    plt.rcParams['axes.facecolor'] = bg_color
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['grid.color'] = bg_color
    plt.rcParams['axes.edgecolor'] = '#545454'
    label_color = 'white'

output_dir = './logs/vits16_pretrain_full_2d_ann'
xlim = (-113, 77)
ylim = (-103, 67)
for task, d in legend_dict.items():
    if not dark:
        sns.set_theme(style="white", palette=None) # palette='viridis'
    if task == 'malignancy':
        fig = sns.jointplot(data=df, x='embd_tsne_dim0', y='embd_tsne_dim1', hue=f'gt_{task}', kind='scatter', xlim=xlim, ylim=ylim, 
                            palette=[sns.color_palette("flare", as_cmap=True).colors[30], sns.color_palette("flare", as_cmap=True).colors[-30]], 
                            alpha=0.6, s=40)
    else:
        # break
        fig = sns.jointplot(data=df, x='embd_tsne_dim0', y='embd_tsne_dim1', hue=f'gt_{task}', kind='scatter', xlim=xlim, ylim=ylim, 
                            palette='flare', alpha=0.6, s=40)
    fig.plot_joint(sns.kdeplot, zorder=0, levels=3, alpha=0.3)
    fig.set_axis_labels('', '')
    fig.ax_joint.set_xticks([])
    fig.ax_joint.set_yticks([])
    handles, labels = fig.ax_joint.get_legend_handles_labels()
    fig.ax_joint.legend(handles=handles, labels=d, fontsize=15, framealpha=0.3, handletextpad=0.1, handlelength=1, loc='lower left')
    if dark:
        plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'embd_tsne_{task}.svg')}", format='svg', 
                    bbox_inches='tight', transparent=True)
    else:
        # plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'embd_tsne_{task}.png')}", bbox_inches='tight', dpi=300)
        plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'embd_tsne_{task}.pdf')}", format='pdf', bbox_inches='tight')

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
    sns.reset_orig()
    plt.style.use('ggplot')
    label_color = 'black'

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 7), sharex='col', sharey='row')
line1, = ax.plot(df['Annotation percentage'], df['Baseline'], label='Baseline', marker='X', linewidth=5, markersize=20, alpha=0.8, c='#B84878')
line2, = ax.plot(df['Annotation percentage'], df['cRedAnno'], label='cRedAnno', marker='o', linewidth=5, markersize=20, alpha=0.8, c='#2F847C')
ax.set_xscale('log')
ax.invert_xaxis()
# ax.set_xlim(1, 1e-2)
ax.set_ylim(70, 90)
ax.legend(handles=[line2, line1], fontsize=21, framealpha=0.4, loc='lower left')
ax.set_xlabel('Percentage of annotations used (logarithmic scale)', fontsize=21, color=label_color)
ax.set_ylabel('Accuracy of malignancy prediction [%]', fontsize=21, color=label_color)
ax.tick_params(labelsize=20)

if dark:
    plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'anno_reduce.svg')}", 
                    format='svg', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    # plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'anno_reduce.png')}", format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'anno_reduce.pdf')}", format='pdf', bbox_inches='tight')


# %%
# Radar plot function
def radar_factory(num_vars, frame='circle'):
    """Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle' | 'polygon'}
        Shape of frame surrounding axes.

    """
    from matplotlib.patches import Circle, RegularPolygon
    from matplotlib.path import Path
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    from matplotlib.spines import Spine
    from matplotlib.transforms import Affine2D
    
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    
    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels, fontsize=None):
            self.set_thetagrids(np.degrees(theta), labels, fontsize=fontsize)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

        def draw(self, renderer):
            """ Draw. If frame is polygon, make gridlines polygon-shaped """
            if frame == 'polygon':
                gridlines = self.yaxis.get_gridlines()
                for gl in gridlines:
                    gl.get_path()._interpolation_steps = num_vars
            super().draw(renderer)


        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)


                return {'polar': spine}
            else:
                raise ValueError("unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


# %%
# Radar plot

## accuracy of competitors
header = ['method', 'Malignancy', 'Sub', 'Cal', 'Sph', 'Mar', 'Lob', 'Spi', 'Tex', 'internalStructure']

method_list_fullanno = [
    {'method':'HSCNN', 'Sub':.719, 'Cal':.908, 'Sph':.552, 'Mar':.725, 'Tex':.834, 'Malignancy':.842},
    {'method':'X-Caps', 'Sub':.9039, 'Sph':.8544, 'Mar':.8414, 'Lob':.7069, 'Spi':.7523, 'Tex':.9310, 'Malignancy':.8639},
    {'method':'MSN-JCN', 'Sub':.7077, 'Cal':.9407, 'Sph':.6863, 'Mar':.7888, 'Lob':.9475, 'Spi':.9375, 'Tex':.89, 'Malignancy':.8707},
    # {'method':'WeakSup (1:3)', 'Sub':.668, 'internalStructure':.973, 'Cal':.915, 'Sph':.664, 'Mar':.796, 'Lob':.743, 'Spi':.814, 'Tex':.822, 'Malignancy':.891},
    # {'method':'WeakSup (1:5)', 'Sub':.431, 'internalStructure':.701, 'Cal':.639, 'Sph':.424, 'Mar':.585, 'Lob':.406, 'Spi':.387, 'Tex':.512, 'Malignancy':.824},
    # {'method':'cRedAnno (1%)', 'Sub':.9181, 'Cal':.9337, 'Sph':.9649, 'Mar':.9077, 'Lob':.8973, 'Spi':.9233, 'Tex':.9376, 'Malignancy':.8596},
    {'method':'cRedAnno (trained)', 'Sub':.9584, 'Cal':.9597, 'Sph':.9740, 'Mar':.9649, 'Lob':.9415, 'Spi':.9441, 'Tex':.9701, 'Malignancy':.8830},
    {'method':'cRedAnno (250-NN)', 'Sub':.96359, 'Cal':.92588, 'Sph':.96229, 'Mar':.94148, 'Lob':.90897, 'Spi':.92328, 'Tex':.92718, 'Malignancy':.88947},
]
df_acc_fullanno = pd.DataFrame(method_list_fullanno, columns=header)

method_list_partialanno = [
    # {'method':'HSCNN', 'Sub':.719, 'Cal':.908, 'Sph':.552, 'Mar':.725, 'Tex':.834, 'Malignancy':.842},
    # {'method':'X-Caps', 'Sub':.9039, 'Sph':.8544, 'Mar':.8414, 'Lob':.7069, 'Spi':.7523, 'Tex':.9310, 'Malignancy':.8639},
    # {'method':'MSN-JCN', 'Sub':.7077, 'Cal':.9407, 'Sph':.6863, 'Mar':.7888, 'Lob':.9475, 'Spi':.9375, 'Tex':.89, 'Malignancy':.8707},
    {'method':'WeakSup (1:3)', 'Sub':.668, 'internalStructure':.973, 'Cal':.915, 'Sph':.664, 'Mar':.796, 'Lob':.743, 'Spi':.814, 'Tex':.822, 'Malignancy':.891},
    {'method':'WeakSup (1:5)', 'Sub':.431, 'internalStructure':.701, 'Cal':.639, 'Sph':.424, 'Mar':.585, 'Lob':.406, 'Spi':.387, 'Tex':.512, 'Malignancy':.824},
    {'method':'cRedAnno (1%, trained)', 'Sub':.9181, 'Cal':.9337, 'Sph':.9649, 'Mar':.9077, 'Lob':.8973, 'Spi':.9233, 'Tex':.9376, 'Malignancy':.8609},
    {'method':'cRedAnno (10%, 150-NN)', 'Sub':.9532, 'Cal':.8947, 'Sph':.9701, 'Mar':.9389, 'Lob':.9181, 'Spi':.9051, 'Tex':.9285, 'Malignancy':.8817},
    # {'method':'cRedAnno (trained)', 'Sub':.9584, 'Cal':.9597, 'Sph':.9740, 'Mar':.9649, 'Lob':.9415, 'Spi':.9441, 'Tex':.9701, 'Malignancy':.8830},
    # {'method':'cRedAnno (250-NN)', 'Sub':.96359, 'Cal':.92588, 'Sph':.96229, 'Mar':.94148, 'Lob':.90897, 'Spi':.92328, 'Tex':.92718, 'Malignancy':.88947},
]
df_acc_partialanno = pd.DataFrame(method_list_partialanno, columns=header)

# %%
# plot
dark=False
output_dir = './logs/vits16_pretrain_full_2d_ann'

if dark:
    bg_color = '#181717'
    plt.style.use(['ggplot','dark_background'])
    plt.rcParams['axes.facecolor'] = '#212020'
    plt.rcParams['figure.facecolor'] = bg_color
    plt.rcParams['grid.color'] = bg_color
    plt.rcParams['axes.edgecolor'] = bg_color
    plt.rcParams['savefig.facecolor'] = bg_color
    label_color = 'white'
else:
    sns.reset_orig()
    sns.set_theme()
    plt.style.use('ggplot')
    label_color = 'black'
plt.rcParams['legend.title_fontsize'] = '21'

def plot_pizza(df_acc):
    columns_titles = ['method', 'Malignancy', 'Sub', 'Cal', 'Sph', 'Mar', 'Lob', 'Spi', 'Tex', 'internalStructure']
    df_acc = df_acc.reindex(columns=columns_titles)

    theta = radar_factory(8, frame='polygon')

    spoke_labels = df_acc.columns[1:-1]
    # title, case_data = data_sample[0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
    # fig.subplots_adjust(top=0.85, bottom=0.05)

    # plot competitors
    palette = itertools.cycle(sns.color_palette("flare"))
    markers = itertools.cycle(('p', '*', '.', 'P', 'X', '+', 'x', 'h', 'H', '1')) 
    for _, row in df_acc[:-2].iterrows():
        values = row['Malignancy':'Tex'].to_numpy()
        c = next(palette)
        m = next(markers)
        line = ax.plot(theta, values, label=row['method'], color=c, marker=m, markersize=12)
        ax.fill(theta[pd.notna(values)], values[pd.notna(values)], alpha=0.1, color=c, label='_nolegend_')
    # plot ours
    palette = itertools.cycle(sns.color_palette("ch:2,r=.2,d=.0,l=.7"))
    for _, row in df_acc[-2:].iterrows():
        values = row['Malignancy':'Tex'].to_numpy()
        c = next(palette)
        line = ax.plot(theta, values, label=row['method'], color=c, lw=4)
        ax.fill(theta[pd.notna(values)], values[pd.notna(values)], alpha=0.1, color=c, label='_nolegend_')


    ax.set_rlabel_position(0)
    ax.set_rgrids([0., 0.2, 0.4, 0.6, 0.8, 1.], fontsize=16.5)
    ax.set_rlim(0, 1)
    # ax.set_title(title,  position=(0.5, 1.1), ha='center')
    ax.xaxis.set_tick_params(pad=10)
    # ax.set_varlabels(spoke_labels, fontsize=21)
    plt.xticks(theta, spoke_labels, size=21, color=label_color)


plot_pizza(df_acc_fullanno)
plt.legend(fontsize=16.5, framealpha=0.4, labelcolor=label_color, title="Full annotation", loc='lower right', bbox_to_anchor=(1.6, 0.))
if dark:
    plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'pizza_fullanno.svg')}", 
                    format='svg', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    # plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'pizza_fullanno.png')}", format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'pizza_fullanno.pdf')}", format='pdf', bbox_inches='tight')


plot_pizza(df_acc_partialanno)
plt.legend(fontsize=16.5, framealpha=0.4, labelcolor=label_color, title="Partial annotation", loc='lower right', bbox_to_anchor=(1.6, 0.))
if dark:
    plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'pizza_partialanno.svg')}", 
                    format='svg', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
else:
    # plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'pizza_partialanno.png')}", format='png', dpi=300, bbox_inches='tight')
    plt.savefig(f"{os.path.join(output_dir, 'results', 'imgs', f'pizza_partialanno.pdf')}", format='pdf', bbox_inches='tight')


# plt.show()
