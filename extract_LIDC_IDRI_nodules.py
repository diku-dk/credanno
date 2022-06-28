"""
 Copyright (c) 2022 * Lu

 Permission is hereby granted, free of charge, to any person obtaining a copy of
 this software and associated documentation files (the "Software"), to deal in
 the Software without restriction, including without limitation the rights to
 use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 the Software, and to permit persons to whom the Software is furnished to do so,
 subject to the following conditions:

 The above copyright notice and this permission notice shall be included in all
 copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 """

# %%
import os
import random
from matplotlib.pyplot import flag

import numpy as np

import pylidc as pl
from pylidc.utils import consensus
import torch

import torch
import skimage.measure as skm
import skimage.io as skio
import skimage.util as sku
import pandas as pd
from scipy import ndimage
import skimage.transform as skt


def make_LIDC_IDRI_imagenet(save_folder:str="../../../datasets/LIDC_IDRI/imagenet", split_point:float=0.7, img_shape=(32, 32), verbose=True):
    """split the dataset on nodule level, following 
    V. Baltatzis et al., “The Pitfalls of Sample Selection: A Case Study on Lung Nodule Classification,” 
    in Predictive Intelligence in Medicine, vol. 12928, I. Rekik, E. Adeli, S. H. Park, and J. Schnabel, Eds. 
    Cham: Springer International Publishing, 2021, pp. 201–211.

    Args:
        save_folder (str, optional): Defaults to "../../../datasets/LIDC_IDRI/imagenet".
        split_point (float, optional): portion of training data. Defaults to 0.7.
        img_shape (tuple, optional): output patch shape. Defaults to (32, 32).
        verbose (bool, optional): Defaults to True.
    """
    assert 0 <= split_point and split_point <= 1, "split_point must be in [0, 1]."
    seed_everything(42)

    scans = pl.query(pl.Scan).filter(
        pl.Scan.slice_thickness <= 2.5
    ) # "CT scans with a slice thickness greater than 2.5mm are removed according to clinical guidelines"
    pids_list = [scan.patient_id for scan in scans]
    print(f'{scans.count()} scans found.')

    splits = ['train', 'val']
    # make DataFrames to store the labels
    dfs = {}
    for split in splits:
        dfs[split] = pd.DataFrame()
        # make split sub-folders
        os.makedirs(f'{save_folder}/Image/{split}/0', exist_ok=True)
        os.makedirs(f'{save_folder}/Mask/{split}/0', exist_ok=True)
        os.makedirs(f'{save_folder}/Image/{split}/1', exist_ok=True)
        os.makedirs(f'{save_folder}/Mask/{split}/1', exist_ok=True)

    n = 0
    for pid in set(pids_list):
        scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
        # get the scan
        vol = scan.to_volume(verbose=verbose)
        # vol = normalise_vol(vol)
        # "every remaining scan is resampled to 1mm isotropic resolution across all three dimensions"
        img_shape_iso = (round(img_shape[0] / scan.pixel_spacing), round(img_shape[1] / scan.pixel_spacing))
        nods = scan.cluster_annotations(verbose=verbose)
        for nid, nod_anns in enumerate(nods):
            nod_saved = False   # flag whether a nodule is saved

            if len(nod_anns) < 3:
                continue    # "retaining only nodules that have been annotated by at least three radiologists"
            malignancy_median = np.median([ann.malignancy for ann in nod_anns])
            if malignancy_median == 3:
                continue    # nodules of median radiologists' score 3 were removed
            malignancy = 0 if malignancy_median < 3 else 1
            n += 1

            median_subtlety = np.median([ann.subtlety for ann in nod_anns])
            median_internalStructure = np.median([ann.internalStructure for ann in nod_anns])
            median_calcification = np.median([ann.calcification for ann in nod_anns])
            median_sphericity = np.median([ann.sphericity for ann in nod_anns])
            median_margin = np.median([ann.margin for ann in nod_anns])
            median_lobulation = np.median([ann.lobulation for ann in nod_anns])
            median_spiculation = np.median([ann.spiculation for ann in nod_anns])
            median_texture = np.median([ann.texture for ann in nod_anns])
            median_diameter = np.median([ann.diameter for ann in nod_anns])

            # random split on nodule level
            random_split = np.random.choice(splits, 1, p=[split_point, 1-split_point])[0]
            folder_image = f'{save_folder}/Image/{random_split}/{malignancy}'
            folder_mask = f'{save_folder}/Mask/{random_split}/{malignancy}'

            for ann in nod_anns:
                # if nod_saved:
                #     break
                img_id = f'{pid}_s{str(ann.scan_id)}_ann{str(ann.id)}_n{str(nid).zfill(2)}'

                # get image and mask
                try:
                    pad_dims = get_pad_dims(in_shape=vol[ann.bbox()].shape[:2], out_shape=img_shape_iso, random_pos=False)
                except AssertionError:
                    continue
                ann_bbox = ann.bbox(pad=pad_dims)
                (cx, cy, cz) = get_centre(ann_bbox)
                ann_bbox_matrix = ann.bbox_matrix(pad=pad_dims)
                coords = [(ann.centroid[d] - ann_bbox_matrix[d][0]) /
                            (ann_bbox_matrix[d][1] - ann_bbox_matrix[d][0]) for d in range(3)]
                if np.any(np.isnan(coords)):
                    continue
                image = vol[ann_bbox][:, :, cz]
                if image.shape[:2] != img_shape_iso:
                    continue
                try:
                    mask = ann.boolean_mask(pad=pad_dims)[:,:,cz]
                except IndexError:
                    continue

                image = skt.resize(image, img_shape)
                mask = skt.resize(mask, img_shape)

                skio.imsave(f'{folder_image}/{img_id}.png', sku.img_as_float(image), check_contrast=False)
                skio.imsave(f'{folder_mask}/{img_id}.png', sku.img_as_ubyte(mask), check_contrast=False)

                # get labels
                dfs[random_split] = dfs[random_split].append(
                    pd.Series(
                        {
                            'id': ann.id,
                            'scan_id': ann.scan_id,
                            'subtlety': median_subtlety,
                            'internalStructure': median_internalStructure,
                            'calcification': median_calcification,
                            'sphericity': median_sphericity,
                            'margin': median_margin,
                            'lobulation': median_lobulation,
                            'spiculation': median_spiculation,
                            'texture': median_texture,
                            'malignancy': malignancy_median,
                            'diameter': median_diameter,
                            'coords': coords,
                        },
                        name=img_id,
                    )
                )
                nod_saved = True
    print(f"Total nodules: {n}")
    for split in splits:
        dfs[split].to_csv(f'{save_folder}/meta_{split}.csv')
    return

# %% helper functions
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def pad_to_shape(img, out_shape):
    # pad the 3D image to a certain shape 
    img = np.asarray(img)
    (w, h) = img.shape[:2]
    if type(out_shape) is not tuple:
        out_shape = (out_shape, out_shape)
    w_pad = out_shape[0] - w
    h_pad = out_shape[1] - h            
    wl = w_pad // 2
    wr = w_pad - wl
    hl = h_pad // 2
    hr = h_pad - hl
    if img.ndim == 2:
        img_pad = np.pad(img, ((wl, wr), (hl, hr)), 'edge')
    else:
        img_pad = np.pad(img, ((wl, wr), (hl, hr), (0, 0)), 'edge')
    return img_pad

def get_pad_dims(in_shape, out_shape, random_pos=False, edge=5):
    """get the edge sizes for padding

    Args:
        in_shape (array or tuple): original shape
        out_shape (int or tuple): desired shape
        random_pos (boolean): make the object at a random position in the padded area

    Returns:
        list: list of 3 tuples
    """
    if type(out_shape) is not tuple:
        out_shape = (out_shape, out_shape)
    w_pad = out_shape[0] - in_shape[0]
    # w1 = o - i1
    # w2 = o - i2
    # i2 = 0.6 * i1
    # -> w2 = o - 0.6 * (o - w1) = 0.4 * o + 0.6 * w1
    h_pad = out_shape[1] - in_shape[1]
    if random_pos:
        assert w_pad > 2 * edge, f"w_pad(={w_pad}) must > 2*edge(={2*edge})"
        assert h_pad > 2 * edge, f"h_pad(={h_pad}) must > 2*edge(={2*edge})"
        wl = random.randint(edge, w_pad - edge)
        hl = random.randint(edge, h_pad - edge)
    else:
        wl = int(w_pad // 2)
        hl = int(h_pad // 2)
    wr = int(w_pad - wl)
    hr = int(h_pad - hl)
    pad_dims = [(wl, wr), (hl, hr), (0, 0)]
    return pad_dims

def get_centre(bbox):
    """get nodule centre location from bbox

    Args:
        bbox (class pylidc.Annotation.bbox): bounding box
    
    Returns:
        list: len(): 3 
    """
    cx = int(0.5*(bbox[0].stop - bbox[0].start))
    cy = int(0.5*(bbox[1].stop - bbox[1].start))
    cz = int(0.5*(bbox[2].stop - bbox[2].start))
    return (cx, cy, cz)


# %%
if __name__ == '__main__':
    make_LIDC_IDRI_imagenet(save_folder="../../../datasets/LIDC_IDRI/imagenet")