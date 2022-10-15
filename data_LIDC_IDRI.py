# %%
import os
import random

import torch
import torch.utils.data
from torchvision.datasets.folder import pil_loader
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
from PIL import Image

import skimage.measure as skm
import skimage.io as skio
import skimage.util as sku
import pandas as pd
import sklearn.preprocessing as skp
import matplotlib.pyplot as plt
from glob import glob


# os.environ["MKL_NUM_THREADS"] = "6"
# os.environ["NUMEXPR_NUM_THREADS"] = "6"
# os.environ["OMP_NUM_THREADS"] = "6"
# torch.set_num_threads(6)


# %%
class RandomRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class LIDC_IDRI_EXPL_pseudo(torch.utils.data.Dataset):
    def __init__(self, df, base_path, split, transform_split=None, stats=None, verbose=False, agg=True, transform=None, soft_labels=False):
        assert split in {"train", "val", "test"}
        self.base_path = base_path
        self.split = "val" if split == "test" else split
        self.img_shape = (224, 224) # (32, 32) #(128, 128)
        self.verbose = verbose
        self.soft_labels = soft_labels
        if transform_split is None:
            transform_split = split

        header = [
            'img_id', 
            'gt_subtlety', 'gt_internalStructure', 'gt_calcification', 'gt_sphericity', 'gt_margin', 'gt_lobulation', 'gt_spiculation', 'gt_texture', 'gt_malignancy',
            'pd_subtlety', 'pd_internalStructure', 'pd_calcification', 'pd_sphericity', 'pd_margin', 'pd_lobulation', 'pd_spiculation', 'pd_texture', 'pd_malignancy',
            'conf_subtlety', 'conf_internalStructure', 'conf_calcification', 'conf_sphericity', 'conf_margin', 'conf_lobulation', 'conf_spiculation', 'conf_texture', 'conf_malignancy',
            ]
        self.header = header

        # df = df[header]
        df = df[list(filter(lambda c: '_' in c, df.columns.to_list()))]  # filter out non-feature columns
        df = df[df.conf_malignancy > 0.9]   # filter out low confidence

        if agg:
            # append a nid column
            df['nid'] = df.apply(lambda row: '_'.join(row.img_id.split('_')[:-2] + row.img_id.split('_')[-1:]), axis=1)
            df_median = df.groupby('nid').median()
            for ftr in header[10:]:
                df[ftr] = df['nid'].map(df_median[ftr])
            # df = df.loc[df['malignancy'] != 3]   # nodules of median radiologists' score 3 were removed
            # df_median = df_median.loc[df_median['malignancy'] != 3]   # nodules of median radiologists' score 3 were removed

        # scenes = json.load(fd)["scenes"]
        # scans = pl.query(pl.Scan).filter(
        #     pl.Scan.slice_thickness < 3
        # ) # "CT scans with slice thickness larger than or equal to 3 mm were also excluded"
        if agg:
            self.img_ids, self.img_class_ids, self.img_ftr_ids, self.scenes, self.fnames = \
                self.prepare_scenes(df, df_median)
        else:
            self.img_ids, self.img_class_ids, self.img_ftr_ids, self.scenes, self.fnames = \
                self.prepare_scenes(df)

        if transform is not None:
            self.transform = transform
        else:
            # if split == "train":
            #     transform_list = [
            #         transforms.Resize((32, 32)),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.RandomVerticalFlip(),
            #         # RandomRotation(angles=[0, 90, 180, 270]),
            #         transforms.GaussianBlur(kernel_size=1),
            #     ]
            # else:
            #     transform_list = [
            #         transforms.Resize((32, 32)),
            #     ]

            if transform_split == "train":
                transform_list = [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    RandomRotation(angles=[0, 90, 180, 270]),
                    # transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                    transforms.GaussianBlur(kernel_size=1),
                ]
            else:
                transform_list = [
                    transforms.Resize(256, interpolation=3),
                    transforms.CenterCrop(224),
                ]
                '''
                inverse_modes_mapping = {
                    0: InterpolationMode.NEAREST,
                    2: InterpolationMode.BILINEAR,
                    3: InterpolationMode.BICUBIC,
                    4: InterpolationMode.BOX,
                    5: InterpolationMode.HAMMING,
                    1: InterpolationMode.LANCZOS,
                }
                '''
            transform_list.append(transforms.ToTensor())
            if stats is not None:
                transform_list.append(transforms.Normalize(*stats))
            self.transform = transforms.Compose(transform_list)

        self.n_classes = len(np.unique(self.img_class_ids, axis=0))

    def prepare_scenes(self, df, df_median=None):
        # pids = []   # patient id
        img_ids = []    # nod id
        scenes = []     # nods
        img_class_ids = []
        img_ftr_ids = []
        fnames = []
        df_header = df.columns.values

        for ann in df.itertuples(index=False):
            # print(ann)
            if df_median is not None:
                img_class_id = 1 if ann.gt_malignancy > 0 else 0 
                img_class_id_pseudo = 1 if ann.pd_malignancy > 0 else 0 
            else:
                img_class_id = np.round(ann.gt_malignancy)
                img_class_id_pseudo = np.round(ann.pd_malignancy)
            imgs_per_ann = glob(f"{os.path.join(self.base_path, 'Image', self.split, str(img_class_id), ann.img_id)}*.png")
            fnames += imgs_per_ann
            img_ids += [ann.img_id] * len(imgs_per_ann)
            if self.soft_labels:
                img_class_id_pseudo = [ann.prob_malignancy_0, ann.prob_malignancy_1]
                img_ftr_id = {}
                for ftr in self.header[10:18]:
                    ftr = ftr.replace('pd_', '')
                    img_ftr_id[ftr] = ann[df.columns.get_loc(f'prob_{ftr}_1') : df.columns.get_loc(f'prob_{ftr}_1') + len(list(filter(lambda c: f'prob_{ftr}_' in c, df_header)))]
            else:
                # add feature labels
                img_ftr_id = {df_header[i].replace('pd_', ''):np.round(ann[i]) for i in range(df.columns.get_loc('pd_subtlety'), df.columns.get_loc('pd_subtlety')+8)}


            img_class_ids += [img_class_id_pseudo] * len(imgs_per_ann)
            img_ftr_ids += [img_ftr_id] * len(imgs_per_ann)

        return img_ids, img_class_ids, img_ftr_ids, scenes, fnames


    def __getitem__(self, idx):
        image_id = self.img_ids[idx]

        # image = self.get_image(image_id)
        image = pil_loader(self.fnames[idx])
        # TODO: sofar only dummy
        img_expl = torch.tensor(skio.imread(self.fnames[idx].replace('Image', 'Mask')))

        if self.transform is not None:
            image = self.transform(image) # in range [0., 1.]
            # image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
            # img_expl = self.transform_img_expl(img_expl)
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        # objects = self.scenes[idx]
        # table_expl = self.gt_table_expls[idx]
        img_class_id = self.img_class_ids[idx]
        img_ftr_id = self.img_ftr_ids[idx]

        # remove objects presence indicator from gt table
        # objects = objects[:, 1:]

        return image, img_class_id, img_ftr_id, image_id, img_expl, idx

    def __len__(self):
        return len(self.fnames)

class LIDC_IDRI_EXPL(torch.utils.data.Dataset):
    def __init__(self, base_path, split, transform_split=None, stats=None, verbose=False, agg=True, transform=None):
        assert split in {"train", "val", "test"}
        self.base_path = base_path
        self.split = "val" if split == "test" else split
        self.img_shape = (224, 224) # (32, 32) #(128, 128)
        self.verbose = verbose
        if transform_split is None:
            transform_split = split

        header = ['id', 'scan_id', 'subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']
        self.header = header

        df = pd.read_csv(f'{self.base_path}/meta_{self.split}.csv', index_col=0, converters={'coords': lambda x: list(map(float, x[1:-1].split(',')))})
        # df[header] = df[header].astype(int)
        if agg:
            # append a nid column
            df['nid'] = df.apply(lambda row: '_'.join(row.name.split('_')[:-2] + row.name.split('_')[-1:]), axis=1)
            df_median = df.groupby('nid').median()
            for ftr in header[2:]:
                df[ftr] = df['nid'].map(df_median[ftr])
            # df = df.loc[df['malignancy'] != 3]   # nodules of median radiologists' score 3 were removed
            df_median = df_median.loc[df_median['malignancy'] != 3]   # nodules of median radiologists' score 3 were removed

        # scenes = json.load(fd)["scenes"]
        # scans = pl.query(pl.Scan).filter(
        #     pl.Scan.slice_thickness < 3
        # ) # "CT scans with slice thickness larger than or equal to 3 mm were also excluded"
        if agg:
            self.img_ids, self.img_class_ids, self.img_ftr_ids, self.scenes, self.fnames = \
                self.prepare_scenes(df, df_median)
        else:
            self.img_ids, self.img_class_ids, self.img_ftr_ids, self.scenes, self.fnames = \
                self.prepare_scenes(df)

        if transform is not None:
            self.transform = transform
        else:
            # if split == "train":
            #     transform_list = [
            #         transforms.Resize((32, 32)),
            #         transforms.RandomHorizontalFlip(),
            #         transforms.RandomVerticalFlip(),
            #         # RandomRotation(angles=[0, 90, 180, 270]),
            #         transforms.GaussianBlur(kernel_size=1),
            #     ]
            # else:
            #     transform_list = [
            #         transforms.Resize((32, 32)),
            #     ]

            if transform_split == "train":
                transform_list = [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                    RandomRotation(angles=[0, 90, 180, 270]),
                    # transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
                    transforms.GaussianBlur(kernel_size=1),
                ]
            else:
                transform_list = [
                    transforms.Resize(256, interpolation=3),
                    transforms.CenterCrop(224),
                ]
                '''
                inverse_modes_mapping = {
                    0: InterpolationMode.NEAREST,
                    2: InterpolationMode.BILINEAR,
                    3: InterpolationMode.BICUBIC,
                    4: InterpolationMode.BOX,
                    5: InterpolationMode.HAMMING,
                    1: InterpolationMode.LANCZOS,
                }
                '''
            transform_list.append(transforms.ToTensor())
            if stats is not None:
                transform_list.append(transforms.Normalize(*stats))
            self.transform = transforms.Compose(transform_list)

        self.n_classes = len(np.unique(self.img_class_ids, axis=0))

    def prepare_scenes(self, df, df_median=None):
        # pids = []   # patient id
        img_ids = []    # nod id
        scenes = []     # nods
        img_class_ids = []
        img_ftr_ids = []
        fnames = []

        for ann in df.itertuples():
            # print(ann)
            if df_median is not None:
                img_class_id = 0 if ann.malignancy < 3 else 1  # 0(unlikely):1-2, 1(suspicious):3-5
            else:
                img_class_id = np.round(ann.malignancy) - 1
            imgs_per_ann = glob(f"{os.path.join(self.base_path, 'Image', self.split, str(img_class_id), ann.Index)}*.png")
            fnames += imgs_per_ann
            img_ids += [ann.Index] * len(imgs_per_ann)
            img_class_ids += [img_class_id] * len(imgs_per_ann)


            # add feature labels
            img_ftr_id = {self.header[i-1]:np.round(ann[i]) - 1 for i in range(3, len(self.header))}
            # img_ftr_id = {
            #     "subtlety": 0 if ann.subtlety < 3 else 1,  # 0(unlikely):1-2, 1(suspicious):3-5
            #     "internalStructure": 1 if ann.internalStructure == 1 else 0,
            #     "calcification": 1 if ann.calcification == 4 else 0,
            #     "sphericity": 0 if ann.sphericity > 4 else 1,
            #     "margin": 0 if ann.margin > 4 else 1,
            #     "lobulation": 0 if ann.lobulation < 3 else 1,
            #     "spiculation": 0 if ann.spiculation < 3 else 1,
            #     "texture": 0 if ann.texture > 4 else 1,
            # }
            img_ftr_ids += [img_ftr_id] * len(imgs_per_ann)

        return img_ids, img_class_ids, img_ftr_ids, scenes, fnames


    def __getitem__(self, idx):
        image_id = self.img_ids[idx]

        # image = self.get_image(image_id)
        image = pil_loader(self.fnames[idx])
        # TODO: sofar only dummy
        img_expl = torch.tensor(skio.imread(self.fnames[idx].replace('Image', 'Mask')))

        if self.transform is not None:
            image = self.transform(image) # in range [0., 1.]
            # image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
            # img_expl = self.transform_img_expl(img_expl)
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        # objects = self.scenes[idx]
        # table_expl = self.gt_table_expls[idx]
        img_class_id = self.img_class_ids[idx]
        img_ftr_id = self.img_ftr_ids[idx]

        # remove objects presence indicator from gt table
        # objects = objects[:, 1:]

        return image, img_class_id, img_ftr_id, image_id, img_expl, idx

    def __len__(self):
        return len(self.fnames)

class LIDC_IDRI_FTR(torch.utils.data.Dataset):
    def __init__(self, base_path, split, agg=False):
        assert split in {"train", "val", "test"}
        self.base_path = base_path
        self.split = "val" if split == "test" else split

        header = ['id', 'scan_id', 'subtlety', 'internalStructure', 'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture', 'malignancy']
        dfs = {}
        for folder in os.listdir(base_path):
            dfs[folder] = pd.read_csv(f'{base_path}/{folder}/meta_{folder}.csv', index_col=0, converters={'coords': lambda x: list(map(float, x[1:-1].split(',')))}) 
            dfs[folder][header] = dfs[folder][header].astype(int)
            if agg:
                # append a nid column
                dfs[folder]['nid'] = dfs[folder].apply(lambda row: '_'.join(row.name.split('_')[:-2] + row.name.split('_')[-1:]), axis=1)
                dfs[folder] = dfs[folder].groupby('nid').median()
                dfs[folder] = dfs[folder].loc[dfs[folder]['malignancy'] != 3]   # nodules of median radiologists' score 3 were removed

        # df = pd.read_csv(f'{self.base_path}/{self.split}/meta_{self.split}.csv', index_col=0, converters={'coords': lambda x: list(map(float, x[1:-1].split(',')))})
        # df[header] = df[header].astype(int)

        # df = dfs[self.split]
        if agg:
            for df in dfs.values():
                # df['subtlety'] = np.where(df['subtlety'] < 3, 0, 1)
                # df['internalStructure'] = np.where(df['internalStructure'] == 1, 1, 0)
                # df['calcification'] = np.where(df['calcification'] == 4, 1, 0)
                # df['sphericity'] = np.where(df['sphericity'] > 4, 0, 1)
                # df['margin'] = np.where(df['margin'] > 4, 0, 1)
                # df['lobulation'] = np.where(df['lobulation'] < 3, 0, 1)
                # df['spiculation'] = np.where(df['spiculation'] < 3, 0, 1)
                # df['texture'] = np.where(df['texture'] > 4, 0, 1)
                labels = np.where(dfs[self.split]['malignancy'] < 3, 0, 1)   # 0(unlikely):1-2, 1(suspicious):3-5
                # labels = F.one_hot(torch.as_tensor(labels - 1, dtype=torch.int64), num_classes=5).float()
        else:
            labels = np.round(dfs[self.split]['malignancy']) - 1
        
        scaler = skp.StandardScaler()
        dfs['train'][header[2:-1]] = scaler.fit_transform(dfs['train'][header[2:-1]])
        inputs = dfs[self.split][header[2:-1]].to_numpy()
        if self.split != 'train':
            inputs = scaler.transform(inputs)

        self.inputs = torch.FloatTensor(inputs)
        self.labels = torch.FloatTensor(labels)

        self.n_classes = len(np.unique(labels, axis=0))
        self.ftr_len = self.inputs.size(-1)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)
