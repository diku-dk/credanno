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



# os.environ["MKL_NUM_THREADS"] = "6"
# os.environ["NUMEXPR_NUM_THREADS"] = "6"
# os.environ["OMP_NUM_THREADS"] = "6"
# torch.set_num_threads(6)

CLASSES = {
    "subtlety": [1, 2, 3, 4, 5],
    "internalStructure": [1, 2, 3, 4],
    "calcification": [1, 2, 3, 4, 5, 6],
    "sphericity": [1, 2, 3, 4, 5],
    "margin": [1, 2, 3, 4, 5],
    "lobulation": [1, 2, 3, 4, 5],
    "spiculation": [1, 2, 3, 4, 5],
    "texture": [1, 2, 3, 4, 5],
    # "malignancy": [1, 2, 3, 4, 5],
}

# %%
class RandomRotation:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return transforms.functional.rotate(x, angle)


class LIDC_IDRI_EXPL(torch.utils.data.Dataset):
    def __init__(self, base_path, split, stats=None, verbose=False, agg=True):
        assert split in {"train", "val", "test"}
        self.base_path = base_path
        self.split = "val" if split == "test" else split
        self.max_objects = 2
        self.img_shape = (224, 224) # (32, 32) #(128, 128)
        self.verbose = verbose

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
            self.img_ids, self.img_class_ids, self.img_ftr_ids, self.scenes, self.fnames, self.gt_img_expls, self.gt_table_expls = \
                self.prepare_scenes(df, df_median)
        else:
            self.img_ids, self.img_class_ids, self.img_ftr_ids, self.scenes, self.fnames, self.gt_img_expls, self.gt_table_expls = \
                self.prepare_scenes(df)

        # transform_list = [
        #     transforms.Resize(self.img_shape),
        #     ]
        # if split == "train" and stats is not None:
        #     transform_list += [
        #         transforms.RandomHorizontalFlip(), 
        #         transforms.RandomVerticalFlip(),
        #         RandomRotation(angles=[0, 90, 180, 270]),
        #         transforms.GaussianBlur(kernel_size=1),
        #         ]
        if split == "train":
            transform_list = [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
            ]
        else:
            transform_list = [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(224),
            ]
        transform_list.append(transforms.ToTensor())
        if stats is not None:
            transform_list.append(transforms.Normalize(*stats))
        self.transform = transforms.Compose(transform_list)

        self.n_classes = len(np.unique(self.img_class_ids, axis=0))
        self.category_dict = CLASSES

        # get ids of category ranges, i.e. shape has three categories from ids 0 to 2
        category_ids = [3]
        for v in CLASSES.values():
            category_ids.append(category_ids[-1] + len(v))
        self.category_ids = np.array(category_ids)

    def object_to_fv(self, nod_ann):
        # return a feature vector
        coords = nod_ann.coords
        subtlety = F.one_hot(torch.as_tensor(np.round(nod_ann.subtlety) - 1, dtype=torch.int64), num_classes=5)
        internalStructure = F.one_hot(torch.as_tensor(np.round(nod_ann.internalStructure) - 1, dtype=torch.int64), num_classes=4)
        calcification = F.one_hot(torch.as_tensor(np.round(nod_ann.calcification) - 1, dtype=torch.int64), num_classes=6)
        sphericity = F.one_hot(torch.as_tensor(np.round(nod_ann.sphericity) - 1, dtype=torch.int64), num_classes=5)
        margin = F.one_hot(torch.as_tensor(np.round(nod_ann.margin) - 1, dtype=torch.int64), num_classes=5)
        lobulation = F.one_hot(torch.as_tensor(np.round(nod_ann.lobulation) - 1, dtype=torch.int64), num_classes=5)
        spiculation = F.one_hot(torch.as_tensor(np.round(nod_ann.spiculation) - 1, dtype=torch.int64), num_classes=5)
        texture = F.one_hot(torch.as_tensor(np.round(nod_ann.texture) - 1, dtype=torch.int64), num_classes=5)
        # concatenate all the classes
        # return a list, len() = 3 + 5 + 4 + 6 + 5 + 5 + 5 + 5 + 5 == 43
        fv = torch.cat((torch.as_tensor(coords), subtlety, internalStructure,
                       calcification, sphericity, margin, lobulation, spiculation, texture), dim=0)
        return fv

    def prepare_scenes(self, df, df_median=None):
        # pids = []   # patient id
        img_ids = []    # nod id
        scenes = []     # nods
        gt_img_expls = []
        img_class_ids = []
        img_ftr_ids = []
        gt_table_expls = []
        fnames = []

        for ann in df.itertuples():
            # print(ann)
            img_ids.append(ann.Index)
            if df_median is not None:
                img_class_id = 0 if ann.malignancy < 3 else 1  # 0(unlikely):1-2, 1(suspicious):3-5
            else:
                img_class_id = np.round(ann.malignancy) - 1
            img_class_ids.append(img_class_id)
            fnames.append(f"{os.path.join(self.base_path, 'Image', self.split, str(img_class_id), ann.Index)}.png")

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
            img_ftr_ids.append(img_ftr_id)

            # # only 1 object per sample
            # # ann = df_median.loc[ann.nid] if df_median is not None else ann
            # objects = [list(self.object_to_fv(ann))]
            # objects = torch.FloatTensor(objects).transpose(0, 1)
            # # objects.shape = torch.Size([43, 1])

            # num_objects = objects.size(1) # =1
            # # pad with 0s
            # if num_objects < self.max_objects:
            #     objects = torch.cat(
            #         [
            #             objects,
            #             torch.zeros(objects.size(0), self.max_objects - num_objects),
            #         ],
            #         dim=1,
            #     )

            # #TODO: get gt table explanation based on the classification rule of the class label
            # gt_table_expl_mask = self.get_table_expl_mask(objects, ann.malignancy)
            # # gt_table_expl_mask.shape = torch.Size([10, 43])
            # gt_table_expls.append(gt_table_expl_mask)

            # # fill in masks
            # mask = torch.zeros(self.max_objects)
            # mask[:num_objects] = 1

            # # concatenate obj indication to end of object list
            # objects = torch.cat((mask.unsqueeze(dim=0), objects), dim=0)
            # # objects.shape = torch.Size([43+1, 10])
            # scenes.append(objects.T)

            # get gt image explanation based on the classification rule of the class label
            # gt_img_expl_mask.shape = torch.Size([128, 128])
            gt_img_expl_mask = torch.tensor(skio.imread(f"{os.path.join(self.base_path, 'Mask', self.split, str(img_class_id), ann.Index)}.png"))
            gt_img_expls.append(gt_img_expl_mask)

        return img_ids, img_class_ids, img_ftr_ids, scenes, fnames, gt_img_expls, gt_table_expls


    def __getitem__(self, idx):
        image_id = self.img_ids[idx]

        # image = self.get_image(image_id)
        image = pil_loader(self.fnames[idx])
        # TODO: sofar only dummy
        img_expl = self.gt_img_expls[idx]

        if self.transform is not None:
            image = self.transform(image) # in range [0., 1.]
            image = (image - 0.5) * 2.0  # Rescale to [-1, 1].
            # img_expl = self.transform_img_expl(img_expl)
        if image.size(0) == 1:
            image = image.repeat(3, 1, 1)

        # objects = self.scenes[idx]
        # table_expl = self.gt_table_expls[idx]
        img_class_id = self.img_class_ids[idx]
        img_ftr_id = self.img_ftr_ids[idx]

        # remove objects presence indicator from gt table
        # objects = objects[:, 1:]

        return image, img_class_id, img_ftr_id, image_id, img_expl

    def __len__(self):
        return len(self.img_ids)

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
