# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import random
import numpy as np

import torch
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torchvision import datasets
from torchvision import transforms as pth_transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
import data_LIDC_IDRI as data

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

def extract_feature_pipeline(args):
    aggregate_labels = True
    stats = ((0.2281477451324463, 0.2281477451324463, 0.2281477451324463), (0.25145936012268066, 0.25145936012268066, 0.25145936012268066))
    utils.seed_everything(42)
    
    # ============ preparing data ... ============
    transform = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize(*stats),
    ]) if not args.anno_wise else None
    dataset_train = data.LIDC_IDRI_EXPL(args.data_path, "train", stats=stats, agg=aggregate_labels, transform=transform)
    # indices = random.choices(range(len(dataset_train)), k=round(1. * len(dataset_train)), weights=None)
    # dataset_train = torch.utils.data.Subset(dataset_train, indices)
    dataset_val = data.LIDC_IDRI_EXPL(args.data_path, "val", stats=stats, agg=aggregate_labels, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset_train, shuffle=False)
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    print(f"Data loaded with {len(dataset_train)} train and {len(dataset_val)} val imgs.")

    # ============ building network ... ============
    if "vit" in args.arch:
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        print(f"Model {args.arch} {args.patch_size}x{args.patch_size} built.")
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch](num_classes=0)
        model.fc = nn.Identity()
    else:
        print(f"Architecture {args.arch} non supported")
        sys.exit(1)
    model.cuda()
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    model.eval()

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, data_loader_train, args.use_cuda)
    print("Extracting features for val set...")
    test_features = extract_features(model, data_loader_val, args.use_cuda)

    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
        test_features = nn.functional.normalize(test_features, dim=1, p=2)

    train_labels = torch.tensor(dataset_train.img_class_ids).long()
    test_labels = torch.tensor(dataset_val.img_class_ids).long()
    train_ftrs_labels = {fk: torch.tensor([dic[fk] for dic in dataset_train.img_ftr_ids]).long() for fk in dataset_train.img_ftr_ids[0].keys()}
    test_ftrs_labels = {fk: torch.tensor([dic[fk] for dic in dataset_val.img_ftr_ids]).long() for fk in dataset_val.img_ftr_ids[0].keys()}

    # partial annotation
    if args.anno_wise:
        # use all annotations for each sample (better to use without consistant transform) (get better performance for k=250)
        nid_list = np.asarray(list(map(lambda ids: '_'.join(ids.split('_')[:-2] + ids.split('_')[-1:]), dataset_train.img_ids)))
        nids, ind, annocounts = np.unique(nid_list, return_index=True, return_counts=True)
        nods_selected = random.sample(list(nids), k=round(args.label_frac * len(nids)))
        indices = list(np.concatenate([np.where(nid_list == nod)[0] for nod in nods_selected]))          
    else:   
        # use only one annotation for each sample (better to use with consistant transform) (get more reasonable best k=20)
        nid_list = np.asarray(list(map(lambda ids: '_'.join(ids.split('_')[:-2] + ids.split('_')[-1:]), dataset_train.img_ids)))
        nids, ind = np.unique(nid_list, return_index=True)
        nod_indices = list(ind[np.argsort(ind)])
        indices = random.sample(nod_indices, k=round(args.label_frac * len(nod_indices)))

    train_features = train_features[indices]
    train_labels = train_labels[indices]
    train_ftrs_labels = {fk: v[indices] for fk, v in train_ftrs_labels.items()}

    # save features and labels
    if args.dump_features and dist.get_rank() == 0:
        torch.save(train_features.cpu(), os.path.join(args.dump_features, "trainfeat.pth"))
        torch.save(test_features.cpu(), os.path.join(args.dump_features, "testfeat.pth"))
        torch.save(train_labels.cpu(), os.path.join(args.dump_features, "trainlabels.pth"))
        torch.save(test_labels.cpu(), os.path.join(args.dump_features, "testlabels.pth"))
    return train_features, train_labels, train_ftrs_labels, test_features, test_labels, test_ftrs_labels


@torch.no_grad()
def extract_features(model, data_loader, use_cuda=True, multiscale=False):
    metric_logger = utils.MetricLogger(delimiter="  ")
    features = None
    for sample in metric_logger.log_every(data_loader, 10):
        samples, target, img_ftr_ids, image_id, img_expl, index = map(lambda x: x.cuda(non_blocking=True) if torch.is_tensor(x) else x, sample)
        if multiscale:
            feats = utils.multi_scale(samples, model)
        else:
            feats = model(samples).clone()

        # init storage feature matrix
        if dist.get_rank() == 0 and features is None:
            features = torch.zeros(len(data_loader.dataset), feats.shape[-1])
            if use_cuda:
                features = features.cuda(non_blocking=True)
            print(f"Storing features into tensor of shape {features.shape}")

        # get indexes from all processes
        y_all = torch.empty(dist.get_world_size(), index.size(0), dtype=index.dtype, device=index.device)
        y_l = list(y_all.unbind(0))
        y_all_reduce = torch.distributed.all_gather(y_l, index, async_op=True)
        y_all_reduce.wait()
        index_all = torch.cat(y_l)

        # share features between processes
        feats_all = torch.empty(
            dist.get_world_size(),
            feats.size(0),
            feats.size(1),
            dtype=feats.dtype,
            device=feats.device,
        )
        output_l = list(feats_all.unbind(0))
        output_all_reduce = torch.distributed.all_gather(output_l, feats, async_op=True)
        output_all_reduce.wait()

        # update storage feature matrix
        if dist.get_rank() == 0:
            if use_cuda:
                features.index_copy_(0, index_all, torch.cat(output_l))
            else:
                features.index_copy_(0, index_all.cpu(), torch.cat(output_l).cpu())
    return features


@torch.no_grad()
def knn_classifier(train_features, train_labels, train_ftrs_labels: dict, test_features, test_labels, test_ftrs_labels: dict, k, T, num_classes=1000):
    top1, top5, total = 0.0, 0.0, 0
    top1_ftrs, top5_ftrs, total_ftrs = {}, {}, {}
    for fk in train_ftrs_labels.keys():
        top1_ftrs[fk], top5_ftrs[fk], total_ftrs[fk] = 0.0, 0.0, 0

    train_features = train_features.t()
    num_test_images, num_chunks = test_labels.shape[0], 10
    imgs_per_chunk = num_test_images // num_chunks
    retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)
    for idx in range(0, num_test_images, imgs_per_chunk):
        # get the cls features for test images
        features = test_features[
            idx : min((idx + imgs_per_chunk), num_test_images), :
        ]
        targets = test_labels[idx : min((idx + imgs_per_chunk), num_test_images)]
        batch_size = targets.shape[0]

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)

        for fk in train_ftrs_labels.keys():
            num_ftr_classes = len(ftr_CLASSES[fk])
            retrieval_one_hot_ftr = torch.zeros(k, num_ftr_classes).to(train_features.device)
            targets_ftr = test_ftrs_labels[fk][idx : min((idx + imgs_per_chunk), num_test_images)]

            candidates_ftr = train_ftrs_labels[fk].view(1, -1).expand(batch_size, -1)
            retrieved_neighbors_ftr = torch.gather(candidates_ftr, 1, indices)

            retrieval_one_hot_ftr.resize_(batch_size * k, num_ftr_classes).zero_()
            retrieval_one_hot_ftr.scatter_(1, retrieved_neighbors_ftr.view(-1, 1), 1)
            distances_transform_ftr = distances.clone().div_(T).exp_()
            probs_ftr = torch.sum(
                torch.mul(
                    retrieval_one_hot_ftr.view(batch_size, -1, num_ftr_classes),
                    distances_transform_ftr.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions_ftr = probs_ftr.sort(1, True)

            # find the predictions_ftr that match the target
            correct_ftr = abs(predictions_ftr - targets_ftr.data.view(-1, 1)) <= 1
            top1_ftrs[fk] = top1_ftrs[fk] + correct_ftr.narrow(1, 0, 1).sum().item()
            if num_ftr_classes > 5:
                top5_ftrs[fk] = top5_ftrs[fk] + correct_ftr.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
            total_ftrs[fk] += targets_ftr.size(0)

        # calculate the dot product and compute top-k neighbors
        similarity = torch.mm(features, train_features)
        distances, indices = similarity.topk(k, largest=True, sorted=True)
        candidates = train_labels.view(1, -1).expand(batch_size, -1)
        retrieved_neighbors = torch.gather(candidates, 1, indices)

        retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
        retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)
        distances_transform = distances.clone().div_(T).exp_()
        probs = torch.sum(
            torch.mul(
                retrieval_one_hot.view(batch_size, -1, num_classes),
                distances_transform.view(batch_size, -1, 1),
            ),
            1,
        )
        _, predictions = probs.sort(1, True)

        # find the predictions that match the target
        correct = predictions.eq(targets.data.view(-1, 1))
        top1 = top1 + correct.narrow(1, 0, 1).sum().item()
        if num_classes > 5:
            top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()  # top5 does not make sense if k < 5
        total += targets.size(0)
    top1 = top1 * 100.0 / total
    top5 = top5 * 100.0 / total
    for fk in train_ftrs_labels.keys():
        top1_ftrs[fk] = top1_ftrs[fk] * 100.0 / total_ftrs[fk]
        top5_ftrs[fk] = top5_ftrs[fk] * 100.0 / total_ftrs[fk]
    return top1, top5, top1_ftrs, top5_ftrs


class ReturnIndexDataset(datasets.ImageFolder):
    def __getitem__(self, idx):
        img, lab = super(ReturnIndexDataset, self).__getitem__(idx)
        return img, idx


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with weighted k-NN on ImageNet')
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument('--nb_knn', default=[10, 20, 50, 100, 150, 200, 250, 300, 350, 400], nargs='+', type=int,
        help='Number of NN to use. 20 is usually working the best.')
    parser.add_argument('--temperature', default=0.1, type=float,
        help='Temperature used in the voting coefficient')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument('--use_cuda', default=True, type=utils.bool_flag,
        help="Should we store the features on GPU? We recommend setting this to False if you encounter OOM")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument("--checkpoint_key", default="teacher", type=str,
        help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--dump_features', default=None,
        help='Path where to save computed features, empty for no saving')
    parser.add_argument('--load_features', default=None, help="""If the features have
        already been computed, where to find them.""")
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument("--label_frac", default=1, type=float, help="fraction of labels to use for finetuning")
    parser.add_argument('--anno_wise', default=False, type=utils.bool_flag,
        help="If treat each annotation as independent when reducing annotation?")
    args = parser.parse_args()

    # for debugging
    # args.pretrained_weights = './logs/vits16_pretrain_full_2d_ann/checkpoint.pth'
    # args.data_path = '../../datasets/LIDC_IDRI/imagenet_2d_ann'
    # args.anno_wise = False
    # args.label_frac = 0.1

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    if args.load_features:
        train_features = torch.load(os.path.join(args.load_features, "trainfeat.pth"))
        test_features = torch.load(os.path.join(args.load_features, "testfeat.pth"))
        train_labels = torch.load(os.path.join(args.load_features, "trainlabels.pth"))
        test_labels = torch.load(os.path.join(args.load_features, "testlabels.pth"))
    else:
        # need to extract features !
        train_features, train_labels, train_ftrs_labels, test_features, test_labels, test_ftrs_labels = extract_feature_pipeline(args)

    if utils.get_rank() == 0:
        if args.use_cuda:
            train_features = train_features.cuda()
            test_features = test_features.cuda()
            train_labels = train_labels.cuda()
            test_labels = test_labels.cuda()
            train_ftrs_labels = {fk: v.cuda() for fk, v in train_ftrs_labels.items()}
            test_ftrs_labels = {fk: v.cuda() for fk, v in test_ftrs_labels.items()}

        print("Features are ready!\nStart the k-NN classification.")
        for k in args.nb_knn:
            top1, top5, top1_ftrs, top5_ftrs = knn_classifier(train_features, train_labels, train_ftrs_labels, 
                test_features, test_labels, test_ftrs_labels,
                k, args.temperature, num_classes=2)
            print(f"\n{k}-NN classifier result: Top1: {top1:.3f}, Top5: {top5:.3f}")
            for fk in train_ftrs_labels.keys():
                print(f'* Acc@1 {top1_ftrs[fk]:.3f} -- {fk}')
    dist.barrier()
