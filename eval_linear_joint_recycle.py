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
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import utils
import vision_transformer as vits
import data_LIDC_IDRI as data
from eval_knn_joint import extract_features


def eval_linear(args):
    aggregate_labels = True
    # args.data_path = '../../datasets/LIDC_IDRI/imagenet_2d_ann'
    # args.pretrained_weights = './logs/vits16_pretrain_full_2d_ann/checkpoint.pth'

    stats = ((0.2281477451324463, 0.2281477451324463, 0.2281477451324463), (0.25145936012268066, 0.25145936012268066, 0.25145936012268066))
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
    utils.seed_everything(args.seed)

    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ building network ... ============
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        model = vits.__dict__[args.arch](patch_size=args.patch_size, num_classes=0)
        embed_dim = model.embed_dim * (args.n_last_blocks + int(args.avgpool_patchtokens))
    # if the network is a XCiT
    elif "xcit" in args.arch:
        model = torch.hub.load('facebookresearch/xcit:main', args.arch, num_classes=0)
        embed_dim = model.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        model = torchvision_models.__dict__[args.arch]()
        embed_dim = model.fc.weight.shape[1]
        model.fc = nn.Identity()
    else:
        print(f"Unknow architecture: {args.arch}")
        sys.exit(1)
    model.cuda()
    model.eval()
    # load weights to evaluate
    utils.load_pretrained_weights(model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size)
    print(f"Model {args.arch} built. embed_dim: {embed_dim}")

    # ============ preparing data ... ============
    valset = data.LIDC_IDRI_EXPL(args.data_path, "val", stats=stats, agg=aggregate_labels)
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # compute class counts
    df = pd.DataFrame(valset.img_ftr_ids)
    for fk in ftr_CLASSES.keys():
        print(f"{fk}: {dict(zip(*np.unique(df[fk], return_counts=True)))}")

    # if args.evaluate:
    #     utils.load_pretrained_linear_weights(linear_classifiers_ftr, linear_classifier, args.arch, args.patch_size)
    #     test_stats = validate_network(val_loader, model, linear_classifiers_ftr, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
    #     print(f"Accuracy of the network on the {len(valset)} test images: {test_stats['acc1']:.1f}%")
    #     return

    trainset = data.LIDC_IDRI_EXPL(args.data_path, "train", stats=stats, agg=aggregate_labels)
    len_trainset = len(trainset)
    basic_frac = 0.01
    # # partial annotation
    # if args.independent_anno:
    #     indices = random.choices(range(len(trainset)), k=round(basic_frac * len(trainset)))
    #     unlabelled_indices = list(set(range(len(trainset))) - set(indices))
    # else:
    #     nid_list = np.asarray(list(map(lambda ids: '_'.join(ids.split('_')[:-2] + ids.split('_')[-1:]), trainset.img_ids)))
    #     nids, ind = np.unique(nid_list, return_index=True)
    #     if args.anno_wise:
    #         # use all annotations for each sample (better to use without consistant transform) (get better performance for k=250)
    #         nods_selected = random.sample(list(nids), k=round(args.label_frac * len(nids)))
    #         indices = list(np.concatenate([np.where(nid_list == nod)[0] for nod in nods_selected]))          
    #     else:   
    #         # use only one annotation for each sample (better to use with consistant transform) (get more reasonable best k=20)
    #         nod_indices = list(ind[np.argsort(ind)])
    #         indices = random.sample(nod_indices, k=round(args.label_frac * len(nod_indices)))
    # trainset = torch.utils.data.Subset(trainset, indices)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # print(f"Data loaded with {len(trainset)} train and {len(valset)} val imgs.")

    # ============ extract features ... ============
    print("Extracting features for train set...")
    train_features = extract_features(model, train_loader)
    if utils.get_rank() == 0:
        train_features = nn.functional.normalize(train_features, dim=1, p=2)
    km = KMeans(n_clusters=round(basic_frac * len(trainset)), random_state=args.seed).fit(train_features.cpu())
    indices, _ = pairwise_distances_argmin_min(km.cluster_centers_, train_features.cpu(), metric='cosine')
    unlabelled_indices = list(set(range(len(trainset))) - set(indices))
    print("Seed samples:", end=' ')
    print(*indices, sep=', ')

    trainset = torch.utils.data.Subset(trainset, indices)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(trainset)} train and {len(valset)} val imgs.")

    unlabelledset_as_val = data.LIDC_IDRI_EXPL(args.data_path, "train", transform_split="val", stats=stats, agg=aggregate_labels)
    unlabelledset_as_val = torch.utils.data.Subset(unlabelledset_as_val, unlabelled_indices)
    # sampler_unlabelled = torch.utils.data.distributed.DistributedSampler(unlabelledset_as_val)
    # unlabelled_loader_as_val = torch.utils.data.DataLoader(
    #     unlabelledset_as_val,
    #     sampler=sampler_unlabelled,
    #     batch_size=args.batch_size_per_gpu,
    #     num_workers=args.num_workers,
    #     pin_memory=True,
    # )

    def train_with_seed_samples():
        # compute class weights
        df = pd.DataFrame(trainset.dataset.img_ftr_ids)
        class_weights_ftr = {fk: torch.from_numpy(compute_class_weight('balanced', classes=np.unique(df[fk]), y=df[fk])) for fk in ftr_CLASSES.keys()}
        class_weights_ftr['internalStructure'] = torch.cat((class_weights_ftr['internalStructure'][:1], torch.Tensor([0]), torch.Tensor([0]), class_weights_ftr['internalStructure'][-1:]), 0)
        class_weights_ftr['calcification'] = torch.cat((torch.Tensor([0]), class_weights_ftr['calcification']), 0)
        class_weights_ftr['sphericity'] = torch.cat((torch.Tensor([0]), class_weights_ftr['sphericity']), 0)

        # for fk, v in class_weights_ftr.items():
        #     print(f"{fk}: {v}")

        linear_classifiers_ftr = {}
        embed_dim_catted = embed_dim
        for fk, v in ftr_CLASSES.items():
            linear_classifiers_ftr[fk] = LinearClassifier(embed_dim, num_labels=len(v))
            embed_dim_catted += linear_classifiers_ftr[fk].linear.out_features
            linear_classifiers_ftr[fk] = linear_classifiers_ftr[fk].cuda()
            linear_classifiers_ftr[fk] = nn.parallel.DistributedDataParallel(linear_classifiers_ftr[fk], device_ids=[args.gpu])

        linear_classifier = LinearClassifier(embed_dim_catted, num_labels=args.num_labels)
        linear_classifier = linear_classifier.cuda()
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

        # set optimizer for CLS
        optimizer = torch.optim.SGD(
            linear_classifier.parameters(),
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
            momentum=0.9,
            weight_decay=0, # we do not apply weight decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

        # Optionally resume from a checkpoint for CLS
        to_restore = {"epoch": 0, "best_acc": 0.}
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, f"ckpt_{args.arch}.pth.tar"),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        start_epoch = to_restore["epoch"]
        best_acc = to_restore["best_acc"]

        # set optimizers for FTRs
        optimizers_ftr = {}
        schedulers_ftr = {}
        best_accs_ftr = {}
        for fk in ftr_CLASSES.keys():
            optimizers_ftr[fk] = torch.optim.SGD(
                linear_classifiers_ftr[fk].parameters(),
                args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
                momentum=0.9,
                weight_decay=0, # we do not apply weight decay
            )
            schedulers_ftr[fk] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers_ftr[fk], args.epochs, eta_min=0)

            # Optionally resume from a checkpoint for FTRs
            to_restore = {"epoch": 0}
            to_restore[f"best_acc_{fk}"] = 0.
            utils.restart_from_checkpoint(
                os.path.join(args.output_dir, f"ckpt_{args.arch}_{fk}.pth.tar"),
                run_variables=to_restore,
                state_dict=linear_classifiers_ftr[fk],
                optimizer=optimizers_ftr[fk],
                scheduler=schedulers_ftr[fk],
            )
            start_epoch = to_restore["epoch"]
            best_accs_ftr[fk] = to_restore[f"best_acc_{fk}"]

        for epoch in tqdm(range(start_epoch, args.epochs)):
            train_loader.sampler.set_epoch(epoch)

            train_stats = train(model, linear_classifiers_ftr, linear_classifier, optimizers_ftr, optimizer, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
            for fk in ftr_CLASSES.keys():
                schedulers_ftr[fk].step()
            scheduler.step()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch}
            if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
                test_stats = validate_network(val_loader, model, linear_classifiers_ftr, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
                print(f"Max Accuracy so far at epoch {epoch} of the network on the {len(valset)} test images: {test_stats['acc1']:.3f}%")
                for fk in ftr_CLASSES.keys():
                    # print(f"{test_stats[f'acc1_{fk}']:.1f}% -- {fk}")
                    best_accs_ftr[fk] = max(best_accs_ftr[fk], test_stats[f'acc1_{fk}'])
                    # print(f'{best_accs_ftr[fk]:.3f} -- {fk}')
                best_acc = max(best_acc, test_stats["acc1"])
                # print(f'{best_acc:.3f} -- malignancy')
                msg1 = [fk[:3] for fk in ftr_CLASSES.keys() if fk != 'internalStructure'] + ['malignancy']
                msg2 = [f'{best_accs_ftr[fk]:.3f}' for fk in ftr_CLASSES.keys() if fk != 'internalStructure'] + [f'{best_acc:.3f}']
                print('\t'.join(msg1))
                print('\t'.join(msg2))
                log_stats = {**{k: v for k, v in log_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()}}
            if utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                epoch_save = epoch + 1
                for fk in ftr_CLASSES.keys():
                    if best_accs_ftr[fk] == test_stats[f'acc1_{fk}']:
                        save_dict = {
                            "epoch": epoch_save,
                            "state_dict": linear_classifiers_ftr[fk].state_dict(),
                            "optimizer": optimizers_ftr[fk].state_dict(),
                            "scheduler": schedulers_ftr[fk].state_dict(),
                            f"best_acc_{fk}": best_accs_ftr[fk],
                        }
                        torch.save(save_dict, os.path.join(args.output_dir, f"ckpt_{args.arch}_{fk}_best.pth.tar"))
                    if epoch_save == args.epochs:
                        save_dict = {
                            "epoch": epoch + 1,
                            "state_dict": linear_classifiers_ftr[fk].state_dict(),
                            "optimizer": optimizers_ftr[fk].state_dict(),
                            "scheduler": schedulers_ftr[fk].state_dict(),
                            f"best_acc_{fk}": best_accs_ftr[fk],
                        }
                        torch.save(save_dict, os.path.join(args.output_dir, f"ckpt_{args.arch}_{fk}.pth.tar"))
                if best_acc == test_stats["acc1"]:
                    save_dict = {
                        "epoch": epoch_save,
                        "state_dict": linear_classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_acc": best_acc,
                    }
                    torch.save(save_dict, os.path.join(args.output_dir, f"ckpt_{args.arch}_best.pth.tar"))
                if epoch_save == args.epochs:
                    save_dict = {
                        "epoch": epoch + 1,
                        "state_dict": linear_classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_acc": best_acc,
                    }
                    torch.save(save_dict, os.path.join(args.output_dir, f"ckpt_{args.arch}.pth.tar"))
        print("Training of the supervised linear classifier on frozen features completed.\n"
                    "Top-1 test accuracy: {acc:.3f}".format(acc=best_acc))
        for fk in ftr_CLASSES.keys():
            # print(f'{best_accs_ftr[fk]:.3f}% -- {fk}')
            utils.restart_from_checkpoint(
                os.path.join(args.output_dir, f"ckpt_{args.arch}_{fk}_best.pth.tar"),
                run_variables=to_restore,
                state_dict=linear_classifiers_ftr[fk],
                optimizer=optimizers_ftr[fk],
                scheduler=schedulers_ftr[fk],
            )
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, f"ckpt_{args.arch}_best.pth.tar"),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        test_stats = validate_network(val_loader, model, linear_classifiers_ftr, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)

        # df_unlabelled = write_results(unlabelledset_as_val, model, linear_classifiers_ftr, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
        # df_unlabelled.to_csv(os.path.join(args.output_dir, f"pseudoset_{int(basic_frac*100)}p.csv"))

        df_results = write_results(valset, model, linear_classifiers_ftr, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
        df_results.to_csv(os.path.join(args.output_dir, f'pred_results_{int(basic_frac*100)}p.csv'))
        return linear_classifiers_ftr, linear_classifier

    # exit()
    # ============ training with pseudo labels ... ============
    
    def train_with_unlabelled_samples(gt_sample_ids, df_unlabelled):
        linear_classifiers_ftr = {}
        embed_dim_catted = embed_dim
        for fk, v in ftr_CLASSES.items():
            linear_classifiers_ftr[fk] = LinearClassifier(embed_dim, num_labels=len(v))
            embed_dim_catted += linear_classifiers_ftr[fk].linear.out_features
            linear_classifiers_ftr[fk] = linear_classifiers_ftr[fk].cuda()
            linear_classifiers_ftr[fk] = nn.parallel.DistributedDataParallel(linear_classifiers_ftr[fk], device_ids=[args.gpu])

        linear_classifier = LinearClassifier(embed_dim_catted, num_labels=args.num_labels)
        linear_classifier = linear_classifier.cuda()
        linear_classifier = nn.parallel.DistributedDataParallel(linear_classifier, device_ids=[args.gpu])

        unlabelledset_as_train = data.LIDC_IDRI_EXPL_pseudo(
            df_unlabelled, args.data_path, "train", transform_split="train", 
            num_labels=round(args.label_frac * len_trainset) - round(basic_frac * len_trainset),
            gt_sample_ids=gt_sample_ids,
            stats=stats, agg=aggregate_labels, soft_labels=args.soft_labels)
        gt_sample_ids = unlabelledset_as_train.gt_sample_ids
        # unlabelledset_as_train = torch.utils.data.Subset(unlabelledset_as_train, unlabelled_indices)
        sampler_unlabelled = torch.utils.data.distributed.DistributedSampler(unlabelledset_as_train)
        unlabelled_loader_as_train = torch.utils.data.DataLoader(
            unlabelledset_as_train,
            sampler=sampler_unlabelled,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
        )
        
        print(f"Data loaded with {len(trainset)} + {unlabelledset_as_train.num_labels}(labelled) \
            and {len(unlabelledset_as_train) - unlabelledset_as_train.num_labels}/{len(unlabelledset_as_val)} (unlabelled) train imgs, \
            and {len(valset)} val imgs.")

        # mergedset_as_train = torch.utils.data.ConcatDataset([trainset, unlabelledset_as_train])
        # sampler_merged = torch.utils.data.distributed.DistributedSampler(mergedset_as_train)
        # merged_loader_as_train = torch.utils.data.DataLoader(
        #     mergedset_as_train,
        #     sampler=sampler_merged,
        #     batch_size=args.batch_size_per_gpu,
        #     num_workers=args.num_workers,
        #     pin_memory=True,
        # )

        # reset optimizer for CLS
        optimizer = torch.optim.SGD(
            linear_classifier.parameters(),
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
            momentum=0.9,
            weight_decay=0, # we do not apply weight decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, eta_min=0)

        # Optionally resume from a checkpoint for CLS
        to_restore = {"epoch": 0, "best_acc": 0.}
        # utils.restart_from_checkpoint(
        #     os.path.join(args.output_dir, f"ckpt_{args.arch}.pth.tar"),
        #     run_variables=to_restore,
        #     state_dict=linear_classifier,
        #     optimizer=optimizer,
        #     scheduler=scheduler,
        # )
        start_epoch = to_restore["epoch"]
        best_acc = to_restore["best_acc"]

        # reset optimizers for FTRs
        optimizers_ftr = {}
        schedulers_ftr = {}
        best_accs_ftr = {}
        for fk in ftr_CLASSES.keys():
            optimizers_ftr[fk] = torch.optim.SGD(
                linear_classifiers_ftr[fk].parameters(),
                args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
                momentum=0.9,
                weight_decay=0, # we do not apply weight decay
            )
            schedulers_ftr[fk] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers_ftr[fk], args.epochs, eta_min=0)

            # Optionally resume from a checkpoint for FTRs
            to_restore = {"epoch": 0}
            to_restore[f"best_acc_{fk}"] = 0.
            # utils.restart_from_checkpoint(
            #     os.path.join(args.output_dir, f"ckpt_{args.arch}_{fk}.pth.tar"),
            #     run_variables=to_restore,
            #     state_dict=linear_classifiers_ftr[fk],
            #     optimizer=optimizers_ftr[fk],
            #     scheduler=schedulers_ftr[fk],
            # )
            start_epoch = to_restore["epoch"]
            best_accs_ftr[fk] = to_restore[f"best_acc_{fk}"]

        for epoch in tqdm(range(args.epochs, args.epochs + 10)):
            unlabelled_loader_as_train.sampler.set_epoch(epoch)

            train_stats = train(model, linear_classifiers_ftr, linear_classifier, optimizers_ftr, optimizer, unlabelled_loader_as_train, epoch, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
            for fk in ftr_CLASSES.keys():
                schedulers_ftr[fk].step()
            scheduler.step()

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch}
            if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
                test_stats = validate_network(val_loader, model, linear_classifiers_ftr, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
                print(f"Max Accuracy so far at epoch {epoch} of the network on the {len(valset)} test images: {test_stats['acc1']:.3f}%")
                for fk in ftr_CLASSES.keys():
                    # print(f"{test_stats[f'acc1_{fk}']:.1f}% -- {fk}")
                    best_accs_ftr[fk] = max(best_accs_ftr[fk], test_stats[f'acc1_{fk}'])
                    # print(f'{best_accs_ftr[fk]:.3f} -- {fk}')
                best_acc = max(best_acc, test_stats["acc1"])
                # print(f'{best_acc:.3f} -- malignancy')
                msg1 = [fk[:3] for fk in ftr_CLASSES.keys() if fk != 'internalStructure'] + ['malignancy']
                msg2 = [f'{best_accs_ftr[fk]:.3f}' for fk in ftr_CLASSES.keys() if fk != 'internalStructure'] + [f'{best_acc:.3f}']
                print('\t'.join(msg1))
                print('\t'.join(msg2))
                
                log_stats = {**{k: v for k, v in log_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()}}
            if utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")
                epoch_save = epoch + 1
                for fk in ftr_CLASSES.keys():
                    if best_accs_ftr[fk] == test_stats[f'acc1_{fk}']:
                        save_dict = {
                            "epoch": epoch_save,
                            "state_dict": linear_classifiers_ftr[fk].state_dict(),
                            "optimizer": optimizers_ftr[fk].state_dict(),
                            "scheduler": schedulers_ftr[fk].state_dict(),
                            f"best_acc_{fk}": best_accs_ftr[fk],
                        }
                        torch.save(save_dict, os.path.join(args.output_dir, f"ckpt_{args.arch}_{fk}_best.pth.tar"))
                    if epoch_save == args.epochs:
                        save_dict = {
                            "epoch": epoch + 1,
                            "state_dict": linear_classifiers_ftr[fk].state_dict(),
                            "optimizer": optimizers_ftr[fk].state_dict(),
                            "scheduler": schedulers_ftr[fk].state_dict(),
                            f"best_acc_{fk}": best_accs_ftr[fk],
                        }
                        torch.save(save_dict, os.path.join(args.output_dir, f"ckpt_{args.arch}_{fk}.pth.tar"))
                if best_acc == test_stats["acc1"]:
                    save_dict = {
                        "epoch": epoch_save,
                        "state_dict": linear_classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_acc": best_acc,
                    }
                    torch.save(save_dict, os.path.join(args.output_dir, f"ckpt_{args.arch}_best.pth.tar"))
                if epoch_save == args.epochs:
                    save_dict = {
                        "epoch": epoch + 1,
                        "state_dict": linear_classifier.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_acc": best_acc,
                    }
                    torch.save(save_dict, os.path.join(args.output_dir, f"ckpt_{args.arch}.pth.tar"))
        print("Training of the supervised linear classifier on frozen features completed.\n"
                    "Top-1 test accuracy: {acc:.3f}".format(acc=best_acc))
        for fk in ftr_CLASSES.keys():
            # print(f'{best_accs_ftr[fk]:.3f}% -- {fk}')
            utils.restart_from_checkpoint(
                os.path.join(args.output_dir, f"ckpt_{args.arch}_{fk}_best.pth.tar"),
                run_variables=to_restore,
                state_dict=linear_classifiers_ftr[fk],
                optimizer=optimizers_ftr[fk],
                scheduler=schedulers_ftr[fk],
            )
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, f"ckpt_{args.arch}_best.pth.tar"),
            run_variables=to_restore,
            state_dict=linear_classifier,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        test_stats = validate_network(val_loader, model, linear_classifiers_ftr, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
        # print(f"Best accuracy till epoch {args.epochs} of the network on the {len(valset)} test images: ")
        # for fk in ftr_CLASSES.keys():
        #     print(f"{test_stats[f'acc1_{fk}']:.3f}% -- {fk}")
        
        df_results = write_results(valset, model, linear_classifiers_ftr, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
        df_results.to_csv(os.path.join(args.output_dir, f'pred_results_{int(args.label_frac*100)}p.csv'))
        return linear_classifiers_ftr, linear_classifier, gt_sample_ids
    

    if args.mode == 'seed':
        linear_classifiers_ftr, linear_classifier = train_with_seed_samples()
        df_unlabelled = write_results(unlabelledset_as_val, model, linear_classifiers_ftr, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
        df_unlabelled.to_csv(os.path.join(args.output_dir, f"pseudoset_{int(basic_frac*100)}p.csv"))
    elif args.mode == 'boost':
        gt_sample_ids = None
        df_unlabelled = pd.read_csv(os.path.join(args.output_dir, f"pseudoset_{int(basic_frac*100)}p.csv"))
        for i in range(1, 6):
            print(f"Experiment {i}")
            linear_classifiers_ftr, linear_classifier, gt_sample_ids = train_with_unlabelled_samples(gt_sample_ids, df_unlabelled)
            # print(f"\nData loaded with {len(trainset)} + {n_requested_labels}(labelled) \
            #     and {n_unlabelled - n_requested_labels}/{len(unlabelledset_as_val)} (unlabelled) train imgs, \
            #     and {len(valset)} val imgs.")
            df_unlabelled = write_results(unlabelledset_as_val, model, linear_classifiers_ftr, linear_classifier, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
            df_unlabelled.to_csv(os.path.join(args.output_dir, f"pseudoset_{int(args.label_frac*100)}p.csv"))
            df_unlabelled = pd.read_csv(os.path.join(args.output_dir, f"pseudoset_{int(args.label_frac*100)}p.csv"))
            del linear_classifiers_ftr, linear_classifier
            os.system(f"rm {os.path.join(args.output_dir, '*.tar')}")


def train(model, linear_classifiers_ftr, linear_classifier, optimizers_ftr, optimizer, loader, epoch, n, avgpool, ftr_CLASSES, class_weights=None):
    for fk in ftr_CLASSES.keys():
        linear_classifiers_ftr[fk].train()
    linear_classifier.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    for fk in ftr_CLASSES.keys():
        metric_logger.add_meter(f'lr_{fk}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if class_weights:
        criterion_ftr = {fk:nn.CrossEntropyLoss(weight=class_weights[fk].float().cuda(non_blocking=True)) for fk in ftr_CLASSES.keys()}
    header = 'Epoch: [{}]'.format(epoch)
    for sample in metric_logger.log_every(loader, 20, header):
        # move to gpu
        inp, target, img_ftr_ids, image_id, img_expl, idx = map(lambda x: x.cuda(non_blocking=True) if torch.is_tensor(x) else x, sample)
        if args.soft_labels:
            target = torch.stack(target, dim=-1).cuda(non_blocking=True)
            target_ftrs = {fk:torch.stack(v, dim=-1).cuda(non_blocking=True) for fk, v in img_ftr_ids.items()}
        else:
            target_ftrs = {fk:v.long().cuda(non_blocking=True) for fk, v in img_ftr_ids.items()}

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
                
        # concatenate EMBD and FTRs as input for linear_classifier
        output_catted = output
        for fk in ftr_CLASSES.keys():
            output_f = linear_classifiers_ftr[fk](output)
            output_catted = torch.cat((output_catted, output_f), dim=-1)

            # compute cross entropy loss
            if class_weights:
                loss = criterion_ftr[fk](output_f, target_ftrs[fk])
            else:
                loss = nn.CrossEntropyLoss()(output_f, target_ftrs[fk])

            # compute the gradients
            optimizers_ftr[fk].zero_grad()
            loss.backward(retain_graph=True)

            # step
            optimizers_ftr[fk].step()

            # log 
            torch.cuda.synchronize()
            metric_logger.meters[f'loss_{fk}'].update(loss.item())
            metric_logger.meters[f'lr_{fk}'].update(optimizers_ftr[fk].param_groups[0]["lr"])

        output_cls = linear_classifier(output_catted)
        loss_cls = nn.CrossEntropyLoss()(output_cls, target)
        # compute the gradients
        optimizer.zero_grad()
        loss_cls.backward()

        # step
        optimizer.step()

        # log 
        torch.cuda.synchronize()
        metric_logger.update(loss=loss_cls.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifiers_ftr, linear_classifier, n, avgpool, ftr_CLASSES):
    for fk in ftr_CLASSES.keys():
        linear_classifiers_ftr[fk].eval()
    linear_classifier.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for sample in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        inp, target, img_ftr_ids, image_id, img_expl, idx = map(lambda x: x.cuda(non_blocking=True) if torch.is_tensor(x) else x, sample)
        target_ftrs = {fk:v.long().cuda(non_blocking=True) for fk, v in img_ftr_ids.items()}
        batch_size = inp.shape[0]

        # forward
        with torch.no_grad():
            if "vit" in args.arch:
                intermediate_output = model.get_intermediate_layers(inp, n)
                output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
                if avgpool:
                    output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                    output = output.reshape(output.shape[0], -1)
            else:
                output = model(inp)
                
        # concatenate EMBD and FTRs as input for linear_classifier
        output_catted = output
        for fk in ftr_CLASSES.keys():
            output_f = linear_classifiers_ftr[fk](output)
            output_catted = torch.cat((output_catted, output_f), dim=-1)
            loss = nn.CrossEntropyLoss()(output_f, target_ftrs[fk])

            acc1, = utils.accuracy(output_f, target_ftrs[fk], topk=(1,), near=1)

            metric_logger.meters[f'loss_{fk}'].update(loss.item())
            metric_logger.meters[f'acc1_{fk}'].update(acc1.item(), n=batch_size)
        
        output_cls = linear_classifier(output_catted)
        loss_cls = nn.CrossEntropyLoss()(output_cls, target)

        if linear_classifier.module.num_labels >= 5:
            acc1, acc5 = utils.accuracy(output_cls, target, topk=(1, 5))
        else:
            acc1, = utils.accuracy(output_cls, target, topk=(1,))

        metric_logger.update(loss=loss_cls.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        if linear_classifier.module.num_labels >= 5:
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    for fk in ftr_CLASSES.keys():
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} -- {fk}'
            .format(top1=metric_logger.meters[f'acc1_{fk}'], losses=metric_logger.meters[f'loss_{fk}'], fk=fk[:3]), end=', ')
    if linear_classifier.module.num_labels >= 5:
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))
    else:
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def write_results(valset, model, linear_classifiers_ftr, linear_classifier, n, avgpool, ftr_CLASSES):
    for fk in ftr_CLASSES.keys():
        linear_classifiers_ftr[fk].eval()
    linear_classifier.eval()
    test_loader = torch.utils.data.DataLoader(valset, shuffle=False, batch_size=valset.__len__(), pin_memory=True, num_workers=args.num_workers)
    dataiter = iter(test_loader)
    sample = dataiter.next()
    inp, target, img_ftr_ids, image_id, img_expl, idx = map(lambda x: x.cuda(non_blocking=True) if torch.is_tensor(x) else x, sample)
    target_ftrs = {fk:v.long().cpu() for fk, v in img_ftr_ids.items()}
    with torch.no_grad():
        if "vit" in args.arch:
            intermediate_output = model.get_intermediate_layers(inp, n)
            output = torch.cat([x[:, 0] for x in intermediate_output], dim=-1)
            if avgpool:
                output = torch.cat((output.unsqueeze(-1), torch.mean(intermediate_output[-1][:, 1:], dim=1).unsqueeze(-1)), dim=-1)
                output = output.reshape(output.shape[0], -1)
        else:
            output = model(inp)

    # concatenate EMBD and FTRs as input for linear_classifier
    output_catted = output

    header = [
        'img_id', 
        'gt_subtlety', 'gt_internalStructure', 'gt_calcification', 'gt_sphericity', 'gt_margin', 'gt_lobulation', 'gt_spiculation', 'gt_texture', 'gt_malignancy',
        'pd_subtlety', 'pd_internalStructure', 'pd_calcification', 'pd_sphericity', 'pd_margin', 'pd_lobulation', 'pd_spiculation', 'pd_texture', 'pd_malignancy',
        'conf_subtlety', 'conf_internalStructure', 'conf_calcification', 'conf_sphericity', 'conf_margin', 'conf_lobulation', 'conf_spiculation', 'conf_texture', 'conf_malignancy',
        ]
    df = pd.DataFrame(columns=header)
    df['img_id'] = image_id

    df = pd.concat([df, pd.DataFrame(output.cpu().numpy())], axis=1)

    # predict FTRs
    for fk in ftr_CLASSES.keys():
        output_f = linear_classifiers_ftr[fk](output)
        output_catted = torch.cat((output_catted, output_f), dim=-1)
        prob_f = output_f.softmax(dim=-1)
        conf, pred = torch.max(prob_f.data, 1)
        df[f'gt_{fk}'] = target_ftrs[fk]
        df[f'pd_{fk}'] = pred.cpu()
        df[f'conf_{fk}'] = conf.cpu()
        df = pd.concat([df, pd.DataFrame(prob_f.cpu().numpy(), columns=[f'prob_{fk}_{c}' for c in ftr_CLASSES[fk]])], axis=1)

    # predict CLS
    output_cls = linear_classifier(output_catted)
    prob_cls = output_cls.softmax(dim=-1)
    conf_cls, pred_cls = torch.max(prob_cls.data, 1)
    df['gt_malignancy'] = target.cpu()
    df['pd_malignancy'] = pred_cls.cpu()
    df['conf_malignancy'] = conf_cls.cpu()
    df = pd.concat([df, pd.DataFrame(prob_cls.cpu().numpy(), columns=[f'prob_malignancy_{c}' for c in range(output_cls.size(-1))])], axis=1)
    
    return df


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=2):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)

class Mlp(nn.Module):
    """Multiple FC layers to train on top of frozen features"""
    def __init__(self, dim, num_labels=2, act_layer=nn.ReLU, drop=0.5):
        super().__init__()
        self.num_labels = num_labels
        self.fc1 = nn.Linear(dim, 512)
        self.act1 = act_layer()
        self.fc2 = nn.Linear(512, 256)
        self.act2 = act_layer()
        self.drop = nn.Dropout(drop)
        self.linear = nn.Linear(256, num_labels)

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)
        # fc layers
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop(x)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Evaluation with linear classification on ImageNet')
    parser.add_argument('--n_last_blocks', default=4, type=int, help="""Concatenate [CLS] tokens
        for the `n` last blocks. We use `n=4` when evaluating ViT-Small and `n=1` with ViT-Base.""")
    parser.add_argument('--avgpool_patchtokens', default=False, type=utils.bool_flag,
        help="""Whether ot not to concatenate the global average pooled features to the [CLS] token.
        We typically set this to False for ViT-Small and to True with ViT-Base.""")
    parser.add_argument('--arch', default='vit_small', type=str, help='Architecture')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch resolution of the model.')
    parser.add_argument('--pretrained_weights', default='', type=str, help="Path to pretrained weights to evaluate.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str, help='Key to use in the checkpoint (example: "teacher")')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument("--lr", default=0.001, type=float, help="""Learning rate at the beginning of
        training (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.
        We recommend tweaking the LR depending on the checkpoint evaluated.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int, help='Per-GPU batch-size')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")
    parser.add_argument('--data_path', default='/path/to/imagenet/', type=str)
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument('--val_freq', default=1, type=int, help="Epoch frequency for validation.")
    parser.add_argument('--output_dir', default="./logs", help='Path to save logs and checkpoints')
    parser.add_argument('--num_labels', default=2, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument('--seed', default=42, type=int, help='seed for initialising training. ')
    parser.add_argument("--label_frac", default=1, type=float, help="fraction of labels to use for finetuning")
    parser.add_argument('--independent_anno', default=True, type=utils.bool_flag,
        help="""If treat each annotation as independent when reducing annotation?
        Default setting this to True to follow previous works""")
    parser.add_argument('--anno_wise', default=False, type=utils.bool_flag,
        help="If use all annotations for each sample?")
    parser.add_argument('--soft_labels', default=False, type=utils.bool_flag,
        help="If use soft label for finetuning?")
    parser.add_argument('--mode', choices=['seed', 'boost'], default='seed', type=str, help="seed or boost")
    args = parser.parse_args()
    
    # # for debugging
    # args.pretrained_weights = './logs/vits16_pretrain_full_2d_ann/checkpoint.pth'
    # args.data_path = '../../datasets/LIDC_IDRI/imagenet_2d_ann'
    # args.label_frac = 0.1
    # args.lr = 0.0005
    # args.mode = 'seed'

    eval_linear(args)
