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

import utils
import vision_transformer as vits
import data_LIDC_IDRI as data


def eval_linear(args):
    aggregate_labels = True
    # args.data_path = '../../datasets/LIDC_IDRI/imagenet_2d_ann'
    # args.pretrained_weights = './logs/vits16_pretrain_full_2d_ann/checkpoint.pth'

    # stats = ((-0.5186440944671631, -0.5186440944671631, -0.5186440944671631), (0.511863112449646, 0.511863112449646, 0.511863112449646))
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
    utils.seed_everything(42)

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

    linear_classifiers = {}
    for fk, v in ftr_CLASSES.items():
        linear_classifiers[fk] = LinearClassifier(embed_dim, num_labels=len(v))
        linear_classifiers[fk] = linear_classifiers[fk].cuda()
        linear_classifiers[fk] = nn.parallel.DistributedDataParallel(linear_classifiers[fk], device_ids=[args.gpu])

    # ============ preparing data ... ============
    valset = data.LIDC_IDRI_EXPL(args.data_path, "val", stats=stats, agg=aggregate_labels)
    # val_transform = pth_transforms.Compose([
    #     pth_transforms.Resize(256, interpolation=3),
    #     pth_transforms.CenterCrop(224),
    #     pth_transforms.ToTensor(),
    #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     pth_transforms.Normalize(*stats),
    # ])
    # dataset_val = datasets.ImageFolder(os.path.join(args.data_path, "val"), transform=val_transform)
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
    #     utils.load_pretrained_linear_weights(linear_classifiers, args.arch, args.patch_size)
    #     test_stats = validate_network(val_loader, model, linear_classifiers, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
    #     print(f"Accuracy of the network on the {len(valset)} test images: {test_stats['acc1']:.1f}%")
    #     return

    trainset = data.LIDC_IDRI_EXPL(args.data_path, "train", stats=stats, agg=aggregate_labels)
    indices = random.choices(range(len(trainset)), k=round(args.label_frac * len(trainset)), weights=None)
    trainset = torch.utils.data.Subset(trainset, indices)
    # train_transform = pth_transforms.Compose([
    #     pth_transforms.RandomResizedCrop(224),
    #     pth_transforms.RandomHorizontalFlip(),
    #     pth_transforms.ToTensor(),
    #     # pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    #     pth_transforms.Normalize(*stats),
    # ])
    # dataset_train = datasets.ImageFolder(os.path.join(args.data_path, "train"), transform=train_transform)
    sampler = torch.utils.data.distributed.DistributedSampler(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    print(f"Data loaded with {len(trainset)} train and {len(valset)} val imgs.")

    # compute class weights
    df = pd.DataFrame(trainset.dataset.img_ftr_ids)
    class_weights_ftr = {fk: torch.from_numpy(compute_class_weight('balanced', classes=np.unique(df[fk]), y=df[fk])) for fk in ftr_CLASSES.keys()}
    class_weights_ftr['internalStructure'] = torch.cat((class_weights_ftr['internalStructure'][:1], torch.Tensor([0]), torch.Tensor([0]), class_weights_ftr['internalStructure'][-1:]), 0)
    class_weights_ftr['calcification'] = torch.cat((torch.Tensor([0]), class_weights_ftr['calcification']), 0)
    class_weights_ftr['sphericity'] = torch.cat((torch.Tensor([0]), class_weights_ftr['sphericity']), 0)

    # for fk, v in class_weights_ftr.items():
    #     print(f"{fk}: {v}")

    # set optimizer
    optimizers = {}
    schedulers = {}
    best_accs = {}
    for fk in ftr_CLASSES.keys():
        optimizers[fk] = torch.optim.SGD(
            linear_classifiers[fk].parameters(),
            args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256., # linear scaling rule
            momentum=0.9,
            weight_decay=0, # we do not apply weight decay
        )
        schedulers[fk] = torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[fk], args.epochs, eta_min=0)

        # Optionally resume from a checkpoint
        to_restore = {"epoch": 0}
        to_restore[f"best_acc_{fk}"] = 0.
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, f"ckpt_{fk}.pth.tar"),
            run_variables=to_restore,
            state_dict=linear_classifiers[fk],
            optimizer=optimizers[fk],
            scheduler=schedulers[fk],
        )
        start_epoch = to_restore["epoch"]
        best_accs[fk] = to_restore[f"best_acc_{fk}"]

    for epoch in tqdm(range(start_epoch, args.epochs)):
        train_loader.sampler.set_epoch(epoch)

        train_stats = train(model, linear_classifiers, optimizers, train_loader, epoch, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
        for fk in ftr_CLASSES.keys():
            schedulers[fk].step()

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            test_stats = validate_network(val_loader, model, linear_classifiers, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
            print(f"Max Accuracy so far at epoch {epoch} of the network on the {len(valset)} test images: ")
            for fk in ftr_CLASSES.keys():
                # print(f"{test_stats[f'acc1_{fk}']:.1f}% -- {fk}")
                best_accs[fk] = max(best_accs[fk], test_stats[f'acc1_{fk}'])
                print(f'{best_accs[fk]:.3f}% -- {fk}')
            log_stats = {**{k: v for k, v in log_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()}}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            epoch_save = epoch + 1
            for fk in ftr_CLASSES.keys():
                if best_accs[fk] == test_stats[f'acc1_{fk}']:
                    save_dict = {
                        "epoch": epoch_save,
                        "state_dict": linear_classifiers[fk].state_dict(),
                        "optimizer": optimizers[fk].state_dict(),
                        "scheduler": schedulers[fk].state_dict(),
                        f"best_acc_{fk}": best_accs[fk],
                    }
                    torch.save(save_dict, os.path.join(args.output_dir, f"ckpt_{fk}_best.pth.tar"))
                if epoch_save == args.epochs:
                    save_dict = {
                        "epoch": epoch + 1,
                        "state_dict": linear_classifiers[fk].state_dict(),
                        "optimizer": optimizers[fk].state_dict(),
                        "scheduler": schedulers[fk].state_dict(),
                        f"best_acc_{fk}": best_accs[fk],
                    }
                    torch.save(save_dict, os.path.join(args.output_dir, f"ckpt_{fk}.pth.tar"))
    print("Training of the supervised linear classifier on frozen features completed.\n"
                "Top-1 test accuracy: ")
    for fk in ftr_CLASSES.keys():
        # print(f'{best_accs[fk]:.3f}% -- {fk}')
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, f"ckpt_{fk}_best.pth.tar"),
            run_variables=to_restore,
            state_dict=linear_classifiers[fk],
            optimizer=optimizers[fk],
            scheduler=schedulers[fk],
        )
    test_stats = validate_network(val_loader, model, linear_classifiers, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
    # print(f"Best accuracy till epoch {args.epochs} of the network on the {len(valset)} test images: ")
    # for fk in ftr_CLASSES.keys():
    #     print(f"{test_stats[f'acc1_{fk}']:.3f}% -- {fk}")
    
    df_results = write_results(valset, model, linear_classifiers, args.n_last_blocks, args.avgpool_patchtokens, ftr_CLASSES)
    df_results.to_csv(os.path.join(args.output_dir, 'pred_resultsFTR.csv'))


def train(model, linear_classifiers, optimizers, loader, epoch, n, avgpool, ftr_CLASSES, class_weights=None):
    for fk in ftr_CLASSES.keys():
        linear_classifiers[fk].train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    for fk in ftr_CLASSES.keys():
        metric_logger.add_meter(f'lr_{fk}', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    if class_weights:
        criterion_ftr = {fk:nn.CrossEntropyLoss(weight=class_weights[fk].float().cuda(non_blocking=True)) for fk in ftr_CLASSES.keys()}
    header = 'Epoch: [{}]'.format(epoch)
    for sample in metric_logger.log_every(loader, 20, header):
        # move to gpu
        # inp = inp.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        inp, target, img_ftr_ids, image_id, img_expl = map(lambda x: x.cuda(non_blocking=True) if torch.is_tensor(x) else x, sample)
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
                
        for fk in ftr_CLASSES.keys():
            output_f = linear_classifiers[fk](output)

            # compute cross entropy loss
            if class_weights:
                loss = criterion_ftr[fk](output_f, target_ftrs[fk])
            else:
                loss = nn.CrossEntropyLoss()(output_f, target_ftrs[fk])

            # compute the gradients
            optimizers[fk].zero_grad()
            loss.backward()

            # step
            optimizers[fk].step()

            # log 
            torch.cuda.synchronize()
            # metric_logger.update(loss_f=loss.item())
            metric_logger.meters[f'loss_{fk}'].update(loss.item())
            # metric_logger.update(lr_f=optimizers[fk].param_groups[0]["lr"])
            metric_logger.meters[f'lr_{fk}'].update(optimizers[fk].param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def validate_network(val_loader, model, linear_classifiers, n, avgpool, ftr_CLASSES):
    for fk in ftr_CLASSES.keys():
        linear_classifiers[fk].eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    for sample in metric_logger.log_every(val_loader, 20, header):
        # move to gpu
        # inp = inp.cuda(non_blocking=True)
        # target = target.cuda(non_blocking=True)
        inp, target, img_ftr_ids, image_id, img_expl = map(lambda x: x.cuda(non_blocking=True) if torch.is_tensor(x) else x, sample)
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
                
        for fk in ftr_CLASSES.keys():
            output_f = linear_classifiers[fk](output)
            loss = nn.CrossEntropyLoss()(output_f, target_ftrs[fk])

            acc1, = utils.accuracy(output_f, target_ftrs[fk], topk=(1,), near=1)

            batch_size = inp.shape[0]
            # metric_logger.update(loss_f=loss.item())
            metric_logger.meters[f'loss_{fk}'].update(loss.item())
            metric_logger.meters[f'acc1_{fk}'].update(acc1.item(), n=batch_size)
    for fk in ftr_CLASSES.keys():
        print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f} -- {fk}'
            .format(top1=metric_logger.meters[f'acc1_{fk}'], losses=metric_logger.meters[f'loss_{fk}'], fk=fk))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def write_results(valset, model, linear_classifiers, n, avgpool, ftr_CLASSES):
    for fk in ftr_CLASSES.keys():
        linear_classifiers[fk].eval()
    test_loader = torch.utils.data.DataLoader(valset, shuffle=False, batch_size=valset.__len__(), pin_memory=True, num_workers=args.num_workers)
    dataiter = iter(test_loader)
    sample = dataiter.next()
    inp, target, img_ftr_ids, image_id, img_expl = map(lambda x: x.cuda(non_blocking=True) if torch.is_tensor(x) else x, sample)
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
    
    header = [
        'img_id', 
        'gt_subtlety', 'gt_internalStructure', 'gt_calcification', 'gt_sphericity', 'gt_margin', 'gt_lobulation', 'gt_spiculation', 'gt_texture', 'gt_malignancy',
        'pd_subtlety', 'pd_internalStructure', 'pd_calcification', 'pd_sphericity', 'pd_margin', 'pd_lobulation', 'pd_spiculation', 'pd_texture', 'pd_malignancy',
        ]
    df = pd.DataFrame(columns=header)
    df['img_id'] = image_id

    for fk in ftr_CLASSES.keys():
        output_f = linear_classifiers[fk](output)
        _, pred = torch.max(output_f.data, 1)
        df[f'gt_{fk}'] = target_ftrs[fk]
        df[f'pd_{fk}'] = pred.cpu()
    
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
    # parser.add_argument('--num_labels', default=2, type=int, help='Number of labels for linear classifier')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument("--label_frac", default=1, type=float, help="fraction of labels to use for finetuning")
    args = parser.parse_args()
    eval_linear(args)
