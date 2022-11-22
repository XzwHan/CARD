#!/usr/bin/env python
# Built by Huangjie Zheng
# Modified from Linear probe file in SimSiam: https://github.com/facebookresearch/simsiam/blob/main/main_lincls.py 
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
from pathlib import Path

import card.builder
import card.diffusion_utils
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=4096, type=int,
                    metavar='N',
                    help='mini-batch size (default: 4096), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=0., type=float,
                    metavar='W', help='weight decay (default: 0.)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training.')
parser.add_argument('--num_classes', default=100, type=int,
                    help='number of classes of the linear head.')
parser.add_argument('--num_timesteps', default=10, type=int,
                    help='number of diffusion steps.')
parser.add_argument('--beta_1', default=0.01, type=float,
                    help='beta_1 of the linear variance schedule for the diffusion process.')
parser.add_argument('--beta_T', default=0.95, type=float,
                    help='beta_T of the linear variance schedule for the diffusion process.')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')

# additional configs:
parser.add_argument('--pretrained', default='', type=str,
                    help='path to pretrained checkpoint')
parser.add_argument('--lars', action='store_true',
                    help='Use LARS')

best_acc1 = 0


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass

        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    # create model
    print("=> creating CARD model with backbone: '{}'".format(args.arch))
    model = card.builder.CARD(
        models.__dict__[args.arch],
        args.num_classes, args.num_timesteps)

    print(model)
    print("=> CARD model parameter size: '{}'".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))

    # freeze all layers but the unet
    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.model_theta.named_parameters():
        param.requires_grad = True

    # load from pre-trained, before DistributedDataParallel constructor
    if args.pretrained:
        if os.path.isfile(args.pretrained):
            print("=> loading checkpoint '{}'".format(args.pretrained))
            checkpoint = torch.load(args.pretrained, map_location="cpu")

            # rename pre-trained keys
            state_dict = checkpoint['state_dict']
            state_dict_encoder = {}
            state_dict_linear = {}
            for k in list(state_dict.keys()):
                # retain only encoder up to before the embedding layer
                if not k.startswith('module.fc'):
                    # remove prefix
                    state_dict_encoder[k[len("module."):]] = state_dict[k]
                else:
                    state_dict_linear[k[len("module.fc."):]] = state_dict[k]

            args.start_epoch = 0
            msg = model.model_phi.load_state_dict(state_dict_encoder)
            print(msg)
            msg = model.linear.load_state_dict(state_dict_linear)
            print(msg)

            print("=> loaded pre-trained model '{}'".format(args.pretrained))
        else:
            print("=> no checkpoint found at '{}'".format(args.pretrained))

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.MSELoss().cuda(args.gpu)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    if args.lars:
        print("=> use LARS optimizer.")
        from apex.parallel.LARC import LARC
        optimizer = LARC(optimizer=optimizer, trust_coefficient=.001, clip=False)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=256, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, args)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                                                    and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
            }, is_best, os.path.join(args.output_dir, f'checkpoint{(epoch + 1):04}.pth'))
            if epoch == args.start_epoch:
                sanity_check(model.module.model_phi.state_dict(), args.pretrained)

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        with open(os.path.join(args.output_dir, f'best_acc.txt'), 'w') as best_acc_result:
            best_acc_result.write("Best acc: " + str(best_acc1))
        save_prediction(val_loader, model, criterion, args)


def train(train_loader, model, criterion, optimizer, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    model.module.model_theta.train()
    """
    Switch to eval mode:
    It is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.module.model_phi.eval()

    # diffusion configs
    betas = card.diffusion_utils.make_beta_schedule(schedule="linear", num_timesteps=args.num_timesteps, 
        start=args.beta_1, end=args.beta_T).cuda(
        args.gpu, non_blocking=True)
    betas_sqrt = torch.sqrt(betas)
    alphas = 1.0 - betas
    one_minus_betas_sqrt = torch.sqrt(alphas)
    alphas_cumprod = alphas.cumprod(dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1).cuda(args.gpu, non_blocking=True), alphas_cumprod[:-1]], dim=0
    )
    posterior_mean_coeff_1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coeff_2 = (
            torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    )
    posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.cuda(args.gpu, non_blocking=True)
        target = nn.functional.one_hot(target, num_classes=args.num_classes).float()
        target = target.cuda(args.gpu, non_blocking=True)

        # antithetic sampling
        n = images.size(0)
        t = torch.randint(
            low=0, high=args.num_timesteps, size=(n // 2 + 1,)
        ).cuda(args.gpu, non_blocking=True)
        t = torch.cat([t, args.num_timesteps - 1 - t], dim=0)[:n]

        # compute output in mode phi
        feats, yhat = model(images, target, t, mode='phi')

        # forward diffusion
        e = torch.randn_like(target).cuda(args.gpu, non_blocking=True)
        y_t = card.diffusion_utils.q_sample(target, yhat,
                                            alphas_bar_sqrt, one_minus_alphas_bar_sqrt, t, noise=e)

        output = model(feats, y_t, t, yhat, mode='theta')
        loss = criterion(e, output)

        # record loss
        losses.update(loss.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_ref = AverageMeter('Acc_ref@1', ':6.2f')
    top5_ref = AverageMeter('Acc_ref@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    # diffusion configs
    betas = card.diffusion_utils.make_beta_schedule(schedule="linear", num_timesteps=args.num_timesteps, 
        start=args.beta_1, end=args.beta_T).cuda(
        args.gpu, non_blocking=True)
    betas_sqrt = torch.sqrt(betas)
    alphas = 1.0 - betas
    one_minus_betas_sqrt = torch.sqrt(alphas)
    alphas_cumprod = alphas.cumprod(dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1).cuda(args.gpu, non_blocking=True), alphas_cumprod[:-1]], dim=0
    )
    posterior_mean_coeff_1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coeff_2 = (
            torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    )
    posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output in mode phi
            feats, yhat = model(images, target, target, mode='phi')

            # compute output
            output = card.diffusion_utils.p_sample_loop(model, feats, yhat, yhat,
                                                        args.num_timesteps, alphas,
                                                        one_minus_alphas_bar_sqrt,
                                                        only_last_sample=True)

            loss = criterion(output, nn.functional.one_hot(target, num_classes=args.num_classes).float())

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            acc1_ref, acc5_ref = accuracy(yhat, target, topk=(1, 5))
            top1_ref.update(acc1_ref[0], images.size(0))
            top5_ref.update(acc5_ref[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        print(' * Ref Acc@1 {top1_ref.avg:.3f} Ref Acc@5 {top5_ref.avg:.3f}'
              .format(top1_ref=top1_ref, top5_ref=top5_ref))

    return top1.avg


def save_prediction(val_loader, model, criterion, args):
    # switch to evaluate mode
    model.eval()

    # diffusion configs
    betas = card.diffusion_utils.make_beta_schedule(schedule="linear", num_timesteps=args.num_timesteps, 
        start=args.beta_1, end=args.beta_T).cuda(
        args.gpu, non_blocking=True)
    betas_sqrt = torch.sqrt(betas)
    alphas = 1.0 - betas
    one_minus_betas_sqrt = torch.sqrt(alphas)
    alphas_cumprod = alphas.cumprod(dim=0)
    alphas_bar_sqrt = torch.sqrt(alphas_cumprod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_cumprod)
    alphas_cumprod_prev = torch.cat(
        [torch.ones(1).cuda(args.gpu, non_blocking=True), alphas_cumprod[:-1]], dim=0
    )
    posterior_mean_coeff_1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )
    posterior_mean_coeff_2 = (
            torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)
    )
    posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )

    test_labels = []
    test_preds = []
    print("=> Staring extract features and labels...")
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output in mode phi
            feats, yhat = model(images, target, target, mode='phi')

            # compute output
            preds = []
            for i in range(1000):
                output = card.diffusion_utils.p_sample_loop(model, feats, yhat, yhat,
                                                            args.num_timesteps, alphas,
                                                            one_minus_alphas_bar_sqrt,
                                                            only_last_sample=True)

                preds.append(output)

            B, nc = output.shape
            tmp = torch.stack(preds).permute(1, 0, 2)
            assert tmp.shape == (B, 1000, nc)
            test_labels.append(target.cpu())
            test_preds.append(tmp.cpu())

        torch.save(torch.cat(test_labels), os.path.join(args.output_dir, f'labels_best.pth'))
        torch.save(torch.cat(test_preds), os.path.join(args.output_dir, f'preds_best.pth'))
        print("=> Finished saving at {}!".format(args.output_dir))

    return


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        name = filename.split('/')[-1]
        shutil.copyfile(filename, filename.replace(name, 'model_best.pth.tar'))


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'module.' + k
        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
