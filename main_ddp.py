import argparse
import os
import random
import time
import numpy as np
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torchvision.models import resnet50
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', nargs='?', default='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/ILSVRC/Data/CLS-LOC',
                    help='path to dataset (default: imagenet)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default=None, type=str, metavar='PATH', 
                    help='path to latest checkpoint')
parser.add_argument('-n', '--name', default='tmp',
                    help='experiment name')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=212, type=int,
                    help='seed for initializing training. ')
parser.add_argument("--local-rank", default=-1, type=int)


def main():
    args = parser.parse_args()

    # logs
    logpath = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/engineering_hw3/outputs/' + args.name
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    logpath = logpath + '/log.txt'
    with open(logpath, 'a') as file:
        file.write(f"Experiment: {args.name}\n")
    args.logpath = logpath

    # seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set device
    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl')
        device = torch.device('cuda:{}'.format(args.local_rank))

    # create model
    print(f"=> creating model resnet50, pretrained: {args.pretrained}")
    model = resnet50(pretrained=args.pretrained)
    model = model.to(device)

    # define loss func, optim, scheduler
    criterion = nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # optionally resume from a checkpoint
    best_acc1 = 0
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            with open(logpath, 'a') as file:
                file.write("=> loading checkpoint '{}'\n".format(args.resume))
            if torch.cuda.is_available() and args.local_rank:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.local_rank)
                checkpoint = torch.load(args.resume, map_location=loc)
            else:
                checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.local_rank is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.to(args.local_rank)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val_label')
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

    val_dataset = datasets.ImageFolder(
        valdir,
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=True)

    val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, device, args)
        return

    for epoch in range(args.start_epoch, args.epochs):

        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device, args)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer' : optimizer.state_dict(),
            'scheduler' : scheduler.state_dict()
        }, is_best, args.name)

def save_checkpoint(state, is_best, exp_name, filename='checkpoint.pth.tar'):
    filepath = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/engineering_hw3/outputs/' + exp_name
    if not os.path.exists(filepath):
        os.makedirs(filepath)
    torch.save(state, filepath+'/'+filename)
    if is_best:
        shutil.copyfile(filepath+'/'+filename, filepath+'/'+'model_best.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, device, args):
    batch_time = []
    losses = []
    top1 = []
    top5 = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # move data to the same device as model
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        batch_time.append(time.time() - end)
        losses.append(loss.item())
        top1.append(acc1[0])
        top5.append(acc5[0])

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.append(time.time() - end)
        end = time.time()

        if args.local_rank == 0 and i % args.print_freq == 0:
            print(f'round {i}({epoch}), batch_time: {batch_time[-1]:6.3f}, loss: {losses[-1]:.4e}, top1: {top1[-1]:6.2f}, top5: {top5[-1]:6.2f}')
            with open(args.logpath, 'a') as file:
                file.write(f'round {i}({epoch}), batch_time: {batch_time[-1]:6.3f}, loss: {losses[-1]:.4e}, top1: {top1[-1]:6.2f}, top5: {top5[-1]:6.2f}\n')

    batch_time = torch.Tensor(batch_time)
    losses = torch.Tensor(losses)
    top1 = torch.Tensor(top1)
    top5 = torch.Tensor(top5)
    if args.local_rank == 0:
        print(f'=== epoch {epoch} average: batch_time: {batch_time.mean():6.3f}, loss: {losses.mean():.4e}, top1: {top1.mean():6.2f}, top5: {top5.mean():6.2f}')
        with open(args.logpath, 'a') as file:
            file.write(f'=== epoch {epoch} average: batch_time: {batch_time.mean():6.3f}, loss: {losses.mean():.4e}, top1: {top1.mean():6.2f}, top5: {top5.mean():6.2f}\n')

def validate(val_loader, model, criterion, device, args):

    model.eval()
    batch_time = []
    losses = []
    top1 = []
    top5 = []

    end = time.time()
    with torch.no_grad():
        for i, (images, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                images = images.cuda(device, non_blocking=True)
                target = target.cuda(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_time.append(time.time() - end)
            end = time.time()
            losses.append(loss.item())
            top1.append(acc1[0])
            top5.append(acc5[0])

            if args.local_rank == 0 and i % args.print_freq == 0:
                print(f'\t\tround {i}, batch_time: {batch_time[-1]:6.3f}, loss: {losses[-1]:.4e}, top1: {top1[-1]:6.2f}, top5: {top5[-1]:6.2f}')
                with open(args.logpath, 'a') as file:
                    file.write(f'\t\tround {i}, batch_time: {batch_time[-1]:6.3f}, loss: {losses[-1]:.4e}, top1: {top1[-1]:6.2f}, top5: {top5[-1]:6.2f}\n')

    batch_time = torch.Tensor(batch_time)
    losses = torch.Tensor(losses)
    top1 = torch.Tensor(top1)
    top5 = torch.Tensor(top5)
    if args.local_rank == 0:
        print(f'final average: batch_time: {batch_time.mean():6.3f}, loss: {losses.mean():.4e}, top1: {top1.mean():6.2f}, top5: {top5.mean():6.2f}')
        with open(args.logpath, 'a') as file:
            file.write(f'final average: batch_time: {batch_time.mean():6.3f}, loss: {losses.mean():.4e}, top1: {top1.mean():6.2f}, top5: {top5.mean():6.2f}\n')
    return top1.mean()

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
    main()