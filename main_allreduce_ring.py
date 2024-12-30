import argparse
import os
import random
import time
import numpy as np
import shutil
import csv

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
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
parser.add_argument('--epochs', default=5, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=64, type=int,
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
parser.add_argument('-n', '--name', default='allreduce',
                    help='experiment name')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--seed', default=212, type=int,
                    help='seed for initializing training. ')



def save_checkpoint(state, is_best, workpath, filename='checkpoint.pth.tar'):
    if not os.path.exists(workpath):
        os.makedirs(workpath)
    torch.save(state, workpath+'/'+filename)
    if is_best:
        shutil.copyfile(workpath+'/'+filename, workpath+'/'+'model_best.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch, device, group, rank, size, args):
    batch_time = []
    losses = []
    top1 = []
    top5 = []
    throughputs = []

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # move data to the same device as model, asynchronous
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # throughputs
        batch_size = images.size(0)
        throughput = batch_size / (time.time() - end)
        throughputs.append(throughput)

        # compute output
        output = model(images)
        loss = criterion(output, target)
        # print(f'== debug before: rank {rank} has loss {loss.item()}')
        dist.all_reduce(loss, op=dist.ReduceOp.SUM, group=group)
        loss /= size
        # print(f'== debug after : rank {rank} has loss {loss.item()}')

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
        # measure gpu situation
        gpu_utilization = torch.cuda.utilization(device)
        memory_allocated = torch.cuda.memory_allocated(device) / 1e9
        memory_reserved = torch.cuda.memory_reserved(device) / 1e9

        with open(args.csv_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                epoch, i, batch_time[-1], losses[-1], top1[-1].item(), top5[-1].item(), throughput,
                gpu_utilization, memory_allocated, memory_reserved
            ])

        if rank == 0 and i % args.print_freq == 0:
            print(f'round {i}({epoch}), batch_time: {batch_time[-1]:6.3f}, loss: {losses[-1]:.4e}, top1: {top1[-1]:6.2f}, top5: {top5[-1]:6.2f}, throughput: {throughputs[-1]:6.2f} samples/s')
            with open(args.logpath, 'a') as file:
                file.write(f'round {i}({epoch}), batch_time: {batch_time[-1]:6.3f}, loss: {losses[-1]:.4e}, top1: {top1[-1]:6.2f}, top5: {top5[-1]:6.2f}, throughput: {throughputs[-1]:6.2f} samples/s\n')
    
    batch_time = torch.Tensor(batch_time)
    losses = torch.Tensor(losses)
    top1 = torch.Tensor(top1)
    top5 = torch.Tensor(top5)
    if rank == 0:
        print(f'=== epoch {epoch} average: batch_time: {batch_time.mean():6.3f}, loss: {losses.mean():.4e}, top1: {top1.mean():6.2f}, top5: {top5.mean():6.2f}')
        with open(args.logpath, 'a') as file:
            file.write(f'=== epoch {epoch} average: batch_time: {batch_time.mean():6.3f}, loss: {losses.mean():.4e}, top1: {top1.mean():6.2f}, top5: {top5.mean():6.2f}\n')

def validate(val_loader, model, criterion, device, group, rank, size, args):

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

            if rank == 0 and i % args.print_freq == 0:
                print(f'\t\tround {i}, batch_time: {batch_time[-1]:6.3f}, loss: {losses[-1]:.4e}, top1: {top1[-1]:6.2f}, top5: {top5[-1]:6.2f}')
                with open(args.logpath, 'a') as file:
                    file.write(f'\t\tround {i}, batch_time: {batch_time[-1]:6.3f}, loss: {losses[-1]:.4e}, top1: {top1[-1]:6.2f}, top5: {top5[-1]:6.2f}\n')

    batch_time = torch.Tensor(batch_time)
    losses = torch.Tensor(losses)
    top1 = torch.Tensor(top1)
    top5 = torch.Tensor(top5)
    if rank == 0:
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


def run(rank, size, args):
    # logs
    workpath = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/engineering_hw3/outputs/' + args.name
    if rank == 0:
        if not os.path.exists(workpath):
            os.makedirs(workpath)
    logpath = workpath + '/log.txt'
    if rank == 0:
        with open(logpath, 'a') as file:
            file.write(f"Experiment: {args.name}\n")
    args.logpath = logpath

    csv_path = workpath + f'/metrics_{rank}.csv'
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Batch', 'Batch Time', 'Loss', 'Top1 Accuracy', 'Top5 Accuracy', 'Throughput (samples/s)', 'GPU Utilization (%)', 'Memory Allocated (GB)', 'Memory Reserved (GB)'])
    args.csv_path = csv_path

    # seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set device
    device = torch.device('cuda:{}'.format(rank))
    torch.cuda.set_device(device)
    group = dist.new_group(list(range(size)))

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
    
    # # optionally resume from a checkpoint
    best_acc1 = 0
    
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
        validate(val_loader, model, criterion, device, group, rank, size, args)
        return

    for epoch in range(args.epochs):

        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, device, group, rank, size, args)

        # evaluate on validation set
        acc1 = validate(val_loader, model, criterion, device, group, rank, size, args)
        
        scheduler.step()
        
        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if rank == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
                'scheduler' : scheduler.state_dict()
            }, is_best, workpath)

def init_process(rank, size, fn, args, backend='nccl'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost' # '127.0.0.1'
    os.environ['MASTER_PORT'] = '8066'
    os.environ['NCCL_ALGO'] = 'Ring' # 'Tree'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, args)


if __name__ == "__main__":
    args = parser.parse_args()
    size = 4
    processes = []
    mp.set_start_method("spawn")
    for rank in range(size):
        p = mp.Process(target=init_process, args=(rank, size, run, args))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()