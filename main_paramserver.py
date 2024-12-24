import argparse
import os
import random
import time
import numpy as np
import shutil
from threading import Lock

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.distributed.optim import DistributedOptimizer
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
from torchvision.models import resnet50, ResNet
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset


def remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), call_method, args=args, kwargs=kwargs)

class DistributedResNet(ResNet):
    def __init__(self, 
        block,
        layers,
        num_gpus = 0
    ):
        super(DistributedResNet50, self).__init__(block, layers)
        print(f"Using {num_gpus} GPUs to train")
        self.num_gpus = num_gpus
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() and self.num_gpus > 0 else "cpu")
        print(f"Putting stage 0-2 on {str(device)}")
        # stage 0
        self.conv1 = self.conv1.to(device)
        self.bn1 = self.bn1.to(device)
        self.relu = self.relu.to(device)
        self.maxpool = self.maxpool.to(device)
        self.layer1 = self.layer1.to(device)
        self.layer2 = self.layer2.to(device)

        if "cuda" in str(device) and num_gpus > 1:
            device = torch.device("cuda:1")
        print(f"Putting stage 3-4 on {str(device)}")
        self.layer3 = self.layer3.to(device)
        self.layer4 = self.layer4.to(device)
        self.avgpool = self.avgpool.to(device)
        self.fc = self.fc.to(device)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        # Move tensor to next device if necessary
        next_device = next(self.layer3.parameters()).device
        x = x.to(next_device)

        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

class ParameterServer(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        model = DistributedResNet(Bottleneck, [3, 4, 6, 3], num_gpus=num_gpus)
        self.model = model
        self.input_device = torch.device(
            "cuda:0" if torch.cuda.is_available() and num_gpus > 0 else "cpu")
    
    def forward(self, inp):
        inp = inp.to(self.input_device)
        out = self.model(inp)
        # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
        # Tensors must be moved in and out of GPU memory due to this.
        out = out.to("cpu")
        return out

# Use dist autograd to retrieve gradients accumulated for this model.
# Primarily used for verification.
def get_dist_gradients(self, cid):
    grads = dist_autograd.get_gradients(cid)
    # This output is forwarded over RPC, which as of 1.5.0 only accepts CPU tensors.
    # Tensors must be moved in and out of GPU memory due to this.
    cpu_grads = {}
    for k, v in grads.items():
        k_cpu, v_cpu = k.to("cpu"), v.to("cpu")
        cpu_grads[k_cpu] = v_cpu
    return cpu_grads

# Wrap local parameters in a RRef. Needed for building the
# DistributedOptimizer which optimizes paramters remotely.
def get_param_rrefs(self):
    param_rrefs = [rpc.RRef(param) for param in self.model.parameters()]
    return param_rrefs

# The global parameter server instance.
param_server = None
# A lock to ensure we only have one parameter server.
global_lock = Lock()


def get_parameter_server(num_gpus=0):
    """
    Returns a singleton parameter server to all trainer processes
    """
    global param_server
    # Ensure that we get only one handle to the ParameterServer.
    with global_lock:
        if not param_server:
            # construct it once
            param_server = ParameterServer(num_gpus=num_gpus)
        return param_server

def run_parameter_server(rank, world_size):
    # The parameter server just acts as a host for the model and responds to
    # requests from trainers.
    # rpc.shutdown() will wait for all workers to complete by default, which
    # in this case means that the parameter server will wait for all trainers
    # to complete, and then exit.
    print("PS master initializing RPC")
    rpc.init_rpc(name="parameter_server", rank=rank, world_size=world_size)
    print("RPC initialized! Running parameter server...")
    rpc.shutdown()
    print("RPC shutdown on parameter server.")


# --------- Trainers --------------------

# nn.Module corresponding to the network trained by this trainer. The
# forward() method simply invokes the network on the given parameter
# server.
class TrainerNet(nn.Module):
    def __init__(self, num_gpus=0):
        super().__init__()
        self.num_gpus = num_gpus
        self.param_server_rref = rpc.remote(
            "parameter_server", get_parameter_server, args=(num_gpus,))

    def get_global_param_rrefs(self):
        remote_params = remote_method(
            ParameterServer.get_param_rrefs,
            self.param_server_rref)
        return remote_params
    
    def forward(self, x):
        model_output = remote_method(
            ParameterServer.forward, self.param_server_rref, x)
        return model_output

# scheduler for DistributedOptimizer (https://discuss.pytorch.org/t/does-distributedoptimizer-support-zero-grad-and-lr-scheduling/81359/2)
def create_lr_schheduler(opt_rref):
    # create and return lr_schheduler
    scheduler = StepLR(opt_rref, step_size=30, gamma=0.1)
    return scheduler

def lrs_step(lrs_rref):
    lrs_rref.local_value().step()


def run_training_loop(rank, num_gpus, train_loader, test_loader, args):
    # Runs the typical nueral network forward + backward + optimizer step, but
    # in a distributed fashion.
    net = TrainerNet(num_gpus=num_gpus)

    # Build DistributedOptimizer.
    param_rrefs = net.get_global_param_rrefs()
    opt = DistributedOptimizer(optim.SGD, param_rrefs, args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    lrs_rrefs = []
    for opt_rref in opt.remote_optimizers:
        lrs_rrefs = rptc.remote(opt_rref.owner(), create_lr_schheduler, args=(opt_rref,))

    for epoch in range(artgs.epochs):
        losses = []
        top1 = []
        top5 = []
        t1 = time.time()
        for i, (data, target) in enumerate(train_loader):
            with dist_autograd.context() as cid:
                # switch to train mode
                net.train()
                model_output = net(data)
                target = target.to(model_output.device)
                criterion = nn.CrossEntropyLoss().to(model_output.device)
                loss = criterion(model_output, target)

                acc1, acc5 = accuracy(output, target, topk=(1, 5))
                losses.append(loss.item())
                top1.append(acc1[0])
                top5.append(acc5[0])
                dist_autograd.backward(cid, [loss])
                # Ensure that dist auograd ran successfully and gradients were
                # returned.
                assert remote_method(
                    ParameterServer.get_dist_gradients,
                    net.param_server_rref,
                    cid) != {}
                opt.step(cid)
                # scheduler update
                futs = []
                for lrs_rref in lrs_rrefs:
                    futs.append(rpc.rpc_async(lrs_rref.owner(), lrs_step, args=(lrs_rref,)))
                [fut.wait() for fut in futs]

                if rank == 0 and i % args.print_freq == 0:
                    print(f'round {i}({epoch}), loss: {losses[-1]:.4e}, top1: {top1[-1]:6.2f}, top5: {top5[-1]:6.2f}')
                    with open(args.logpath, 'a') as file:
                        file.write(f'round {i}({epoch}), top1: {top1[-1]:6.2f}, top5: {top5[-1]:6.2f}\n')

        losses = torch.Tensor(losses)
        top1 = torch.Tensor(top1)
        top5 = torch.Tensor(top5)
        if rank == 0:
            print(f'=== epoch {epoch}: batch_time: {time.time() - t1}, loss: {losses.mean():.4e}, top1: {top1.mean():6.2f}, top5: {top5.mean():6.2f}')
            with open(args.logpath, 'a') as file:
                file.write(f'=== epoch {epoch}: batch_time: {time.time() - t1}, loss: {losses.mean():.4e}, top1: {top1.mean():6.2f}, top5: {top5.mean():6.2f}\n')


        print(f"Training epoch {epoch} complete!")
        print("Getting accuracy....")
        get_accuracy(rank, test_loader, net, args)


def get_accuracy(rank, test_loader, model, args):
    model.eval()
    losses = []
    top1 = []
    top5 = []
    t1 = time.time()
    # Use GPU to evaluate if possible
    device = torch.device("cuda:0" if model.num_gpus > 0
        and torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        for i, (data, target) in enumerate(test_loader):
            out = model(data, -1)
            out, target = out.to(device), target.to(device)
            criterion = nn.CrossEntropyLoss().to(device)
            loss = criterion(out, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.append(loss.item())
            top1.append(acc1[0])
            top5.append(acc5[0])

    losses = torch.Tensor(losses)
    top1 = torch.Tensor(top1)
    top5 = torch.Tensor(top5)
    if rank == 0:
        print(f'eval epoch {epoch}: batch_time: {time.time() - t1}, loss: {losses.mean():.4e}, top1: {top1.mean():6.2f}, top5: {top5.mean():6.2f}')
        with open(args.logpath, 'a') as file:
            file.write(f'eval epoch {epoch}: batch_time: {time.time() - t1}, loss: {losses.mean():.4e}, top1: {top1.mean():6.2f}, top5: {top5.mean():6.2f}\n')


# Main loop for trainers.
def run_worker(rank, world_size, num_gpus, train_loader, test_loader):
    print(f"Worker rank {rank} initializing RPC")
    rpc.init_rpc(
        name=f"trainer_{rank}",
        rank=rank,
        world_size=world_size)

    print(f"Worker {rank} done initializing RPC")

    run_training_loop(rank, num_gpus, train_loader, test_loader)
    rpc.shutdown()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('data', metavar='DIR', nargs='?', default='/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/public/public_datas/ILSVRC/Data/CLS-LOC',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
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
    parser.add_argument('-n', '--name', default='tmp',
                        help='experiment name')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--seed', default=212, type=int,
                        help='seed for initializing training. ')
    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="""Total number of participating processes. Should be the sum of
            master node and all training nodes.""")
    parser.add_argument(
        "--rank",
        type=int,
        default=None,
        help="Global rank of this process. Pass in 0 for master.")
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=2,
        help="""Number of GPUs to use for training, Currently supports between 0
            and 2 GPUs. Note that this argument will be passed to the parameter servers.""")
    parser.add_argument(
        "--master_addr",
        type=str,
        default="10.244.193.23", # localhost
        help="""Address of master, will default to localhost if not provided.
            Master must be able to accept network traffic on the address + port.""")
    parser.add_argument(
        "--master_port",
        type=str,
        default="8090",
        help="""Port that master is listening on, will default to 29500 if not
            provided. Master must be able to accept network traffic on the host and port.""")


    args = parser.parse_args()
    assert args.rank is not None, "must provide rank argument."
    assert args.num_gpus <= 3, f"Only 0-2 GPUs currently supported (got {args.num_gpus})."
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ["MASTER_PORT"] = args.master_port

    logpath = '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/chenxinyan-240108120066/chenxinyan/engineering_hw3/outputs/' + args.name
    if not os.path.exists(logpath):
        os.makedirs(logpath)
    logpath = logpath + '/log.txt'
    with open(logpath, 'a') as file:
        file.write(f"Experiment: {args.name}\n")
    args.logpath = logpath

    processes = []
    world_size = args.world_size
    if args.rank == 0:
        p = mp.Process(target=run_parameter_server, args=(0, world_size))
        p.start()
        processes.append(p)
    else:
        # Get data to train on
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
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, sampler=train_sampler,
            num_workers=args.workers, shuffle=True, pin_memory=True)
        test_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=args.batch_size, sampler=val_sampler,
            num_workers=args.workers, shuffle=True, pin_memory=True)   # ?: shuffle=True??
        # start training worker on this node
        p = mp.Process(
            target=run_worker,
            args=(
                args.rank,
                world_size, args.num_gpus,
                train_loader,
                test_loader))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


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
