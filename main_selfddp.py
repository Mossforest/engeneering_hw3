import argparse
import os
import random
import time
import numpy as np
import shutil
import itertools
import copy

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
from torch.cuda._utils import _get_device_index
from torch.distributed.distributed_c10d import _get_default_group
from torch.nn.parallel.scatter_gather import scatter_kwargs, gather
from torch.nn.parallel.parallel_apply import parallel_apply
from contextlib import contextmanager
from torch.distributed.elastic.multiprocessing.errors import record

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


class MyDDP(nn.Module):
    def __init__(self, module, device_ids=None,
                 output_device=None, dim=0, broadcast_buffers=True,
                 bucket_cap_mb=25,
                 find_unused_parameters=False,
                 check_reduction=False):

        super(MyDDP, self).__init__()

        self.device_ids = list(map(lambda x: _get_device_index(x, True), device_ids))

        if output_device is None:
            output_device = device_ids[0]

        self.output_device = _get_device_index(output_device, True)
        self.process_group = _get_default_group()
        self.dim = dim
        self.module = module
        self.broadcast_buffers = broadcast_buffers
        self.find_unused_parameters = find_unused_parameters
        self.require_backward_grad_sync = True
        self.require_forward_param_sync = True

        MB = 1024 * 1024
        # used for intra-node param sync and inter-node sync as well
        self.broadcast_bucket_size = int(250 * MB)
        # reduction bucket size
        self.bucket_bytes_cap = int(bucket_cap_mb * MB)

        # 1. sync params and buffers
        module_states = list(self.module.state_dict().values())
        if len(module_states) > 0:
            dist._broadcast_coalesced(self.process_group, module_states, self.broadcast_bucket_size)

        self._ddp_init_helper()
    
    def __getstate__(self):
        self._check_default_group()
        attrs = copy.copy(self.__dict__)
        del attrs['process_group']
        del attrs['reducer']
        return attrs

    def __setstate__(self, state):
        # If serializable, then the process group should be the default one
        self.process_group = _get_default_group()
        super(MyDDP, self).__setstate__(state)
        self.__dict__.setdefault('require_forward_param_sync', True)
        self.__dict__.setdefault('require_backward_grad_sync', True)
        self._ddp_init_helper()

    def _check_default_group(self):
        pickle_not_supported = False
        try:
            if self.process_group != _get_default_group():
                pickle_not_supported = True
        except RuntimeError:
            pickle_not_supported = True

        if pickle_not_supported:
            raise RuntimeError("DDP Pickling/Unpickling are only supported "
                               "when using DDP with the default process "
                               "group. That is, when you have called "
                               "init_process_group and have not passed "
                               "process_group argument to DDP constructor")

    @contextmanager
    def no_sync(self):
        r"""
        A context manager to disable gradient synchronizations across DDP
        processes. Within this context, gradients will be accumulated on module
        variables, which will later be synchronized in the first
        forward-backward pass exiting the context.

        Example::

            >>> ddp = torch.nn.DistributedDataParallel(model, pg)
            >>> with ddp.no_sync():
            ...   for input in inputs:
            ...     ddp(input).backward()  # no synchronization, accumulate grads
            ... ddp(another_input).backward()  # synchronize grads
        """
        old_require_backward_grad_sync = self.require_backward_grad_sync
        self.require_backward_grad_sync = False
        try:
            yield
        finally:
            self.require_backward_grad_sync = old_require_backward_grad_sync
    
    def _ddp_init_helper(self):
        """
        Initialization helper function that does the following:

        (1) replicating the module from device[0] to the other devices
        (2) bucketing the parameters for reductions
        (3) resetting the bucketing states
        (4) registering the grad hooks
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """
        # if self.device_ids and len(self.device_ids) > 1:

        #     import warnings
        #     warnings.warn(
        #         "Single-Process Multi-GPU is not the recommended mode for "
        #         "DDP. In this mode, each DDP instance operates on multiple "
        #         "devices and creates multiple module replicas within one "
        #         "process. The overhead of scatter/gather and GIL contention "
        #         "in every forward pass can slow down training. "
        #         "Please consider using one DDP instance per device or per "
        #         "module replica by explicitly setting device_ids or "
        #         "CUDA_VISIBLE_DEVICES. "
        #         "NB: There is a known issue in nn.parallel.replicate that "
        #         "prevents a single DDP instance to operate on multiple model "
        #         "replicas."
        #     )
        #     # TODO：(确认一下ddp有没有这个warning 感觉不对)（在等卡）

        #     # only create replicas for single-device CUDA modules
        #     #
        #     # TODO: we don't need to replicate params in here. they're always going to
        #     # be broadcasted using larger blocks in broadcast_coalesced, so it might be
        #     # better to not pollute the caches with these small blocks
        #     self._module_copies = replicate(self.module, self.device_ids, detach=True)
        #     self._module_copies[0] = self.module

        #     for module_copy in self._module_copies[1:]:
        #         for param, copy_param in zip(self.module.parameters(), module_copy.parameters()):
        #             copy_param.requires_grad = param.requires_grad
        #     exit()

        # else:
        self._module_copies = [self.module]

        self.modules_params = [list(m.parameters()) for m in self._module_copies]

        parameters, expect_sparse_gradient = self._build_params_for_reducer()
        
        bucket_size_limits = [self.bucket_bytes_cap]
        (
            bucket_indices,
            per_bucket_size_limits,
        ) = dist._compute_bucket_assignment_by_size(
            parameters,
            bucket_size_limits,
            expect_sparse_gradient,
        )

        # Note: reverse list of buckets because we want to approximate the
        # order in which their gradients are produced, and assume they
        # are used in the forward pass in the order they are defined.
        param_to_name_mapping = self._build_debug_param_to_name_mapping(parameters)
        self.reducer = dist.Reducer(
            parameters,
            list(reversed(bucket_indices)),
            list(reversed(per_bucket_size_limits)),
            self.process_group,
            expect_sparse_gradient,
            self.bucket_bytes_cap,
            self.find_unused_parameters,
            False, #self.gradient_as_bucket_view,
            param_to_name_mapping,
            self.bucket_bytes_cap,
        )

        # # passing a handle to torch.nn.SyncBatchNorm layer
        # self._passing_sync_batchnorm_handle(self._module_copies)
        
        # TODO: where is hook?
    
    def train(self, mode=True):
        super(MyDDP, self).train(mode)
        for module in self._module_copies[1:]:
            module.train(mode)
    
    def forward(self, *inputs, **kwargs):
        if self.require_forward_param_sync:
            self._sync_params()

        if self.device_ids:
            inputs, kwargs = scatter_kwargs(inputs, kwargs, self.device_ids, dim=self.dim)
            if len(self.device_ids) == 1:
                output = self.module(*inputs[0], **kwargs[0])
            else:
                replicas = self._module_copies[:len(inputs)]
                outputs = parallel_apply(replicas, inputs, kwargs, self.device_ids[:len(replicas)])
                output = gather(outputs, self.output_device, dim=self.dim)
        else:
            output = self.module(*inputs, **kwargs)

        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            # We'll return the output object verbatim since it is a freeform
            # object. We need to find any tensors in this object, though,
            # because we need to figure out which parameters were used during
            # this forward pass, to ensure we short circuit reduction for any
            # unused parameters. Only if `find_unused_parameters` is set.
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False

        return output

    def _sync_params(self):
        with torch.no_grad():
            # only do intra-node parameters sync for replicated single-device
            # CUDA modules
            if self.device_ids and len(self.device_ids) > 1:
                # intra-node parameter sync
                result = torch.cuda.comm.broadcast_coalesced(
                    self.modules_params[0],
                    self.device_ids,
                    self.broadcast_bucket_size)
                for tensors, module_params in zip(result[1:],
                                                  self.modules_params[1:]):
                    for tensor, param in zip(tensors, module_params):
                        param.set_(tensor)
                        # Assume we have just run the optimizer and zeroed the
                        # grads of the parameters on the root model. We need
                        # to zero the grads on all model replicas as well.
                        # This snippet is copied from torch.optim.Optimizer.
                        if param.grad is not None:
                            param.grad.detach_()
                            param.grad.zero_()

            # module buffer sync
            if self.broadcast_buffers and len(self.modules_buffers) > 0:
                # Synchronize buffers across processes.
                # The process with rank 0 is considered the authoritative copy.
                dist._broadcast_coalesced(self.process_group, self.modules_buffers, self.broadcast_bucket_size)
                # only do intra-node buffer sync for replicated single-device
                # CUDA modules
                if self.device_ids and len(self.device_ids) > 1:
                    # intra-node buffer sync
                    result = torch.cuda.comm.broadcast_coalesced(
                        self.modules_buffers,
                        self.device_ids,
                        self.broadcast_bucket_size)
                    for tensors, module_buffers in zip(result[1:],
                                                       self.modules_buffers[1:]):
                        for tensor, buffer in zip(tensors, module_buffers):
                            buffer.set_(tensor)
    
    def _build_debug_param_to_name_mapping(self, parameters):
        param_to_param_index = {parameters[i]: i for i in range(len(parameters))}
        param_set = set(parameters)
        param_index_to_param_fqn = {}
        for module_name, module in self.module.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fqn = f"{module_name}.{param_name}"
                # Bypass ignored parameters since those are not reduced by DDP
                # to begin with.
                if param.requires_grad:
                    if param not in param_set:
                        assert False
                        # self._log_and_throw(
                        #     ValueError,
                        #     f"Param with name {fqn} found in module parameters, but not DDP parameters."
                        #     " This indicates a bug in DDP, please report an issue to PyTorch.",
                        # )
                    param_index = param_to_param_index[param]
                    param_index_to_param_fqn[param_index] = fqn

        # Ensure we covered all parameters
        if len(param_set) != len(param_index_to_param_fqn):
            assert False
                # ValueError,
                # (
                #     "Expected param to name mapping to cover all parameters, but"
                #     f" got conflicting lengths: {len(param_set)} vs "
                #     f"{len(param_index_to_param_fqn)}. This indicates a bug in DDP"
                #     ", please report an issue to PyTorch."
                # ),

        return param_index_to_param_fqn
    
    def _build_params_for_reducer(self):
        # Build tuple of (module, parameter) for all parameters that require grads.
        modules_and_parameters = [
            (module, parameter)
            for module_name, module in self.module.named_modules()
            for parameter in [
                param
                # Note that we access module.named_parameters instead of
                # parameters(module). parameters(module) is only needed in the
                # single-process multi device case, where it accesses replicated
                # parameters through _former_parameters.
                for param_name, param in module.named_parameters(recurse=False)
                if param.requires_grad
                # and f"{module_name}.{param_name}" not in self.parameters_to_ignore
            ]
        ]

        # Deduplicate any parameters that might be shared across child modules.
        memo = set()
        modules_and_parameters = [
            # "p not in memo" is the deduplication check.
            # "not memo.add(p)" is always True, and it's only there to cause "add(p)" if needed.
            (m, p)
            for m, p in modules_and_parameters
            if p not in memo and not memo.add(p)  # type: ignore[func-returns-value]
        ]

        # Build list of parameters.
        parameters = [parameter for _, parameter in modules_and_parameters]

        # Checks if a module will produce a sparse gradient.
        def produces_sparse_gradient(module):
            if isinstance(module, (torch.nn.Embedding, torch.nn.EmbeddingBag)):
                return module.sparse
            return False

        # Build list of booleans indicating whether or not to expect sparse
        # gradients for the corresponding parameters.
        expect_sparse_gradient = [
            produces_sparse_gradient(module) for module, _ in modules_and_parameters
        ]

        self._assign_modules_buffers()

        return parameters, expect_sparse_gradient
    
    def _assign_modules_buffers(self):
        """
        Assign self.module.named_buffers to self.modules_buffers.

        Assigns module buffers to self.modules_buffers which are then used to
        broadcast across ranks when broadcast_buffers=True. Note that this
        must be called every time buffers need to be synced because buffers can
        be reassigned by user module,
        see https://github.com/pytorch/pytorch/issues/63916.
        """
        # Collect buffers for modules, filtering out buffers that should be ignored.
        named_module_buffers = [
            (buffer, buffer_name)
            for buffer_name, buffer in self.module.named_buffers()
            # if buffer_name not in self.parameters_to_ignore
        ]
        self.modules_buffers = [
            buffer for (buffer, buffer_name) in named_module_buffers
        ]
        # Dict[str, tensor] representing module buffers not ignored by DDP.
        self.named_module_buffers = {
            buffer_name: buffer for (buffer, buffer_name) in named_module_buffers
        }


def _find_tensors(obj):
    r"""
    Recursively find all tensors contained in the specified object.
    """
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []

@record
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
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         with open(logpath, 'a') as file:
    #             file.write("=> loading checkpoint '{}'\n".format(args.resume))
    #         if torch.cuda.is_available() and args.local_rank:
    #             # Map model to be loaded to specified single gpu.
    #             loc = 'cuda:{}'.format(args.local_rank)
    #             checkpoint = torch.load(args.resume, map_location=loc)
    #         else:
    #             checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if args.local_rank is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.local_rank)
    #         model.load_state_dict(checkpoint['state_dict'])
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         scheduler.load_state_dict(checkpoint['scheduler'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    # initial DDP
    model = MyDDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    

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
    # os.environ['MASTER_ADDR'] = 'localhost' # '127.0.0.1'
    # os.environ['MASTER_PORT'] = '8095' # '8090'
    os.environ['NCCL_DEBUG'] = 'INFO'
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'DETAIL'
    torch.set_printoptions(threshold=1, edgeitems=1, linewidth=100000)
    main()