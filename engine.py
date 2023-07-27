# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
import random
from typing import Iterable, Optional
from nas_utils import CandidatePool, TradeOffLoss
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import torch.distributed as dist
from losses import DistillationLoss
import utils
import torch

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, nas_mode = False,
                    args = None, lr_scheduler = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    num_steps = len(data_loader)

    for iteration, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if nas_mode:
            model.module.set_random_sample_config()
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        with torch.cuda.amp.autocast():
            outputs = model(samples, return_intermediate=(args.distillation_type == 'soft_fd'))
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        if lr_scheduler is not None:
            lr_scheduler.step_update(
                (epoch * num_steps + iteration))

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def path_filtering(model: torch.nn.Module, proxy_samples: torch.tensor, proxy_targets: torch.tensor,
                    device: torch.device, cand_pool: CandidatePool, num_subnets: int, num_kept_subnet:int, epsilon:float,
                    compression_ratio_contraint:float):
    
    model.module.set_largest_config()
    largest_flops = round(model.module.flops() / 1e9, 2)
    candidate_subnets_with_scores_and_flops = []
    proxy_criterion = torch.nn.CrossEntropyLoss()

    '''
    step I.
    Sample the subnetwork using epsilon greedy rules
    '''
    for i in range(num_subnets):
        if random.random() < epsilon: # get from candidate pools
            subnet = cand_pool.get_one_subnet()
            if subnet is None:
                subnet = model.module.get_random_sample_config()
        else:
            subnet = model.module.get_random_sample_config()

        with torch.no_grad():
            # set the configuration of subnetwork
            model.module.set_sample_config(subnet)
            # inference
            with torch.cuda.amp.autocast():
                output = model(proxy_samples)
                proxy_loss = proxy_criterion(output, proxy_targets)
            proxy_loss = proxy_loss.item()

            # Synchronization and calculate the metric with respect to flops
            t = torch.tensor([proxy_loss], device = 'cuda')
            dist.barrier()
            dist.all_reduce(t)
            proxy_losses  = t.tolist()[0]

            # calulate score
            flops = round(model.module.flops() / 1e9, 2)
            candidate_subnets_with_scores_and_flops.append((subnet, proxy_losses, flops))
    '''
    step II.
    ranking the sampled subnetworks and keep the top-k
    '''
    candidate_subnets_with_scores_and_flops = sorted(candidate_subnets_with_scores_and_flops, key=lambda x: x[1])
    candidate_subnets_with_scores_and_flops = candidate_subnets_with_scores_and_flops[:min(len(candidate_subnets_with_scores_and_flops), num_kept_subnet)]

    '''
    step III.
    register benchmarked subnet into candidate pools,
    the candidate will automatically maintain the subnet with promising performance
    '''
    for subnet, score, flops in candidate_subnets_with_scores_and_flops:
        if flops <= (compression_ratio_contraint * largest_flops):
            cand_pool.add_one_subnet_with_score_and_flops(subnet, score, flops)
    
    return [subnet for subnet, _, _ in candidate_subnets_with_scores_and_flops]


def train_one_epoch_greedy(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, data_loader_proxy: Iterable = None,
                    cand_pool: CandidatePool = None, num_subnets = 10, num_kept_subnet = 5,
                    epsilon = 0.0, all_choices = None, compression_ratio_contraint = 1.0, lr_scheduler = None):

    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('eps_prob', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    num_steps = len(data_loader)

    
    proxy_samples = []
    proxy_targets = []
    for samples, targets in data_loader_proxy:
        proxy_samples.append(samples)
        proxy_targets.append(targets)
    
    proxy_samples = torch.cat(proxy_samples, dim = 0).to(device, non_blocking=True)
    proxy_targets = torch.cat(proxy_targets, dim = 0).to(device, non_blocking=True)

    print(f"*Running greedy NAS with proxy val set with size:{proxy_samples.shape}, number of sample subnet: {num_subnets}")
    print(f"*Current candidate pool size: {cand_pool.get_size()}/{cand_pool.max_pool_size}")

    filtered_subnets_pools = [] 
    filtered_subnets_pools_update_freq = num_kept_subnet


    '''
    Training the filtered subnet, and refresh the pools every `num_kept_subnet` iterations
    '''
    for iteration, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        # resample k paths using path filtering algorithm every k iterations
        if ((iteration) % filtered_subnets_pools_update_freq) == 0:
            filtered_subnets_pools = path_filtering(model, proxy_samples, proxy_targets,
                                     device, cand_pool, num_subnets, num_kept_subnet, epsilon,
                                     compression_ratio_contraint) 
        subnet = filtered_subnets_pools[(iteration % filtered_subnets_pools_update_freq)]

        # set the configuration of subnetwork
        model.module.set_sample_config(subnet)
        with torch.cuda.amp.autocast():
            outputs = model(samples, return_intermediate=(args.distillation_type == 'soft_fd'))
            loss = criterion(samples, outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        if lr_scheduler is not None:
            lr_scheduler.step_update(
                (epoch * num_steps + iteration))

        metric_logger.update(eps_prob=epsilon)
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}





@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        # metric_logger.update(loss=loss.item())
        metric_logger.meters['loss'].update(loss.item(), n=batch_size)
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        # metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()

    print('Acc {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(top1=metric_logger.meters['acc1'], losses=metric_logger.meters['loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
