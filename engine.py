# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Train and eval functions used in main.py
"""
import math
import sys
import random
from typing import Iterable, Optional
from greedy_nas_utils import CandidatePool
import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma
import torch.distributed as dist
from losses import DistillationLoss
import utils
from greedy_nas_utils import TradeOffLoss
import torch

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        if args.nas_mode:
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

def train_one_epoch_greedy(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, data_loader_proxy: Iterable = None,
                    cand_pool: CandidatePool = None, num_subnets = 10, num_kept_subnet = 5,
                    epsilon = 0.0, all_choices = None, proxy_metrics: Optional[TradeOffLoss] = None):
    
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    # get the flops of uncompressed model and smallest models
    model.module.set_largest_config()
    largest_flops = model.module.flops() / 1e9
    model.module.set_smallest_config()
    smallest_flops = model.module.flops() / 1e9

    proxy_criterion = torch.nn.CrossEntropyLoss()
    
    proxy_samples = []
    proxy_targets = []
    for samples, targets in data_loader_proxy:
        proxy_samples.append(samples)
        proxy_targets.append(targets)
    
    proxy_samples = torch.cat(proxy_samples, dim = 0).to(device, non_blocking=True)
    proxy_targets = torch.cat(proxy_targets, dim = 0).to(device, non_blocking=True)
    
    #proxy_samples = torch.cat([batch] for batch_ )

    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)

        '''
        step I. 
        Sample the subnetwork using epsilon greedy rules
        '''
        # 1.A get `num_subnets` sub-networks
        #candidate_subnets = []
        candidate_subnets_with_scores = []
        for i in range(num_subnets):
            if random.random() < epsilon: # get from candidate pools
                subnet = cand_pool.get_one_subnet()
            else:
                subnet = model.module.get_random_sample_config()
            #candidate_subnets.append(subnet)
        
        #subnet_losses = [0.0 for _ in range(num_subnets)]
        # 1.B ranking the subnetworks
        #for proxy_samples, proxy_targets in data_loader_proxy:
        #    proxy_samples = proxy_samples.to(device, non_blocking=True)
        #    proxy_targets = proxy_targets.to(device, non_blocking=True)

            
        #for i, subnet in enumerate(candidate_subnets):
            with torch.no_grad():
                # set the configuration of subnetwork
                model.module.set_sample_config(subnet)
                #proxy_loss = 0.0
                # loop over proxy data loader to calculate loss
                #print(len(data_loader_proxy))
                # for proxy_samples, proxy_targets in  data_loader_proxy:
                #     proxy_samples = proxy_samples.to(device, non_blocking=True)
                #     proxy_targets = proxy_targets.to(device, non_blocking=True)

                with torch.cuda.amp.autocast():
                    output = model(proxy_samples)
                    proxy_loss = proxy_criterion(output, proxy_targets)
                proxy_loss = proxy_loss.item()
                
                
                #print(f"rank:{dist.get_rank()}, proxy_loss_before: {proxy_loss}")
                # 1.C Synchronization and calculate the metric with respect to flops
                t = torch.tensor([proxy_loss], device = 'cuda')
                dist.barrier()
                dist.all_reduce(t)
                proxy_losses  = t.tolist()[0]
                #print(f"rank:{dist.get_rank()}, proxy_loss_after: {proxy_losses}")
        #for i, (subnet, proxy_losses) in enumerate(zip(candidate_subnets, subnet_losses)):
                #model.module.set_sample_config(subnet)
                flops = model.module.flops() / 1e9 
                normalized_flops = (flops - smallest_flops) / (largest_flops - smallest_flops) # min-max normalization
                #print(normalized_flops)
                proxy_scores = proxy_metrics(proxy_losses, normalized_flops)
                #print("Proxy score:", proxy_scores)
                candidate_subnets_with_scores.append((subnet, proxy_scores))
        '''
        step II. 
        ranking the sampled subnetworks and keep the top-k 
        '''
        candidate_subnets_with_scores = sorted(candidate_subnets_with_scores, key=lambda x: x[1])
        candidate_subnets_with_scores = candidate_subnets_with_scores[:min(len(candidate_subnets_with_scores), num_kept_subnet)]
        
        '''
        step III.
        register benchmarked subnet into candidate pools, 
        the candidate will automatically maintain the subnet with promising performance
        '''
        for subnet, score in candidate_subnets_with_scores:
            cand_pool.add_one_subnet_with_score(subnet, score)
        
        
        '''
        step IV.
        train the selected subnetworks
        '''
        loss_value = 0.0        
        for subnet, _ in candidate_subnets_with_scores:
            #print("Running subnet", subnet)
            # 1. set the configuration of subnetwork
            model.module.set_sample_config(subnet)
            with torch.cuda.amp.autocast():
                outputs = model(samples, return_intermediate=(args.distillation_type == 'soft_fd'))
                loss = criterion(samples, outputs, targets)

            loss_value += loss.item()

            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)

            optimizer.zero_grad()

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss_scaler(loss, optimizer, clip_grad=max_norm,
                        parameters=model.parameters(), create_graph=is_second_order)

            torch.cuda.synchronize()
        

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}





@torch.no_grad()
def evaluate(nas_config, data_loader, model, device, args = None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    nas_test_config_list = [args.nas_test_config] if args.nas_test_config else [[1, 4], [1, 3], [2, 4], [4, 4]]
    if args.subnet:
        nas_test_config_list = [args.subnet]
    
    
    for nas_test_config in nas_test_config_list:
        print(f'Test config {nas_test_config} now!')
        if args.nas_mode:
            # Sample the subnet to test accuracy
            test_config = []
            for ratios in nas_config['sparsity']['choices']:
                if args.subnet:
                    test_config.append(ratios[0])
                else:
                    test_config.append(nas_test_config)

            model.module.set_sample_config(test_config)  

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
            metric_logger.meters[f'{nas_test_config}_loss'].update(loss.item(), n=batch_size)
            metric_logger.meters[f'{nas_test_config}_acc1'].update(acc1.item(), n=batch_size)
            # metric_logger.meters[f'{nas_test_config}_acc5'].update(acc5.item(), n=batch_size)
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        # print('{name}: * Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        #     .format(name=nas_test_config, top1=metric_logger.meters[f'{nas_test_config}_acc1'], 
        #             top5=metric_logger.meters[f'{nas_test_config}_acc5'], losses=metric_logger.meters[f'{nas_test_config}_loss']))
        print('{name}: * Acc {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
            .format(name=nas_test_config, top1=metric_logger.meters[f'{nas_test_config}_acc1'], 
                    losses=metric_logger.meters[f'{nas_test_config}_loss']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
