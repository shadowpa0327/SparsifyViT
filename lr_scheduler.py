# --------------------------------------------------------
# TinyViT Learning rate scheduler
# Copyright (c) 2022 Microsoft
# Based on the code: Swin Transformer
#   (https://github.com/microsoft/swin-transformer)
# --------------------------------------------------------

import torch
from timm.scheduler.cosine_lr import CosineLRScheduler



def build_scheduler(args, optimizer, n_iter_per_epoch):
    num_steps = int(args.epochs * n_iter_per_epoch)
    warmup_steps = int(args.warmup_epochs * n_iter_per_epoch)
    # = int(
    #    config.TRAIN.LR_SCHEDULER.DECAY_EPOCHS * n_iter_per_epoch)

    lr_scheduler = None
    if args.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=num_steps,
            lr_min=args.min_lr,
            warmup_lr_init=args.warmup_lr,
            warmup_t=warmup_steps,
            cycle_limit=1,
            t_in_epochs=False,
        )
    else:
        raise NotImplementedError(f"scheduler {args.sched} is not supported, only support cosine for short")
    return lr_scheduler

