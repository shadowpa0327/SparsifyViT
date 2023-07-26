# Sparsity Research on ViT

This reposity contains the PyTorch training code for the original DeiT models. Currently the code base are forked from the official [DeiT repo](https://github.com/facebookresearch/deit)

Here, I have build an interface and add some naive methods for add sparsity into the ViT.


## Some warning !
- Now we can't not support non-uniform subnet (maybe find by EA) evaluate & finetuned

# Sample subnet
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch \
--master_port 29503 --nproc_per_node=2 --use_env sampling_subnet.py \
--model Sparse_deit_small_patch16_224 \
--data-path /dataset/imagenet \
--batch-size 256 \
--nas-mode \
--nas-weight 'KD_nas_124+13_150epoch_round3/[1, 3]_best_checkpoint.pth' \
--sample_num 3000 \
--output_dir sample_KD_round3_3000 \
--dist-eval \
--eval \
--wandb

## SubNet finetuned command
'''
python -m torch.distributed.launch --master_port 29515 --nproc_per_node=8 --use_env main.py \
--model Sparse_deit_small_patch16_224 \
--data-path /dev/shm/imagenet \
--epochs 50 \
--batch-size 128 \
--lr 5e-6 \
--min-lr 1e-6 \
--nas-weights KD_nas_124+13_150epoch_round2/best_checkpoint.pth \
--nas-config configs/kd_ea_7.85M.yaml \
--teacher-model deit_small_patch16_224 \
--distillation-type soft \
--distillation-alpha 1.0 \
--output_dir KD_ea_7.85M_50epoch \
--subnet kd_ea_7.85M \
--dist-eval \
--wandb
'''

## Sparsity NAS Training scripts
- Use CUDA_VISIBLE_DEVICES=0,1,2,3 to choose which GPUs to run
- Normal command
    - training
        ```
        python -m torch.distributed.launch --master_port 29510 --nproc_per_node=2 --use_env main.py \
        --model Sparse_deit_small_patch16_224 \
        --data-path /dataset/imagenet \
        --epochs 150 \
        --batch-size 128 \
        --pretrained \
        --lr 5e-5 \
        --min-lr 1e-6 \
        --nas-config configs/deit_small_nxm_uniform24.yaml \
        --nas-test-config 2 4 \
        --output_dir nas_uniform_24_150epoch \
        --dist-eval \
        --wandb
        ```
    - eval 
        ```
        python -m torch.distributed.launch --master_port 29510 --nproc_per_node=2 --use_env main.py \
        --model Sparse_deit_small_patch16_224 \
        --data-path /dataset/imagenet \
        --nas-config configs/deit_small_nxm_uniform24.yaml \
        --nas-weights nas_uniform_24_150epoch/best_checkpoint.pth \
        --nas-test-config 2 4 \
        --eval \
        --dist-eval
        ```
- KD command
    - training
        ```
        python -m torch.distributed.launch --master_port 29510 --nproc_per_node=2 --use_env main.py \
        --model Sparse_deit_small_patch16_224 \
        --data-path /dataset/imagenet \
        --epochs 150 \
        --batch-size 128 \
        --pretrained \\
        --lr 5e-5 \
        --min-lr 1e-6 \
        --nas-config configs/deit_small_nxm_nas_1234.yaml \
        --nas-test-config 2 4 \
        --output_dir KD_nas_124+13_150epoch \
        --teacher-model deit_small_patch16_224 \
        --distillation-type soft \
        --distillation-alpha 1.0 \
        --dist-eval \
        --wandb
        ```
    - eval 
        ```
        python -m torch.distributed.launch --master_port 29510 --nproc_per_node=2 --use_env main.py \
        --model Sparse_deit_small_patch16_224 \
        --data-path /dataset/imagenet \
        --nas-config configs/deit_small_nxm_uniform24.yaml \
        --nas-weights KD_nas_124+13_150epoch/checkpoint.pth \
        --nas-test-config 2 4 \
        --eval \
        --dist-eval
        ```
- Cifar-100 command
    - training
        ```
        python -m torch.distributed.launch --nporc_per_node=8 --use_env main.py \
            --model deit_small_patch16_224 \
            --batch-size 128 \
            --finetune https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth \
            --data-set CIFAR \
            --data-path /dev/shm/cifar100 \
            --opt adamw \
            --weight-decay 0.01 \
            --lr 5e-6 \
            --min-lr 1e-7 \
            --drop-path 0.05 \
            --output_dir deit_s_224_cifar_100_0629 \
            --epochs 1000
        ```






