#!/usr/bin/bash
cd /mnt/vepfs/ML/ml-users/dingwen/code/DistillBEV

export PYTHONPATH=$PYTHONPATH:/mnt/vepfs/ML/ml-users/dingwen/code/DistillBEV

logdir=/mnt/vepfs/ML/ml-users/dingwen/outputs/lidar_to_bevformer_r50
mkdir -p ${logdir}

TORCH_DISTRIBUTED_DEBUG=DETAIL python -m torch.distributed.launch --nnodes=$MLP_WORKER_NUM --nproc_per_node=$MLP_WORKER_GPU --node_rank=$MLP_ROLE_INDEX --master_addr=$MLP_WORKER_0_HOST --master_port=$MLP_WORKER_0_PORT \
       	tools/train.py configs/lidar2camera_bev_distillation/teacher_to_bevformer/lidarformer_to_bevformer_nus_1x1conv_r50.py \
        --cfg-options checkpoint_config.interval=4 \
        model.distill_params.fg_feat_loss_weights=[5e-3,] \
        model.distill_params.bg_feat_loss_weights=[4e-3,] \
        model.distill_params.spatial_loss_weights=[5e-4,] \
        --work-dir ${logdir} --launcher pytorch --deterministic