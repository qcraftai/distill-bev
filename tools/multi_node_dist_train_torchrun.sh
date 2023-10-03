#!/usr/bin/env bash

CONFIG=$1

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
torchrun \
--nproc_per_node $MLP_WORKER_GPU --master_addr $MLP_WORKER_0_HOST \
--node_rank $MLP_ROLE_INDEX --master_port $MLP_WORKER_0_PORT --nnodes $MLP_WORKER_NUM \
$(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:2}