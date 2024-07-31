#!/usr/bin/env bash

export NCCL_LL_THRESHOLD=0
# export PYTHONPATH=/mnt/lustre/share/pymc/new:$PYTHONPATH

PROG=path_to_main.py # eg, im/classification/main.py
DATA=path_to_imagenet1k


GPUS=$1
batch_size=$2
ARCH=$3
PORT=$(( $RANDOM + 2000 ))
export MASTER_PORT=${MASTER_PORT:-$PORT}
LR=$5
DATASET=$6

echo $PORT

OUTPUT_DIR=./checkpoints/$4
RESUME=./checkpoints/$4/checkpoint_best.pth


mkdir -p logs
START_TIME=`date +%Y%m%d-%H:%M:%S`

torchrun --standalone --nproc_per_node=$GPUS $PROG \
    --data-set $DATASET --data-path $DATA \
    --batch-size $batch_size --dist-eval --output_dir $OUTPUT_DIR \
    --resume $RESUME --model $ARCH --epochs 300 --lr $LR \
    --weight-decay $7 \
    --warmup-epochs $8 \
    --clip-grad ${9} \
    --use_mcloader \
    --broadcast_buffers | tee logs/${START_TIME}_${ARCH}.log