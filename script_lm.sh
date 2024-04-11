#! /usr/bin/bash

ARCH=$1
gpus=$2
DATA_DIR=$3


BATCH_SIZE=16
TOKENS_PER_SAMPLE=512
MAX_TOKEN=$((TOKENS_PER_SAMPLE*BATCH_SIZE))


prefix=lm
MAX_UPDATE=50000
WARM_UP=4000

PORT=$(( $RANDOM + 2000 ))
echo $PORT
LR=0.0005
CLIP_NORM=1.0
decay=0.1

dir_name=logs_train

mkdir -p $dir_name

START_TIME=`date +%Y%m%d-%H:%M:%S`
UPDATE_FREQ=$(( 128 / $BATCH_SIZE / $gpus ))

torchrun --standalone --nproc_per_node=$gpus \
$(which fairseq-train) --task language_modeling \
    $DATA_DIR \
    --save-dir checkpoints_train/$prefix/${ARCH} \
    --arch $ARCH --share-decoder-input-output-embed \
    --distributed-world-size $gpus \
    --save-interval-updates 50000 \
    --dropout 0.1 \
    --bf16 \
    --optimizer adam --adam-betas '(0.9, 0.98)' --weight-decay $decay --clip-norm $CLIP_NORM \
    --lr $LR --lr-scheduler inverse_sqrt --warmup-updates $WARM_UP --warmup-init-lr 1e-07 \
    --tokens-per-sample $TOKENS_PER_SAMPLE --sample-break-mode none \
    --max-tokens $MAX_TOKEN --update-freq $UPDATE_FREQ \
    --batch-size $BATCH_SIZE \
    --max-update $MAX_UPDATE --log-interval 100 2>&1 | tee $dir_name/${START_TIME}_${ARCH}.log