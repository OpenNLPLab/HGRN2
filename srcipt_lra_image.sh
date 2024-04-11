export HYDRA_FULL_ERROR=1
export DATA_PATH=path_to_data_path
program_path=path_to_lra_dir

TASK=$1
ARCH=$2
BS=$3
N_LAYERS=$4
D_MODEL=$5
NORM=$6
lr=${7}
wd=${8}
cards=${9}
n_works=${10}
dropout=${11}
n_works=4
PRENORM=${12}
warmup_steps=${13}
training_steps=${14}
expand_ratio_glu=${15}
expand_ratio=${16}
training_epochs=${17}
use_lower_bound=${18}
encoder=${19}
use_series=${20}
gradient_clip=${21}

echo ${cards}


# mkdir -p logs
START_TIME=`date +%Y%m%d-%H:%M:%S`

mkdir -p logs_${TASK}

torchrun --standalone \
 --nproc_per_node=${cards} \
${program_path}/train.py wandb=null experiment=${ARCH}-lra-${TASK} \
trainer.devices=$cards \
trainer.precision=bf16 \
loader.batch_size=${BS} \
loader.num_workers=${n_works} \
scheduler.num_warmup_steps=${warmup_steps} \
scheduler.num_training_steps=${training_steps} \
optimizer.lr=${lr} optimizer.weight_decay=${wd} \
model.n_layers=${N_LAYERS} model.d_model=${D_MODEL} \
model.norm=${NORM} model.prenorm=True train.seed=2222 \
model.dropout=${dropout} \
model.expand_ratio_glu=${expand_ratio_glu} \
model.expand_ratio=${expand_ratio} \
model.use_lower_bound=${use_lower_bound} \
trainer.max_epochs=${training_epochs} \
model.encoder=${encoder} \
model.use_series=${series} \
dataset.grayscale=false \
dataset.augment=true \
decoder.mode=pool | tee logs_${TASK}/${START_TIME}_${ARCH}-lra-${TASK}.log