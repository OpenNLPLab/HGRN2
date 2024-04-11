import os
import time

DATASET = "IMNET"

##### tiny
GPUS = 8
ALL = 1024
BATCH = ALL // GPUS
LRS = [0.00075]
decays = [0.05]
clip_grads = [5]
warmups = [20]
archs = ["hgrn2_vit_tiny"]

# ##### small
# GPUS = 16
# ALL = 2048
# BATCH = ALL // GPUS
# LRS = [0.00075]
# decays = [0.1]
# clip_grads = [5]
# warmups = [10]
# archs = ["hgrn2_vit_small"]

for i, arch in enumerate(archs):
    for decay in decays:
        for LR in LRS:
            for warmup in warmups:
                for clip_grad in clip_grads:
                    print(decay, LR, arch, warmup, clip_grad, ALL)
                    time.sleep(10)
                    pid = os.fork()
                    if pid == 0:
                        print(arch)
                        os.system(f'bash script_im.sh {GPUS} {BATCH} {arch} \
                                    {arch}_{DATASET}_{ALL}_{LR}_{decay}_warmup_{warmup}_cg_{clip_grad} {LR} {DATASET} \
                                    {decay} {warmup} {clip_grad}')
                    else:
                        os.wait()