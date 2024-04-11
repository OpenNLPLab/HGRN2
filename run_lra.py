import os
import sys
import time

# change
use_decay = True

batches = {
    "cifar": 64,
    "imdb": 64,
    "listops": 128,
    "pathfinder": 128,
    "pathfinderx": 256,
    "aan": 128,
}

gpus = {
    "cifar": 8,
    "imdb": 2,
    "listops": 8,
    "pathfinder": 8,
    "pathfinderx": 8,
    "aan": 8,
}

d_model_dict = {
    "cifar": 512,
    "imdb": [128],
    "listops": 64,
    "pathfinder": [128],
    "pathfinderx": 32,
    "aan": 256,
}

n_layers_dict = {
    "cifar": [6],
    "imdb": [4],
    "listops": [5],
    "pathfinder": 6,
    "pathfinderx": 6,
    "aan": 2,
}

norm_dict = {
    "cifar": "synbatch",
    "imdb": "layer",
    "listops": "synbatch",
    "pathfinder": "batch",
    "pathfinderx": "batch",
    "aan": "synbatch",
}

lr_dict = {
    "cifar": [3e-3], 
    "imdb": [0.005],
    "listops": [0.0005],
    "pathfinder": [2e-3],
    "pathfinderx": [0.00075],
    "aan": [0.005], 
}

wd_dict = {
    "cifar": 0,
    "imdb": [0],
    "listops": [0],
    "pathfinder": 0,
    "pathfinderx": 0,
    "aan": [0,], 
}

dropout_dict = {
    "cifar": [0],
    "imdb": 0.1,
    "listops": [0],
    "pathfinder": 0,
    "pathfinderx": 0,
    "aan": [0],
}

prenorm_dict = {
    "cifar": [True],
    "imdb": True,
    "listops": True,
    "pathfinder": [True,],
    "pathfinderx": True,
    "aan": True,
}

warmup_steps_dict = {
    "cifar": 30000,
    "imdb": [10000],
    "listops": [5000],
    "pathfinder": [50000],
    "pathfinderx": [150000],
    "aan": [312],
}

training_steps_dict = {
    "cifar": 50000,
    "imdb": 50000,
    "listops": [50000],
    "pathfinder": [500000,],
    "pathfinderx": 500000,
    "aan": [50000],
}

expand_ratio_glu_dict = {
    "cifar": 1,
    "imdb": [1],
    "listops": [1],
    "pathfinder": [1],
    "pathfinderx": 1,
    "aan": [2],
}

expand_ratio_dict = {
    "cifar": 2,
    "imdb": [8],
    "listops": [4],
    "pathfinder": [2],
    "pathfinderx": 1,
    "aan": [2],
}

training_epochs_dict = {
    "cifar": 75, #[100],
    "imdb": 40,
    "listops": [50],
    "pathfinder": [100],
    "pathfinderx": 200,
    "aan": [20],
}

use_lower_bound_dict = {
    "cifar": True,
    "imdb": True,
    "listops": True,
    "pathfinder": True,
    "pathfinderx": True,
    "aan": True,
}

encoder_dict = {
    "cifar": "position",
    "imdb": "id",
    "listops": "position",
    "pathfinder": "id",
    "pathfinderx": "position",
    "aan": "position",
}

use_series_dict = {
    "cifar": False,
    "imdb": False,
    "listops": False,
    "pathfinder": False,
    "pathfinderx": True,
    "aan": False,
}

gradient_clip_dict = {
    "cifar": 0,
    "imdb": 0,
    "listops": 0,
    "pathfinder": 0,
    "pathfinderx": 1,
    "aan": 1,
}

tasks = ["aan", "imdb", "listops"]
archs = ["hgrn2_1d"]

tasks = ["cifar", "pathfinder", "pathfinderx"]
archs = ["hgrn2_2d"]

def to_iter(*args):
    n = len(args)
    new_args = []
    for i in range(n):
        if not isinstance(args[i], list):
            arg = [args[i]]
        else:
            arg = args[i]
        new_args.append(arg)

    return helper(*new_args)


def helper(*args):
    n = len(args)
    if n == 1:
        res = [[arg] for arg in args[0]]
        return res
    else:
        arr = helper(*args[1:])
        res = []
        for par in args[0]:
            for data in arr:
                res.append([par] + list(data))
        return res

def get_name(pars):
    arr = map(str, pars)
    return "-".join(arr)

for i, task in enumerate(tasks):
    pars = to_iter(
        archs,
        n_layers_dict[task],
        d_model_dict[task],
        batches[task],
        norm_dict[task],
        lr_dict[task],
        wd_dict[task],
        dropout_dict[task],
        prenorm_dict[task],
        warmup_steps_dict[task],
        training_steps_dict[task],
        expand_ratio_glu_dict[task],
        expand_ratio_dict[task],
        training_epochs_dict[task],
        use_lower_bound_dict[task],
        encoder_dict[task],
        use_series_dict[task],
        gradient_clip_dict[task],
    )
    print(pars)
    print(task)
    print(len(pars))
    # time.sleep(10)
    j = 0
    for (
        arch,
        n_layers,
        d_model,
        total_batch,
        norm,
        lr,
        wd,
        dropout,
        prenorm,
        warmup_steps,
        training_steps,
        expand_ratio_glu,
        expand_ratio,
        training_epochs,
        use_lower_bound,
        encoder,
        use_series,
        gradient_clip,
    ) in pars:
        gpu = gpus[task]
        batch = total_batch // gpu
        workers = gpu * 20
        j += 1
        if task == "cifar":
            file = "script_lra_image"
        else:
            file = "script_lra_others"

        pid = os.fork()
        if pid != 0:
            os.system(
                f"sh {file}.sh {task} {arch} {batch} {n_layers} {d_model} {norm} {lr} {wd} {gpu} {workers} {dropout} {prenorm} {warmup_steps} {training_steps} {expand_ratio_glu} {expand_ratio} {training_epochs} {use_lower_bound} {encoder} {use_series} {gradient_clip}"
            )
        else:
            os.wait()
        