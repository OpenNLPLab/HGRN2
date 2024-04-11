code_dir=path_to_zoology

export cache_dir=your_data_cache

mkdir -p log

START_TIME=`date +%Y%m%d-%H:%M:%S`

name=hgrn2
LOG_FILE=log/${START_TIME}-${name}.log


python -m zoology.launch ${code_dir}/zoology/experiments/hgrn/${name}.py -p 2>&1 | tee -a $LOG_FILE