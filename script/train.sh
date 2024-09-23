#!/bin/bash
#SBATCH -J train                      # 作业名
#SBATCH -o log/%j.out                       # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                   # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=4                     # 单任务使用的 CPU 核心数为 4
#SBATCH --gres=gpu:nvidia_rtx_a6000:1 
#SBATCH -t 4:00:00
source ~/.bashrc

eval "$(conda shell.bash hook)"
conda activate pet
model_path="output/flan-t5-base"
lr=5e-4
warmup_rate=0.04
beta=1.4
max_length=256
batch_size=64
accumulate=1
dataset='MWP'
gen_target='as'
task_prime="sum_base"
task_name="${gen_target}_${task_prime}" # task_name is used to reflect the src_file (how data is split)
src_file_path="data/MWP/${beta}/${task_prime}/${beta}_entropies_${task_name}.csv"

wandb online
cmd="python -m run.train \
    --model_type ft5 \
    --model_name $model_path \
    --batch_size $batch_size \
    --epoch_num 50 \
    --data_dir data/$dataset/$beta \
    --save_path model/$dataset/$gen_target/$model_path/$task_name/lr_$lr/wp_$warmup_rate \
    --lr $lr \
    --gen_target $gen_target \
    --dataset $dataset \
    --src_file_path $src_file_path \
    --task_name $task_name \
    --max_length $max_length \
    --accumulate $accumulate \
    --warmup_rate $warmup_rate"

echo "Running command: $cmd"
eval "$cmd"