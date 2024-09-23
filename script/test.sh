#!/bin/bash
#SBATCH -J train                      # 作业名
#SBATCH -o log/%j.out                       # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                   # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=4                     # 单任务使用的 CPU 核心数为 4
#SBATCH --gres=gpu:geforce_rtx_2080_ti:1 
#SBATCH -t 6:00:00
source ~/.bashrc

eval "$(conda shell.bash hook)"
conda activate pet
as_path="model/MWP/as/output/flan-t5-base/as_sum_base/lr_1e-4/wp_0.08/best_bleu"
es_path="model/MWP/es/output/flan-t5-base/es_sum_base/lr_1e-4/wp_0.08/best_bleu"
save_path="data/MWP/1.4/42/test/sum_base_lr1e4_wp008_train.csv"

wandb online
cmd="python -m run.test \
    --as_path $as_path \
    --es_path $es_path \
    --save_path $save_path"

echo "Running command: $cmd"
eval "$cmd"