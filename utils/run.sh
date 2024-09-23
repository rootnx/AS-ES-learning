#!/bin/bash
#SBATCH -J en_cal                      # 作业名
#SBATCH -o log/%j.out                       # 屏幕上的输出文件重定向到 test.out
#SBATCH -p compute                            # 作业提交的分区为 compute
#SBATCH -N 1                                  # 作业申请 1 个节点
#SBATCH --ntasks-per-node=1                   # 单节点启动的进程数为 1
#SBATCH --cpus-per-task=4                     # 单任务使用的 CPU 核心数为 4
#SBATCH --gres=gpu:nvidia_rtx_a6000:1 
#SBATCH -t 3:00:00
source ~/.bashrc

eval "$(conda shell.bash hook)"
conda activate pet
model_path="model/MWP/cs/output/flan-t5-base/cs/lr_8e-5/best_bleu"

#param needed for mode 1
cs_path='data/MWP/math_icl.csv' 
piece_path='data/MWP/piece_comma_base.csv'
entropy_path='data/MWP/entropy_comma_base.csv' #cp for conditional probability
seg_name='base'
data_dir='data/MWP'
ratio=1.4

piece_entropy_path='data/MWP/1.4/1.4_entropies_sum.csv'
es_path='data/MWP/1.4/es_sum.csv'
as_path='data/MWP/1.4/as_sum.csv'

wandb offline
cmd="python -m utils.MWP.run \
    --model_type ft5 \
    --model_name $model_path \
    --cs_path $cs_path \
    --piece_path $piece_path \
    --entropy_path $entropy_path \
    --data_dir $data_dir \
    --ratio $ratio \
    --piece_entropy_path $piece_entropy_path \
    --es_path $es_path \
    --as_path $as_path \
    --seg_name $seg_name"

echo "Running command: $cmd"
eval "$cmd"

