#!/bin/sh
#SBATCH --job-name=lxmert_vqa_{job_id}
#SBATCH -o /mnt/nfs/work1/miyyer/kalpesh/projects/mixmatch/lxmert/logs/log_vqa_{job_id}.txt
#SBATCH --time=167:00:00
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=45GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}

cd /mnt/nfs/work1/miyyer/kalpesh/projects/mixmatch/lxmert

source lxmert-venv/bin/activate

# The name of this experiment.
name=vqa_{job_id}

output=saved_models/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp schedulers/schedule_vqa_{job_id}.sh $output/run.bash

# See Readme.md for option details.
PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/vqa.py \
    --train train,nominival --valid minival  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA snap/pretrained/model \
    --batchSize 32 --optim bert --lr 5e-5 --epochs 4 \
    --tqdm --output $output {extra_flags}
