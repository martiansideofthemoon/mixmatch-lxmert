#!/bin/sh
#SBATCH --job-name=lxmert_nlvr2_{job_id}
#SBATCH -o /mnt/nfs/work1/miyyer/kalpesh/projects/mixmatch/lxmert/logs/log_nlvr2_{job_id}.txt
#SBATCH --time=167:00:00
#SBATCH --partition=2080ti-long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=45GB
#SBATCH -d singleton

# Experiment Details :- {top_details}
# Run Details :- {lower_details}
# The name of this experiment.


cd /mnt/nfs/work1/miyyer/kalpesh/projects/mixmatch/lxmert

source lxmert-venv/bin/activate

name=nlvr2_{job_id}

# Save logs and models under snap/nlvr2; Make backup.
output=saved_models/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp schedulers/schedule_nlvr2_{job_id}.sh $output/run.bash

# See run/Readme.md for option details.
PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/nlvr2.py \
    --train train --valid valid \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERT snap/pretrained/model \
    --batchSize 24 --optim bert --lr 5e-5 --epochs 10 \
    --tqdm --output $output {extra_flags}

