#!/bin/bash
#$ -N lm-gen
#$ -q gpu.q
#$ -l m_mem_free=30G
#$ -j y  
#$ -o job_output/$JOB_NAME-$JOB_ID-$TASK_ID.log
#$ -t 1-400

module load python/3.11.5
source $HOME/venv3115/bin/activate

model='opt-13b'
dataset='coqa'

bsize=20

python3 -m ../pipeline.generate --model $model --dataset $dataset --idx $SGE_TASK_ID --batch_size $bsize
