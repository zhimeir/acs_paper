#!/bin/bash
#$ -N cxr-infer
##$ -q gpu.q
#$ -l m_mem_free=50G
#$ -j y  
#$ -o job_output/$JOB_NAME-$JOB_ID-$TASK_ID.log
#$ -t 1-100

module load python/3.11.5
source $HOME/venv3115/bin/activate

model="trained"
data="cxr"

bsize=100

python3 -u ../dataeval/load_run.py --idx $SGE_TASK_ID --batch_size $bsize --data $data --model $model