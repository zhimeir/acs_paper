#!/bin/bash
#$ -N cxr-gen
##$ -q gpu.q
#$ -l m_mem_free=40G
#$ -j y  
#$ -o job_output/$JOB_NAME-$JOB_ID-$TASK_ID.log
#$ -t 1-20

module load python/3.11.5
source $HOME/venv3115/bin/activate

bsize=500

python3 -m ../generate --idx $SGE_TASK_ID --batch_size $bsize
