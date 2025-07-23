#!/bin/bash
#$ -N cxr-stack
##$ -q gpu.q
#$ -l m_mem_free=20G
#$ -j y  
#$ -o job_output/$JOB_NAME-$JOB_ID.log

module load python/3.11.5
source $HOME/venv3115/bin/activate


python3 -u ../generate_encode.py --num_batch 20
