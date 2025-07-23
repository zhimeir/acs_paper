#!/bin/bash
#$ -N summ
##$ -q gpu.q
#$ -l m_mem_free=10G
#$ -j y  
#$ -o job_output/$JOB_NAME-$JOB_ID-$TASK_ID.log

# module load python/3.11.5
source $HOME/venv3115/bin/activate

# model="opt-13b"
# data='coqa'
# repN=10

# python3 -u _plot.py --data $data --model $model --repN $repN 

module load R/4.4.0

R CMD BATCH ../process_llm_en.R