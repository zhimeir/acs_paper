#!/bin/bash
#$ -N ada_cxr
##$ -q gpu.q
#$ -l m_mem_free=30G
#$ -j y  
#$ -o job_output/$JOB_NAME-$JOB_ID-$TASK_ID.log
#$ -t 1-6

module load python/3.11.5
source $HOME/venv3115/bin/activate

repN=200

python3 ../fdr_ada.py --task_id $SGE_TASK_ID  --repN $repN 