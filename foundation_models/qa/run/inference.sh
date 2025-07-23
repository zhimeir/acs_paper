#!/bin/bash
#$ -N infer
#$ -q gpu.q
#$ -l m_mem_free=50G
#$ -j y  
#$ -o job_output/$JOB_NAME-$JOB_ID-$TASK_ID.log
#$ -t 1-400

module load python/3.11.5
source $HOME/venv3115/bin/activate

# model="opt-13b"
# data='coqa'
model='llama-2-13b-chat-hf'
data='triviaqa'

bsize=20
calN=1000
trainN=1000

python3 -u ../dataeval/load_run.py --idx $SGE_TASK_ID --batch_size $bsize --data $data --model $model