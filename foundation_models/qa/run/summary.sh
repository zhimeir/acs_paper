#!/bin/bash
#$ -N summ
##$ -q gpu.q
#$ -l m_mem_free=50G
#$ -j y  
#$ -o job_output/$JOB_NAME-$JOB_ID-$TASK_ID.log
##$ -t 1-3

module load python/3.12.3
source $HOME/venv3115/bin/activate

model="llama-2-13b-chat-hf"
data='triviaqa'

bsize=20
bnum=385
calN=500
trainN=500
repN=5

python3 -u ../_summary.py --batch_num $bnum --batch_size $bsize --data $data --model $model --cal_size $calN --train_size $trainN
