#!/bin/bash
#$ -N cxr-summ
##$ -q gpu.q
#$ -l m_mem_free=50G
#$ -j y  
#$ -o job_output/$JOB_NAME-$JOB_ID-$TASK_ID.log
##$ -t 1-4

module load python/3.11.5
source $HOME/venv3115/bin/activate

model="trained"
data="cxr"

bsize=100
bnum=100

calN=1000
trainN=1000

python3 -u ../_summary.py --batch_num $bnum --batch_size $bsize --data $data --model $model --cal_size $calN --train_size $trainN