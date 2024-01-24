#!/bin/bash
#PBS -A PAS2159
#PBS -l walltime=0:30:00
#PBS -l nodes=1:ppn=10:gpus=1
#PBS -j oe
# uncomment if using qsub
#cd $PBS_O_WORKDIR
#echo $PBS_O_WORKDIR
cd /users/PAS1043/osu7903/work/vit/ana
source /users/PAS1043/osu7903/work/vit/venv/bin/activate
module load cuda/11.8.0
python -u test_ViT_standard.py >& logs/test_vit_standrad.log
