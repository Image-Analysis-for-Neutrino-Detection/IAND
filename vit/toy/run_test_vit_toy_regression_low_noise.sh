#!/bin/bash
#PBS -A PAS2159
#PBS -l walltime=0:10:00
#PBS -l nodes=1:ppn=10:gpus=1
#PBS -j oe
# uncomment if using qsub
#cd $PBS_O_WORKDIR
#echo $PBS_O_WORKDIR
cd /users/PAS1043/osu7903/work/vit/toy
source /users/PAS1043/osu7903/work/vit/venv/bin/activate
module load cuda/11.8.0
python -u train_vit_regression.py \
            --train_dir 'regression/low_noise/train/' \
            --test_dir 'regression/low_noise/test/' \
            >& logs/test_vit_toy_regression_low_noise.log
