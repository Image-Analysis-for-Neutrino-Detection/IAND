#!/bin/bash
#PBS -A PAS2159
#PBS -l walltime=0:10:00
#PBS -l nodes=1:ppn=10:gpus=1
#PBS -j oe
# uncomment if using qsub
source $HOME/.bashrc_old
source $HOME/work/IAND/venv/bin/activate
echo "got here"
#
# Get to the right directory
cd $HOME/work/IAND/ana/richard/vit
echo "got here pwd"
pwd
module load cuda/11.8.0
echo "start python"
python -u train_vit_regression.py \
            --train_dir '/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/images/' \
            --label_type 'THETA' \
            >& logs/test_vit_neutrino_THETA.log
