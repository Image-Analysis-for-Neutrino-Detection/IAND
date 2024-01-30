#!/bin/bash
#PBS -A PAS2159
#PBS -l walltime=2:00:00
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
            --train_path '/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/images_train/images_train_all.tar' \
            --test_path '/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/images_test/images_test_all.tar' \
            --label_type 'PHI' \
            --output_model_path '/fs/ess/PAS2159/neutrino/torch_models/simple_phi_regression.pt' \
            >& logs/test_vit_neutrino_PHI_tar.log
