#!/bin/bash
#PBS -A PAS2159
#PBS -l walltime=2:00:00
#PBS -l nodes=1:ppn=20:gpus=1
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
python -u train_vit_regression_pandas.py \
            --INPUT_DIR '/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/' \
            --label_type 'PHI' \
            --output_model_path '' \
            --vit_dim $vit_dim \
            --vit_depth $vit_depth\
            --vit_heads $vit_heads\
            --vit_mlp_dim $vit_mlp_dim\
            --vit_dropout $vit_dropout\
            --vit_emb_dropout $vit_emb_dropout\
            >& 'logs/'$tag'.log'
