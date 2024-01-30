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
python -u apply_model_vit_pandas.py \
            --INPUT_DIR '/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/' \
            --label_type 'PHI' \
            --cpu_or_gpu 'gpu' \
            --input_model_path '/fs/ess/PAS2159/neutrino/torch_models/simple_phi_regression.pt' \
            --output_ana_path '/fs/ess/PAS2159/neutrino/ana/pd_results_dphi.csv' \
            >& logs/apply_vit_neutrino_PHI_pandas_gpu.log

