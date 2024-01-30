#!/bin/bash
#PBS -A PAS2159
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=4
#PBS -j oe
#
source $HOME/.bashrc_old
source $HOME/work/IAND/venv/bin/activate
#
# Get to the right directory
cd $HOME/work/IAND/ana/richard
#
INPUT_DIR='/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/'
OUTPUT_DIR_TRAIN='/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/images_train/'
OUTPUT_DIR_TEST='/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/images_test/'
python -u extract_images.py \
        --INPUT_DIR $INPUT_DIR \
        --OUTPUT_DIR_TRAIN $OUTPUT_DIR_TRAIN\
        --OUTPUT_DIR_TRAIN $OUTPUT_DIR_TEST\
        --TEST_PERCENT 0.1 \
        >& 'logs/extract_images.log'


