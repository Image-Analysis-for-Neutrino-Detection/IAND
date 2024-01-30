#!/bin/bash
#PBS -A PAS2159
#PBS -l walltime=0:30:00
#PBS -l nodes=1:ppn=2
#PBS -j oe
#
source $HOME/.bashrc_old
source $HOME/work/IAND/venv/bin/activate
#
# Get to the right directory
cd $HOME/work/IAND/ana/richard
#

echo 'INPUT_DIR: '$INPUT_DIR
echo 'OUTPUT_DIR: '$OUTPUT_DIR
echo 'RUN_NUMBER: '$RUN_NUMBER
python -u ana_make_image_df.py \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR\
        --run_number $RUN_NUMBER \
        >& 'logs/ana_make_image_df_'$RUN_NUMBER'.log'


