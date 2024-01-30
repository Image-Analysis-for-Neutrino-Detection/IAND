#!/bin/bash

START=3
END=100
export INPUT_DIR='/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/'
export OUTPUT_DIR='/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/'
for ((i=START;i<=END;i++))
do
    RUN_NUMBER=$i
    echo 'Run: '$RUN_NUMBER
    qsub -v INPUT_DIR=$INPUT_DIR,OUTPUT_DIR=$OUTPUT_DIR,RUN_NUMBER=$RUN_NUMBER run_make_image_df.sh
done


