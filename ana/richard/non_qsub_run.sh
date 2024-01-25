#!/bin/bash

START=1
END=1
export INPUT_DIR='/fs/ess/PAS2159/neutrino/signal_fixed/dataframe_converted/'
export OUTPUT_DIR='/fs/ess/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/'
for ((i=START;i<=END;i++))
do
    export RUN_NUMBER=$i
    echo 'Run: '$RUN_NUMBER
    source run_make_image_df.sh 
done
