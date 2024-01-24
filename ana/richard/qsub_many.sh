#!/bin/bash

START=1
END=20
export INPUT_DIR='/fs/ess/PAS2159/neutrino/signal_fixed/dataframe_converted/'
export OUTPUT_DIR='/fs/ess/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/'
for ((i=START;i<=END;i++))
do
    RUN_NUMBER=$i
    echo 'Run: '$RUN_NUMBER
    qsub -v INPUT_DIR=$INPUT_DIR,OUTPUT_DIR=$OUTPUT_DIR,RUN_NUMBER=$RUN_NUMBER run_make_image.sh
done


