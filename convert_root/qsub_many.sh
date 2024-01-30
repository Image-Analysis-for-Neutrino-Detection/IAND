
#!/bin/bash

START=1
END=2
export INPUT_DIR='/fs/scratch/PAS2159/neutrino/signal_fixed/root/'
export OUTPUT_DIR='/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/'
for ((i=START;i<=END;i++))
do
    RUN_NUMBER=$i
    echo 'Run: '$RUN_NUMBER
    qsub -v INPUT_DIR=$INPUT_DIR$i'/',OUTPUT_DIR=$OUTPUT_DIR,RUN_NUMBER=$RUN_NUMBER run_convert.sh
done


