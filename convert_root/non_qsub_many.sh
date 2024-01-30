
#!/bin/bash

START=1
END=100
export INPUT_DIR='/fs/ess/PAS2159/neutrino/signal_fixed/root/'
export OUTPUT_DIR='/fs/ess/PAS2159/neutrino/signal_fixed/dataframe_converted/'
for ((i=START;i<=END;i++))
do
    export RUN_NUMBER=$i
    export INPUT_DIR='/fs/scratch/PAS2159/neutrino/signal_fixed/root/'$i'/'
    export OUTPUT_DIR='/fs/scratch/PAS2159/neutrino/signal_fixed/dataframe_converted/'
    echo 'Run: '$RUN_NUMBER
    source run_convert.sh 
done


