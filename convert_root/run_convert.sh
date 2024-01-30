
#!/bin/bash
#PBS -A PAS2159
#PBS -l walltime=1:00:00
#PBS -l nodes=1:ppn=10
#PBS -j oe
# uncomment if using qsub
#cd $PBS_O_WORKDIR
#echo $PBS_O_WORKDIR
cd /users/PAS1043/osu7903/work/IAND/convert_root
#source /users/PAS1043/osu7903/work/vit/venv/bin/activate
which python
echo 'INPUT_DIR: '$INPUT_DIR
echo 'OUTPUT_DIR: '$OUTPUT_DIR
echo 'RUN_NUMBER: '$RUN_NUMBER
python -u convert_root_to_pandas.py \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR\
        --run_number $RUN_NUMBER \
        >& 'logs/convert_root_pandas_run_'$RUN_NUMBER'.log'

