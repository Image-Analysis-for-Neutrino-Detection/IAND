
echo 'INPUT_DIR: '$INPUT_DIR
echo 'OUTPUT_DIR: '$OUTPUT_DIR
echo 'RUN_NUMBER: '$RUN_NUMBER
python -u convert_root_to_pandas.py \
        --input_dir $INPUT_DIR \
        --output_dir $OUTPUT_DIR\
        --run_number $RUN_NUMBER \
        >& 'logs/convert_root_pandas_run_'$RUN_NUMBER'.log'

