Example code to convert root files made by icemc to pandas dataframes.

Files:
- convert_root_to_pandas.py: python code which loops over events in root files
   SimulatedAnitaTruthFile and SimulatedAnitaEventFile.
   write out root files to directories specified by driver code below.
   Output root files contain:
   - MC truth for the neutrino in each event
   - volts array for each event
   - channel IDs for each event
   - times array fpr each event
 
==> Example input directory with root files:
       /fs/ess/PAS2159/neutrino/signal_fixed/root/
==> Example output directory with pandas dataframs files:
       /fs/ess/PAS2159/neutrino/signal_fixed/dataframe_converted/
       ==> Note output format is pkl or pickle.   Read these using
       pandas read_pickle method.   Pickle is used because the output is
       compressed, and because numpy arrays are more easily stored in
       pkl format versus csv format.
       
- run_convert.sh: script for running the above python.  Not
   called directly, but instead called by non_qsub_many.sh or qsub_many.sh.

- non_qsub_many.sh: loops over calls to run_convert.sh.  Does this
   in the current terminal (does not submit to batch system).
   Edit to chaneg parameters, then type ate the command prompt:
      source non_qsub_many.sh

- qsub_many.sh: loops over calls to run_convert.sh, and submits
   each call as a single job to the batch system using qsub.
   Edit to chaneg parameters, then type ate the command prompt:
      source qsub_many.sh

