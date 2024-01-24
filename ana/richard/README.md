Example code to read the dataframes in:
   /fs/ess/PAS2159/neutrino/signal_fixed/dataframe_converted/

which contain:
- MC truth for the neutrino in each event
- volts array for each event
- channel IDs for each event
- times array fpr each event

Files:
- ana_make_image.py: python code which loops oever event in dataframe file
    and creates simple power in phi-channel vs time bin image.  Writes this
    out to a matching dataframe in:
   /fs/ess/PAS2159/neutrino/signal_fixed/dataframe_converted/volts_images/

- run_make_image.sh: script for running the above python.  Not
   called directly, but instead called by non_qsub_run.sh or qsub_many.sh.

- non_qsub_run.sh: loops over calls to run_make_image.sh.  Does this
   in the current terminal (does not submit to batch system).
   Edit to chaneg parameters, then type ate the command prompt:
      source non_qsub_run.sh

- qsub_many.sh: loops over calls to run_make_image.sh, and submits
   each call as a single job to the batch system using qsub.
   Edit to chaneg parameters, then type ate the command prompt:
      source qsub_many.sh

