# IAND
Image Analysis for Neutrino Detection

# Setup
Create a virtual environment.

1) Do this once:
   python --version
      >>> should be 3.9.X o
      >>> if not and on osc, do this at command line:
             module load python/3.9-2022.05
   python -m venv venv  
   source venv/bin/activate (or venv/bin/activate.csh)  
   pip install -r requirements.txt

2) Afterwards, just have to do at command line:
    source venv/bin/activate (or venv/bin/activate.csh)
    
    OR if in vscode, select the python envrionment
         venv/bin/python

Directory Structure:
1) convert_root: Code for going from icemc root files to simple panadas dataframe.   Has its own README.
2) correlator: Placeholder for correlator code.
3) vit: Code containing examples on how to run Vision Transformers.
4) ana: Split into subdirectories by user.   Right now has these: richard, jack, michael.   (Kaeli feel free to add yours oif you like!).   The /richard directory has some simple code to analyze the
pnadas outfrom from "convert_root" and make simple images.   Has aa README.

