#!/bin/bash
#PBS -q default@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:ngpus=0:mem=5gb:scratch_local=30gb
#PBS -N VZ_add_translation


# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=/storage/brno2/home/stepanb2 # substitute username and path to to your real username and path

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

#loads the Gaussian's application modules, version 03
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn

pip install requests
pip install json
pip install argparse
pip install os
pip install keybert
pip install re


python $DATADIR/VZ_append_truth.py --dataset_path $DATADIR >> $DATADIR/Out_file_truth_append.out || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }


# clean the SCRATCH directory
clean_scratch