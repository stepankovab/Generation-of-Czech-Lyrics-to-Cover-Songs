#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=45gb:scratch_local=100gb:gpu_cap=cuda86:cl_galdor=True:brno=True
#PBS -N tinyLlamaOneLine2


# define a DATADIR variable: directory where the input files are taken from and where output will be copied to
DATADIR=/storage/brno2/home/stepanb2 # substitute username and path to to your real username and path

# append a line to a file "jobs_info.txt" containing the ID of the job, the hostname of node it is run on and the path to a scratch directory
# this information helps to find a scratch directory in case the job fails and you need to remove the scratch directory manually 
echo "$PBS_JOBID is running on node `hostname -f` in a scratch directory $SCRATCHDIR" >> $DATADIR/jobs_info.txt

#loads the Gaussian's application modules, version 03
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn


# run Gaussian 03 with h2o.com as input and save the results into h2o.out file
# if the calculation ends with an error, issue error message an exit
python $DATADIR/train_one_line.py --model "tinyLlama" --dataset_path $DATADIR --dataset_type 2 >> $DATADIR/Out_file_ll_2.out || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }

# move the output to user's DATADIR or exit in case of failure

# python $DATADIR/GPT2_baseline_infer.py --model "tinyLlama" --dataset_path $DATADIR --dataset_type 5 >> $DATADIR/Out_file_ll2i.out || { echo >&2 "Calculation ended up erroneously (with a code $?) !!"; exit 3; }


# clean the SCRATCH directory
clean_scratch