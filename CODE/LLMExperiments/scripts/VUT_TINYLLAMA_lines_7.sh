#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem=45gb:scratch_local=100gb:cl_galdor=True:brno=True
#PBS -N VUT_TINYLLAMA_lines_7
DATADIR=/storage/brno2/home/stepanb2
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
pip install nvidia-nccl-cu12==2.19.3
pip install nvidia-cublas-cu12
pip install nvidia-cudnn-cu12==8.9.2.26
pip install nvidia-curand-cu12==10.3.2.106
pip install nvidia-cusolver-cu12==11.4.5.107
pip install nvidia-cuda-nvrtc-cu12==12.1.105
pip install nvidia-cufft-cu12==11.0.2.54
pip install nvidia-cublas-cu12==12.1.3.1
pip install nvidia-cusparse-cu12==12.1.0.106
pip install triton==2.2.0
pip install torch
pip install transformers
pip install argparse
python $DATADIR/train_model.py --model 'VUT_TINYLLAMA' --model_path $DATADIR/trained_models --generation_method 'lines' --dataset_path $DATADIR --dataset_type 7  >> $DATADIR/Out_file_VUT_TINYLLAMA_lines_7.out
clean_scratch