
import os


def create_script(memory, scratch, py_file_name, model, method, dataset_type, others=""):
    job_name = f"{model}_{method}_{dataset_type}"
    filename = f"{job_name}.sh"
    content = f"""#!/bin/bash
#PBS -q gpu@meta-pbs.metacentrum.cz
#PBS -l walltime=24:0:0
#PBS -l select=1:ncpus=1:ngpus=1:mem={memory}gb:scratch_local={scratch}gb{others}
#PBS -N {job_name}

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

python $DATADIR/{py_file_name}.py --model "{model}" --model_path $DATADIR/trained_models --generation_method "{method}" --dataset_path $DATADIR --dataset_type {dataset_type}  >> $DATADIR/Out_file_{job_name}.out 

clean_scratch
"""

    with open(os.path.join("CODE/LLMExperiments/scripts", filename), "w", encoding="utf-8") as f:
        f.write(content)


create_script(memory=10, scratch=30, py_file_name="train_model", model="OSCAR_GPT2", method="whole", dataset_type="0")



# others = ":gpu_cap=cuda86"