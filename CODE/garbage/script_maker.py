import os

def create_script(memory, scratch, py_file_name, model, method, dataset_type, others=""):
    job_name = f"{model}_{method}_{dataset_type}"
    filename = f"{job_name}.sh"
    content = f"#!/bin/bash\n#PBS -q gpu@meta-pbs.metacentrum.cz\n#PBS -l walltime=24:0:0\n#PBS -l select=1:ncpus=1:ngpus=1:mem={memory}gb:scratch_local={scratch}gb{others}\n#PBS -N {job_name}\nDATADIR=/storage/brno2/home/stepanb2\nmodule add py-pip/21.3.1-gcc-10.2.1-mjt74tn\npip install nvidia-nccl-cu12==2.19.3\npip install nvidia-cublas-cu12\npip install nvidia-cudnn-cu12==8.9.2.26\npip install nvidia-curand-cu12==10.3.2.106\npip install nvidia-cusolver-cu12==11.4.5.107\npip install nvidia-cuda-nvrtc-cu12==12.1.105\npip install nvidia-cufft-cu12==11.0.2.54\npip install nvidia-cublas-cu12==12.1.3.1\npip install nvidia-cusparse-cu12==12.1.0.106\npip install triton==2.2.0\npip install torch\npip install transformers\npip install argparse\npython $DATADIR/{py_file_name}.py --model '{model}' --model_path $DATADIR/trained_models --generation_method '{method}' --dataset_path $DATADIR --dataset_type {dataset_type}  >> $DATADIR/Out_file_{job_name}.out\nclean_scratch"

    with open(os.path.join("scripts", filename), "w") as f:
        f.write(content)


for model in ["OSCAR_GPT2", "VUT_GPT2", "TINYLLAMA", "VUT_TINYLLAMA"]:
    memory = 45
    scratch = 100
    others = ":cl_galdor=True:brno=True"
    if model == "OSCAR_GPT2":
        memory = 15
        scratch = 30

    for dataset_type in [0,1,2,3,4,5,6,7,8,9,10]:
        create_script(memory=memory, scratch=scratch, py_file_name="train_model", model=model, method="lines", dataset_type=dataset_type, others=others)


    for dataset_type in [0,1,3,5,9,11,12,13]:
        create_script(memory=memory, scratch=scratch, py_file_name="train_model", model=model, method="whole", dataset_type=dataset_type, others=others)

