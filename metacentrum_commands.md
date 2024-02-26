
# Metacentrum commands


get a place to run things with a good gpu
```sh
qsub -I -l walltime=24:0:0 -q gpu@meta-pbs.metacentrum.cz -l select=1:ncpus=1:ngpus=1:mem=45gb:scratch_local=100gb:gpu_cap=cuda86:cl_galdor=True:brno=True 
```

load newest python with pip
```sh
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
```

install torch
```sh
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
```

renew connection
```sh
kauth
```



