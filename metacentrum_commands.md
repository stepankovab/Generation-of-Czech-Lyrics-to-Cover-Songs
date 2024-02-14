
# Metacentrum commands


get a place to run things with a gpu
```sh
qsub -I -l select=1:ncpus=1:ngpus=1:mem=11gb:scratch_local=11gb -l walltime=24:0:0 -q gpu
```

get to my brno2 folder
```sh
cd ../../../brno2/home/stepanb2
```

load newest python with pip
```sh
module add py-pip/21.3.1-gcc-10.2.1-mjt74tn
```




