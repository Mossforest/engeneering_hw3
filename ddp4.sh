#!/bin/bash

conda init
source ~/.bashrc
conda activate py39

python -m torch.distributed.launch --nproc_per_node 4 main_ddp.py -n ddp
