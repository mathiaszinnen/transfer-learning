#!/bin/bash

torchrun --standalone --nproc_per_node=gpu ../multigpu_torchrun.py 10 10


#torchrun --standalone ../multigpu_torchrun.py 20 10 --load_model_pth 'snapshot_10.pth'

