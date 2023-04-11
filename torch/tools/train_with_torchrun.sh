#!/bin/bash

torchrun --standalone --nproc_per_node=gpu ../multigpu_torchrun.py 10 10 --train_anns /hdd/datasets/ODOR-v3/instances_train.json --train_imgs /hdd/datasets/ODOR-v3/imgs


#torchrun --standalone ../multigpu_torchrun.py 20 10 --load_model_pth 'snapshot_10.pth'

