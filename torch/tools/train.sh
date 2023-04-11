#!/bin/bash

python ../multigpu_torchrun.py 10 10 --train_imgs /hdd/datasets/ODOR-v3/imgs --train_anns /hdd/datasets/ODOR-v3/instances_train.json


#torchrun --standalone ../multigpu_torchrun.py 20 10 --load_model_pth 'snapshot_10.pth'

