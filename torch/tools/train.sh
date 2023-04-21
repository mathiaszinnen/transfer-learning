#!/bin/bash

python ../multigpu_torchrun.py 10 10 --train_imgs /hdd/datasets/ODOR-v3/imgs --train_anns /hdd/datasets/ODOR-v3/validation_quicksplit/instances_train.json --valid_anns /hdd/datasets/ODOR-v3/validation_quicksplit/instances_valid.json --batch_size 8


#torchrun --standalone ../multigpu_torchrun.py 20 10 --load_model_pth 'snapshot_10.pth'

