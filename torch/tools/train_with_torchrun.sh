#!/bin/bash

torchrun --standalone --nproc_per_node=gpu ../train.py 10 10 \
  --train_anns /hdd/datasets/ODOR-v3/validation_quicksplit/instances_train.json \
  --valid_anns /hdd/datasets/ODOR-v3/validation_quicksplit/instances_valid.json \
  --train_imgs /hdd/datasets/ODOR-v3/imgs --batch_size 8


#torchrun --standalone ../multigpu_torchrun.py 20 10 --load_model_pth 'snapshot_10.pth'

