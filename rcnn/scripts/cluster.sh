#!/bin/bash

#SBATCH --job-name=none-odor
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/logs/transfer-%x-%j-on-%N.out
#SBATCH -e /home/%u/logs/transfer-%x-%j-on-%N.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=5:00:00

source /net/cluster/zinnen/miniconda/etc/profile.d/conda.sh
conda activate artworks-object-detection


export FINETUNE_PTH=/net/cluster/shared_dataset/ODOR/public/images
# when possible transfer data from /cluster to /scratch, /scratch are located in SSD and data transfer would be faster.
# Please exchange the --your_name-- with your name
TMPDIR=/scratch/ODOR/
if [ -d "$TMPDIR" ]; then
	echo "Data already on scratch"
else
	mkdir $TMPDIR
	echo "Copying data to '$TMPDIR'.."
	cp -r $FINETUNE_PTH $TMPDIR
fi

N_RUN=$1

PRETRAIN_MODEL=/net/cluster/zinnen/models/openimages_100ep-final-0.pth
NAME=ia_oi-${N_RUN}
IMGS=$TMPDIR/images
TRAIN_COCO=/net/cluster/shared_dataset/ODOR/public/annotations_trainvalid.json
VALID_COCO=/net/cluster/shared_dataset/ODOR/public/annotations_valid.json
CHECKPOINT=../$NAME
BATCH_SIZE=32
LR=0.001
TRAIN_EPOCHS=100
FREEZE_EPOCHS=10

python ../train.py \
--name $NAME \
--imgs $IMGS \
--train_coco $TRAIN_COCO \
--valid_coco $VALID_COCO \
--batch_size $BATCH_SIZE \
--lr $LR \
--train_epochs $TRAIN_EPOCHS \
--freeze_epochs $FREEZE_EPOCHS \
--save_checkpoint $CHECKPOINT

echo "MODEL TRAINED"

TEST_COCO=/net/cluster/shared_dataset/ODOR/private/annotations_test.json
TEST_IMGS=/net/cluster/shared_dataset/ODOR/private/images

python ../test.py \
--imgs $TEST_IMGS \
--test_coco $TEST_COCO \
--load_checkpoint $CHECKPOINT \
--batch_size $BATCH_SIZE \
| tee ../results/${NAME}_results.txt

echo "MODEL EVALUATED"
