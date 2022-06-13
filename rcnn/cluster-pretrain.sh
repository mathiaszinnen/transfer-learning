#!/bin/bash

#SBATCH --job-name=pretrain
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12000
#SBATCH --gres=gpu:1
#SBATCH -o /home/%u/logs/transfer-%x-%j-on-%N.out
#SBATCH -e /home/%u/logs/transfer-%x-%j-on-%N.err
#SBATCH --mail-type=ALL
#Timelimit format: "hours:minutes:seconds" -- max is 24h
#SBATCH --time=24:00:00

source /net/cluster/zinnen/miniconda/etc/profile.d/conda.sh
conda activate artworks-object-detection



N_RUN=$2
PRETRAINING_DS=$1
DS_BASE=/net/cluster/shared_dataset/olfactory_objects/styled-datasets

case ${PRETRAINING_DS} in
	soi)
		TRAIN_COCO=${DS_BASE}/styled-openimages-adho/labels.json
		VALID_COCO=${DS_BASE}/styled-openimages-adho/labels.json
		IMGDIR=${DS_BASE}/styled-openimages-adho/images
		;;
	sia)
		TRAIN_COCO=${DS_BASE}/styled-iconart/labels.json
		VALID_COCO=${DS_BASE}/styled-iconart/labels.json
		IMGDIR=${DS_BASE}/styled-iconart/images
		;;
	spa)
		TRAIN_COCO=${DS_BASE}/styled-peopleart-adho/labels.json
		VALID_COCO=${DS_BASE}/styled-peopleart-adho/labels.json
		IMGDIR=${DS_BASE}/styled-peopleart-adho/images
		;;
esac

# when possible transfer data from /cluster to /scratch, /scratch are located in SSD and data transfer would be faster.
# Please exchange the --your_name-- with your name
TMPDIR=/scratch/${PRETRAINING_DS}/
if [ -d "$TMPDIR" ]; then
	echo "Data already on scratch"
else
	mkdir $TMPDIR
	echo "Copying data from ${IMGDIR} to '$TMPDIR'.."
	cp -r $IMGDIR $TMPDIR
fi


NAME=${PRETRAINING_DS}-${N_RUN}
IMGS=$TMPDIR/images
CHECKPOINT=../$NAME
BATCH_SIZE=32
LR=0.001
TRAIN_EPOCHS=100
FREEZE_EPOCHS=10

python train.py \
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

