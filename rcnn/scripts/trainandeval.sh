
NAME=skript
IMGS=/net/cluster/shared_dataset/ODOR/public/images
TRAIN_COCO=/net/cluster/shared_dataset/ODOR/public/annotations_train.json
VALID_COCO=/net/cluster/shared_dataset/ODOR/public/annotations_valid.json
CHECKPOINT=skript
BATCH_SIZE=32
LR=0.001
TRAIN_EPOCHS=1
FREEZE_EPOCHS=1

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
--batch_size $BATCH_SIZE

echo "MODEL EVALUATED"
