#python ../test.py --load_model_path snapshot_10.pth \
#	--test_anns /hdd/datasets/annotations-nightly/snapshots/v3/instances_test.json \
#	--test_imgs /hdd/datasets/annotations-nightly/imgs \
#	--preds_pth 'preds.json'

python ../test.py --load_model_path snapshot_30.pth \
	--test_anns /hdd/datasets/annotations-nightly/snapshots/v3/validation_quicksplit/instances_train.json \
	--test_imgs /hdd/datasets/annotations-nightly/imgs \
	--preds_pth 'preds_30.json'
