from fastai.callback.progress import CSVLogger
from fastai.callback.tracker import ReduceLROnPlateau
from icevision import parsers
from icevision.data.data_splitter import SingleSplitSplitter
from icevision import tfms
from icevision import models
from icevision.data import Dataset
from icevision.metrics import COCOMetric, COCOMetricType

import wandb
from fastai.callback.wandb import *

import torch
import argparse

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def main(args):
    train_parser = parsers.COCOBBoxParser(annotations_filepath=args.train_coco, img_dir=args.imgs)
    train_records = train_parser.parse(data_splitter=SingleSplitSplitter())[0]
    valid_parser = parsers.COCOBBoxParser(annotations_filepath=args.valid_coco, img_dir=args.imgs)
    valid_records = valid_parser.parse(data_splitter=SingleSplitSplitter())[0]

    train_tfms = tfms.A.Adapter([*tfms.A.aug_tfms(size=args.img_size, presize=768), tfms.A.Normalize()])
    valid_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(args.img_size), tfms.A.Normalize()])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    num_finetune_classes = len(train_parser.class_map)
    model_type = models.torchvision.faster_rcnn
    model = model_type.model(num_classes=num_finetune_classes)
    
    if args.load_checkpoint == 'none':
        args.load_checkpoint = None

    if args.load_checkpoint is not None:
        # determine number of classes for the old model
        pretrained_model = torch.load(args.load_checkpoint)['model']
        num_pretrain_classes = pretrained_model['roi_heads.box_predictor.cls_score.weight'].shape[0]

        # create new model
        model = model_type.model(num_classes=num_pretrain_classes)
        model.load_state_dict(pretrained_model)

        if num_pretrain_classes != num_finetune_classes:
            # throw away the old detection head to match number of new classes
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_finetune_classes)

    model.to(device)

    train_ds = Dataset(train_records, train_tfms)
    train_dl = model_type.train_dl(train_ds, batch_size=args.batch_size, num_workers=4, shuffle=True)
    valid_ds = Dataset(valid_records, valid_tfms)
    valid_dl = model_type.valid_dl(valid_ds, batch_size=args.batch_size, num_workers=4, shuffle=False)

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox)]
    learn = model_type.fastai.learner(dls=[train_dl, valid_dl], model=model, metrics=metrics, cbs=[CSVLogger(fname=f'losslogs/{args.name}_losslog.csv'),ReduceLROnPlateau(monitor='valid_loss', min_delta=0.1, patience=2)])

    learn.fine_tune(args.train_epochs, args.lr, freeze_epochs=args.freeze_epochs)

    learn.save(args.save_checkpoint)

    print(f'model saved to {args.save_checkpoint}.pth')


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--name', type=str, help='experiment name for logs')
    argparser.add_argument('--imgs', type=str, help='path to images dir (train/valid)')
    argparser.add_argument('--img_size', type=int, help='image size to resize to', default=384)
    argparser.add_argument('--train_coco', type=str, help='path to train annotations in coco format')
    argparser.add_argument('--valid_coco', type=str, help='path to validation annotations in coco format')
    argparser.add_argument('--train_epochs', type=int, help='Number of epochs to train the whole model')
    argparser.add_argument('--freeze_epochs', type=int, help='Number of epochs to train the head only')
    argparser.add_argument('--batch_size', type=int, help='Batch size')
    argparser.add_argument('--lr', type=float, help='learning rate')

    argparser.add_argument('--load_checkpoint', type=str, help='path to checkpoint to continue training from')
    argparser.add_argument('--save_checkpoint', type=str, help='path to checkpoint to save trained model to')

    args = argparser.parse_args()

    main(args)
