import argparse
import torch
from fastai.callback.progress import CSVLogger

from icevision import parsers
from icevision.data.data_splitter import SingleSplitSplitter
from icevision import tfms
from icevision import models
from icevision.data import Dataset
from icevision.metrics import COCOMetric, COCOMetricType


def main(args):
    test_parser = parsers.COCOBBoxParser(annotations_filepath=args.test_coco, img_dir=args.imgs)
    test_records = test_parser.parse(data_splitter=SingleSplitSplitter())[0]

    test_tfms = tfms.A.Adapter([*tfms.A.resize_and_pad(args.img_size), tfms.A.Normalize()])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    num_classes = len(test_parser.class_map)
    model_type = models.torchvision.faster_rcnn
    model = model_type.model(num_classes=num_classes)
    model.eval()
    model.to(device)

    test_ds = Dataset(test_records, test_tfms)
    test_dl = model_type.valid_dl(test_ds, batch_size=args.batch_size, num_workers=4, shuffle=False)

    metrics = [COCOMetric(metric_type=COCOMetricType.bbox, print_summary=True)]
    learn = model_type.fastai.learner(dls=[test_dl], model=model, metrics=metrics)

    learn.load(args.load_checkpoint)

    learn.validate(dl=test_dl)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument('--name', type=str, help='experiment name for logs')
    argparser.add_argument('--imgs', type=str, help='path to images dir')
    argparser.add_argument('--img_size', type=int, help='image size to resize to', default=384)
    argparser.add_argument('--test_coco', type=str, help='path to testset annotations in coco format')
    argparser.add_argument('--load_checkpoint', type=str, help='path to checkpoint to continue training from')
    argparser.add_argument('--batch_size', type=int, help='batch size')

    args = argparser.parse_args()

    main(args)
