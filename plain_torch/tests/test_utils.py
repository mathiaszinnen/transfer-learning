import json

from pycocotools.coco import COCO

from plain_torch.utils import coco_eval


def test_coco_eval():
    # check if it raises an exception
    gts = COCO('test_gts.json')
    with open('test_res.json') as f:
        preds = json.load(f)
    coco_eval(preds, gts)
