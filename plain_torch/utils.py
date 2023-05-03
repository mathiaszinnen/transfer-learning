import json
import os
from typing import List, Dict

import torchvision
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import Dataset, DistributedSampler, SequentialSampler, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch
from torchvision.ops import box_area, box_convert


def prepare_dataloader(dataset: Dataset, batch_size: int, is_distributed: bool):
    if is_distributed:
        sampler = DistributedSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
        collate_fn=lambda batch: tuple(zip(*batch))
    )


def load_model(num_classes, lr):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    return model, optimizer


def outputs_to_device(outputs: List[Dict[str, torch.Tensor]], device: torch.device):
    for dict_element in outputs:
        for v in dict_element.values():
            v.to(device)
    return outputs


def coco_anns_from_preds(preds, targets, id_offset):
    anns = []
    for pred, tgt in zip(preds, targets):
        for box, label, score in zip(pred['boxes'], pred['labels'], pred['scores']):
            id_offset += 1
            box = torch.unsqueeze(box, 0)
            area = box_area(box)
            coco_box = box_convert(box, 'xyxy', 'xywh').type(torch.int).squeeze()
            # todo: do we have to sort out w, h = 0 here?
            ann = {
                'id': id_offset,
                'image_id': tgt['image_id'].item(),
                'bbox': coco_box.tolist(),
                'area': area.item(),
                'category_id': label.item(),
                'score': score.item()
            }
            anns.append(ann)
    return anns


def write_preds(pth, tgt_coco, preds):
    coco_out = {
        'images': list(tgt_coco.imgs.values()),
        'annotations': preds,
        'categories': list(tgt_coco.cats.values())
    }
    dirname = os.path.dirname(pth)
    if dirname != '':
        os.makedirs(os.path.dirname(pth), exist_ok=True)
    with open(pth, 'w') as f:
        json.dump(coco_out, f)


def coco_eval(preds: List, targets: COCO):
    tmp_path = 'tmp.json'
    with open(tmp_path, 'w') as f:
        json.dump(preds, f)
    targets.loadRes(tmp_path)
    cocoeval = COCOeval(cocoGt=targets, iouType='bbox')
    cocoeval.evaluate()
    cocoeval.accumulate()
    cocoeval.summarize()
    os.remove(tmp_path)