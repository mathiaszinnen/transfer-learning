import json
import os
from typing import List, Dict

import torchvision
from torch.utils.data import Dataset, DistributedSampler, SequentialSampler, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch


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


def write_preds(pth, tgt_coco, preds):
    coco_out = {
        'images': list(tgt_coco.imgs.values()),
        'annotations': preds,
        'categories': list(tgt_coco.cats.values())
    }
    os.makedirs(os.path.dirname(pth), exist_ok=True)
    with open(pth, 'w') as f:
        json.dump(coco_out, f)