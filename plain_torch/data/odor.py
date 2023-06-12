import torchvision
import pycocotools
import numpy as np
import torch
from utils import show_debug_img


class CocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.coco = pycocotools.coco.COCO(ann_file)

    def __getitem__(self, idx):
        img, targets = super().__getitem__(idx)
        w,h = img.size
        labels = [t['category_id'] for t in targets]
        boxes_with_labels = [t['bbox'] + [t['category_id']] for t in targets]
        for box in boxes_with_labels:
            if box[0] + box[2] > w or box[0] < 0:
                print('box will break')
                print(box)
                print(w)
            if box[1] + box[3] > h or box[1] < 0:
                print('box will break')
                print(box)
                print(h)
        img = np.array(img)
        #pre_tfm = show_debug_img(img, target) # for debugging
        if self._transforms is not None:
            tfmd = self._transforms(image=img, bboxes=boxes_with_labels)
            img = tfmd['image']
            boxes = [t[:4] for t in tfmd['bboxes']]
            labels = [t[4] for t in tfmd['bboxes']]

        boxes = torchvision.ops.box_convert(torch.tensor(boxes), in_fmt='xywh', out_fmt='xyxy')
        target = {
            "image_id": torch.tensor(self.ids[idx]),
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            # "area": [t['area'] for t in targets],
            # "iscrowd": [t['iscrowd'] for t in targets]
        }
        #post_tfm = show_debug_img(img, target) # for debugging
        return img, target

    @property
    def num_classes(self):
        return len(self.coco.cats)


def get_dataset(img_folder, ann_file, transforms):
    return CocoDataset(img_folder, ann_file, transforms=transforms)
