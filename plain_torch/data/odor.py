import torchvision
import pycocotools
import numpy as np
import torch
from utils import show_debug_img


class CocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        # self.coco = pycocotools.coco.COCO(ann_file)

    def __getitem__(self, idx):
        try:
            img, targets = super().__getitem__(idx)
            w, h = img.size
            labels = [t['category_id'] for t in targets]
            boxes_with_labels = [t['bbox'] + [t['category_id']] for t in targets]
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
            }
            #post_tfm = show_debug_img(img, target) # for debugging
            return img, target
        except Exception as e:
            failing_img = self.coco.imgs[idx]
            print(f'Could not load img: {failing_img}')
            raise e

    @property
    def num_classes(self):
        return len(self.coco.cats)


def get_dataset(img_folder, ann_file, transforms):
    return CocoDataset(img_folder, ann_file, transforms=transforms)

