from typing import Optional, Callable

import torch
from torchvision.datasets.coco import CocoDetection
from torchvision import transforms as T


class ODORDataset(CocoDetection):

    def __init__(
            self,
            img_root: str,
            annFile: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            transforms: Optional[Callable] = T.ToTensor()
    ) -> None:
        super(ODORDataset, self).__init__(img_root, annFile, transforms, transform, target_transform)

    def num_classes(self):
        return len(self.coco.categories)

    def __getitem__(self, item):
        image, tgt = super().__getitem__(item)
        tgts_filtered = []
        for t in tgt:
            tgts_filtered.append({
                'id': t['id'],
                'image_id': t['image_id'],
                'category_id': t['category_id'],
                'bbox': torch.tensor(t['bbox']),
                'area': t['area'],
                'iscrowd': t['iscrowd'],
                'rotation': t['rotation'],
                'occluded': t['occluded']
            })
        return image, tgts_filtered
