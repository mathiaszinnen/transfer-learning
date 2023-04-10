from typing import Tuple, Optional, Dict

import torch
import torchvision
from torchvision.transforms import functional as F
from torch import Tensor, nn


class ConvertImageDtype(nn.Module):
    def __init__(self, dtype: torch.dtype) -> None:
        super().__init__()
        self.dtype = dtype

    def forward(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class PILToTensor(nn.Module):
    def __call__(
        self, image: Tensor, target: Optional[Dict[str, Tensor]] = None
    ) -> Tuple[Tensor, Optional[Dict[str, Tensor]]]:
        image = F.pil_to_tensor(image)
        return image, target


class ResizeImg(nn.Module):
    def __call__(self, image, target):
        image = F.resize(image, [400, 400])
        return image, target


class ConvertCOCOTargets(nn.Module):
    """
    Transform to convert COCO targets from COCODetection dataset format to format
    required by torchvisionFasterRCNN models
    based on https://github.com/pytorch/vision/blob/5b07d6c9c6c14cf88fc545415d63021456874744/references/detection/coco_utils.py#L47
    """

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["image_id"] = image_id

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


class CocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target

    @property
    def num_classes(self):
        return len(self.coco.cats)


def get_dataset(img_folder, ann_file, transforms=None):
    t = [ConvertCOCOTargets(), PILToTensor(), ResizeImg(), ConvertImageDtype(torch.float)]

    if transforms is not None:
        t.append(transforms)
    transforms = Compose(t)

    return CocoDataset(img_folder, ann_file, transforms=transforms)