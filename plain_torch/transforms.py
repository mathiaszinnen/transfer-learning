"""
Based on https://github.com/pytorch/vision/blob/292117405af46e1cdbf2f9ec6eb4752d276bbbb6/references/detection/transforms.py
Wrapping all used transforms to have image and target parameters for now. There should be a more elegant solution though.
"""

from typing import Optional, Dict, Tuple

from torch import nn, Tensor
from torchvision.transforms import functional as F, transforms as T
import torch

#
# class ApplyOnImage(nn.Module):
#     def __init__(self, transform):
#         super().__init__()
#         self.transform = transform
#
#     def __call__(self, image, target):
#         return self.transform(image), target



class RandomHorizontalFlip(T.RandomHorizontalFlip):
    def __call__(self, img, target):
        img = super().forward(img)
        return img, target


class RandomGrayscale(T.RandomGrayscale):
    def __call__(self, img, target):
        img = super().forward(img)
        return img, target


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
    def __init__(self, max_size):
        self.max_size = max_size


    def __call__(self, image, target):
        image = F.resize(image, [self.max_size, self.max_size])
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


def get_train_transforms(hflip_prob=.5, gs_prob=.1, size=400):
    return Compose([
        RandomHorizontalFlip(p=hflip_prob),
        RandomGrayscale(p=gs_prob),
        ConvertCOCOTargets(),
        PILToTensor(),
        ResizeImg(size),
        ConvertImageDtype(torch.float),
    ])


def get_test_transforms(size=400):
    return Compose([
        ConvertCOCOTargets(),
        PILToTensor(),
        ResizeImg(size),
        ConvertImageDtype(torch.float),
    ])
