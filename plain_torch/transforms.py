import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch.nn as nn
import torch
import cv2


def get_train_transforms(size=400):
    return A.Compose([
        # A.RandomCrop(width=size, height=size),
        A.HorizontalFlip(p=0.5),
        A.ToGray(p=0.5),
        A.ToFloat(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco'))

def get_test_transforms(size=400):
    return A.Compose([
        # A.RandomCrop(width=size, height=size),
        # A.HorizontalFlip(p=0.5),
        # A.ToGray(p=0.5),
        A.ToFloat(),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='coco'))

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


# def get_train_transforms(hflip_prob=.5, gs_prob=.1, size=400):
#     return A.Compose([
#         RandomHorizontalFlip(p=hflip_prob),
#         RandomGrayscale(p=gs_prob),
#         ConvertCOCOTargets(),
#         PILToTensor(),
#         ResizeImg(size),
#         ConvertImageDtype(torch.float),
#     ])


# def get_test_transforms(size=400):
#     return Compose([
#         ConvertCOCOTargets(),
#         PILToTensor(),
#         ResizeImg(size),
#         ConvertImageDtype(torch.float),
#     ])
