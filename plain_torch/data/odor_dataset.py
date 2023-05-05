import torchvision
import pycocotools
from plain_torch.utils import show_debug_img


class CocoDataset(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms
        self.coco = pycocotools.coco.COCO(ann_file)

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)
        image_id = self.ids[idx]
        target = dict(image_id=image_id, annotations=target)
        #pre_tfm = show_debug_img(img, target) # for debugging
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        #post_tfm = show_debug_img(img, target) # for debugging
        return img, target

    @property
    def num_classes(self):
        return len(self.coco.cats)


def get_dataset(img_folder, ann_file, transforms):
    return CocoDataset(img_folder, ann_file, transforms=transforms)
