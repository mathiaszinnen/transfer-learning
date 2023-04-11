import torchvision


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


def get_dataset(img_folder, ann_file, transforms):
    return CocoDataset(img_folder, ann_file, transforms=transforms)
