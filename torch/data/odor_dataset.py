from typing import Optional, Callable

from torchvision.datasets.coco import CocoDetection


class ODORDataset(CocoDetection):

    def __init__(
        self,
        img_root: str,
        annFile: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None
    ) -> None:
        super(ODORDataset, self).__init__(img_root, annFile, transforms, transform, target_transform)

    def num_classes(self):
        return len(self.coco.categories)

