from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class CustomFRCNN(FasterRCNN):
    def eager_outputs(self, losses, detections):
        if self.training:
            return losses
        else:
            return losses, detections


def fasterrcnn_resnet50_fpn(num_classes, trainable_layers=3):
    """Create imagenet1k-pretrained Faster R-CNN"""
    backbone = resnet_fpn_backbone('resnet50', weights=ResNet50_Weights.DEFAULT, trainable_layers=trainable_layers)
    model = CustomFRCNN(backbone, num_classes=num_classes)
    return model
