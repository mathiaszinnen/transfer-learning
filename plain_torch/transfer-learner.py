import copy

import torch.optim.adamw
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class TransferRCNN:
    """
    Wrapper class for fine-tuning a model
    """
    def __init__(self, model):
        self._model = model
        self._optimizer = None

    def freeze_layers(self, n_layers):
        params = []
        for param in self._model.named_parameters():
            params.append(param)

        frozen_params = []
        if n_layers != 0:
            # freeze conv1 + bn1
            for param in list(self._model.parameters())[:3]:
                frozen_params.append(param)
            # freeze n layers
            for n in list(range(n_layers+1)):
                layer_params = [p for p in self._model.named_parameters() if f'layer{n}' in p[0]]
                fpn_params = [p for p in self._model.named_parameters() if f'blocks.{n}' in p[0]]
                frozen_params.extend(layer_params)
                frozen_params.extend(fpn_params)

        self._optimizer = torch.optim.adamw.AdamW([p for p in params if p not in frozen_params], lr=lr, weight_decay=weight_decay)

    def finetune(self, dataset, epochs, freeze_epochs=10, n_freeze_layers=3):
        """
        Fine-tunes a faster R-CNN model using dataset
        :param dataset: dataset to fine-tune on
        :param epochs: number of fine-tuning epochs
        :param freeze_epochs: number of epochs to freeze backbone before fine-tuning the whole model
        :param freeze_layers: number of frozen layers (default -1 is all backbone layers)
        :return:
        """

        num_classes = dataset.num_classes

        in_features = self._model.roi_heads.box_predictor.cls_score.in_features

        self.freeze_layers(n_freeze_layers)

        self._model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        print(f"Start training with {n_freeze_layers} layers frozen")

        for epochs in range(freeze_epochs):
            train_one_epoch(model, optimizer, data_loader, device, epoch, args.print_freq, scaler)
            lr_scheduler.step()
            if args.output_dir:
                checkpoint = {
                    "model": model_without_ddp.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "args": args,
                    "epoch": epoch,
                }
                if args.amp:
                    checkpoint["scaler"] = scaler.state_dict()
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, f"model_{epoch}.pth"))
                utils.save_on_master(checkpoint, os.path.join(args.output_dir, "checkpoint.pth"))

            # evaluate after every epoch
            evaluate(model, data_loader_test, device=device)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f"Training time {total_time_str}")
        return

    def pretrain(self, dataset, epochs):
        pass

    def evaluate(self, dataset):
        pass


if __name__ == '__main__':
    cocov2_model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    transferLearner = TransferRCNN(cocov2_model)

    transferLearner.freeze_layers(2)
