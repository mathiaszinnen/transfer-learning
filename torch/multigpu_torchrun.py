"""
Based on https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series
"""
import math
import wandb
import sys

import torch
import torch.nn.functional as F
import torchvision.models.detection
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from data.odor_dataset import get_dataset
from data.example_dataset import MyTrainDataset

import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os
from os.path import splitext
import transforms
from tqdm import tqdm


def ddp_setup():
    init_process_group(backend="nccl")


def is_distributed():
    """
    Ugly hack to check if script is started with torchrun or directly
    :return: true if distributed launch
    """
    return os.environ.get("MASTER_ADDR") is not None


class Trainer:
    def __init__(
            self,
            model: torch.nn.Module,
            train_data: DataLoader,
            optimizer: torch.optim.Optimizer,
            save_every: int,
            save_model_pth: str,
            load_model_pth: str,
            log_interval: int,
    ) -> None:
        if is_distributed():
            self.gpu_id = int(os.environ["LOCAL_RANK"])
        else:
            self.gpu_id = 0
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.save_model_pth = save_model_pth
        self.log_interval = log_interval
        if load_model_pth is not None:
            print(f"Loading snapshot from {load_model_pth}")
            self._load_snapshot(load_model_pth)

        if is_distributed():
            self.model = DDP(self.model, device_ids=[self.gpu_id])

    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        loss_dict = self.model(source, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_dict['all'] = losses

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        return loss_dict

    def _run_epoch(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        if is_distributed():
            self.train_data.sampler.set_epoch(epoch)
        for batch_n, (source, targets) in enumerate(self.train_data):
            iteration = batch_n * b_sz
            source = [img.to(self.gpu_id) for img in source]
            targets = [{k: v.to(self.gpu_id) for k, v in t.items()} for t in targets]
            loss_dict = self._run_batch(source, targets)
            if batch_n % self.log_interval == 0:
                wandb.log(loss_dict)
                # print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {iteration} | "
                #       f"Loss: {loss_dict['all']:.4f} | "
                #       f"CLS loss: {loss_dict['loss_classifier']:.4f} | "
                #       f"BOX loss: {loss_dict['loss_box_reg']:.4f} | "
                #       f"OBJ loss: {loss_dict['loss_objectness']:.4f} | "
                #       f"RPN loss: {loss_dict['loss_rpn_box_reg']:.4f}")

    def _save_snapshot(self, epoch):
        if self.gpu_id != 0:
            return
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        save_pth = f'{splitext(self.save_model_pth)[0]}_{epoch}.pth'
        torch.save(snapshot, save_pth)
        print(f"Epoch {epoch} | Training snapshot saved at {save_pth}")

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run + 1, max_epochs + 1):
            self._run_epoch(epoch)
            if epoch % self.save_every == 0:
                self._save_snapshot(epoch)
        self._save_snapshot(epoch)


def load_model(num_classes, lr):
    # model = torch.nn.Linear(20, 1)  # load your model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    return model, optimizer


def prepare_dataloader(dataset: Dataset, batch_size: int):
    if is_distributed():
        sampler = DistributedSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=sampler,
        collate_fn=lambda batch: tuple(zip(*batch))
    )


def wandb_setup():
    wandb.login()
    wandb.init(project='Transfer-Learning')


def get_train_transforms():
    return transforms.Compose([
        transforms.ConvertCOCOTargets(),
        transforms.PILToTensor(),
        transforms.ResizeImg(),
        transforms.ConvertImageDtype(torch.float)
    ])


def main(save_every: int, total_epochs: int, batch_size: int, train_imgs, train_anns,
         output_model_pth, load_model_pth, log_interval, lr):
    if is_distributed():
        ddp_setup()
    wandb_setup()
    train_ts = get_train_transforms()
    dataset = get_dataset(train_imgs, train_anns, train_ts)
    print(f"Dataset with {len(dataset)} instances loaded")
    model, optimizer = load_model(dataset.num_classes, lr)
    train_data = prepare_dataloader(dataset, batch_size)
    trainer = Trainer(model, train_data, optimizer, save_every, output_model_pth, load_model_pth, log_interval)
    trainer.train(total_epochs)
    if is_distributed():
        destroy_process_group()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='simple distributed training job')
    parser.add_argument('total_epochs', type=int, help='Total epochs to train the model')
    parser.add_argument('save_every', type=int, help='How often to save a snapshot')
    parser.add_argument('--batch_size', default=1, type=int, help='Input batch size on each device (default: 32)')
    parser.add_argument('--train_imgs', default=None, type=str, help='Path to folder containing training images')
    parser.add_argument('--train_anns', default=None, type=str, help='Path to training annotations file')
    parser.add_argument('--output_model_pth', default='snapshot.pth', type=str, help='Where to save the trained model')
    parser.add_argument('--load_model_pth', default=None, type=str, help='Path to checkpoint to continue training from')
    parser.add_argument('--log_interval', default=1, type=int, help='Log losses every N batches')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate.')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size, args.train_imgs, args.train_anns,
         args.output_model_pth, args.load_model_pth, args.log_interval, args.lr)
