import os
from os.path import splitext

import torch
import wandb
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP


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
            is_wandb: bool,
            is_distributed: bool,
    ) -> None:
        if is_distributed:
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
        self.is_wandb = is_wandb
        self.is_distributed = is_distributed

        if is_distributed:
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
        if self.is_distributed:
            self.train_data.sampler.set_epoch(epoch)
        for batch_n, (source, targets) in enumerate(self.train_data):
            iteration = batch_n * b_sz
            source = [img.to(self.gpu_id) for img in source]
            targets = [{k: v.to(self.gpu_id) for k, v in t.items()} for t in targets]
            loss_dict = self._run_batch(source, targets)
            if batch_n % self.log_interval == 0:
                if self.is_wandb:
                    wandb.log(loss_dict)
                else:
                    print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {iteration} | "
                          f"Loss: {loss_dict['all']:.4f} | "
                          f"CLS loss: {loss_dict['loss_classifier']:.4f} | "
                          f"BOX loss: {loss_dict['loss_box_reg']:.4f} | "
                          f"OBJ loss: {loss_dict['loss_objectness']:.4f} | "
                          f"RPN loss: {loss_dict['loss_rpn_box_reg']:.4f}")

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