"""
Based on https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series
"""
import wandb

from data.odor_dataset import get_dataset
from utils import prepare_dataloader
from utils import load_model
from trainer import Trainer

from torch.distributed import init_process_group, destroy_process_group
import os
import transforms


def ddp_setup():
    init_process_group(backend="nccl")


def is_distributed():
    """
    Ugly hack to check if script is started with torchrun or directly
    :return: true if distributed launch
    """
    return os.environ.get("MASTER_ADDR") is not None


def wandb_setup():
    wandb.login()
    wandb.init(project='Transfer-Learning')


def main(save_every: int, total_epochs: int, batch_size: int, train_imgs, train_anns, valid_anns,
         output_model_pth, load_model_pth, log_interval, lr, is_wandb):
    if is_distributed():
        ddp_setup()
    if is_wandb:
        wandb_setup()
    train_ts = transforms.get_train_transforms()
    valid_ts = transforms.get_test_transforms()
    train_ds = get_dataset(train_imgs, train_anns, train_ts)
    valid_ds = get_dataset(train_imgs, valid_anns, valid_ts)
    print(f"Dataset with {len(train_ds)} instances loaded")
    model, optimizer = load_model(train_ds.num_classes, lr)
    train_data = prepare_dataloader(train_ds, batch_size, is_distributed())
    eval_data = prepare_dataloader(valid_ds, batch_size, is_distributed(), shuffle=False)
    trainer = Trainer(model, train_data, eval_data, optimizer, save_every,
                      output_model_pth, load_model_pth, log_interval,
                      is_wandb, is_distributed())
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
    parser.add_argument('--valid_anns', type=str, help='Path to validation annotations.')
    parser.add_argument('--output_model_pth', default='snapshot.pth', type=str, help='Where to save the trained model')
    parser.add_argument('--load_model_pth', default=None, type=str, help='Path to checkpoint to continue training from')
    parser.add_argument('--log_interval', default=1, type=int, help='Log losses every N batches')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate.')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    main(args.save_every, args.total_epochs, args.batch_size, args.train_imgs, args.train_anns, args.valid_anns,
         args.output_model_pth, args.load_model_pth, args.log_interval, args.lr, args.wandb)
