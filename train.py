# training slot attention (SA_module) with lightning and wandb logger, using SlotAttentionLogger callback, saving best validation loss model.
import argparse

import torch
import wandb
import lightning as L
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor


from slot_attention_base import SA_module
from callbacks import SlotAttentionLogger
from data import CLEVR


def main():
    parser = argparse.ArgumentParser(description='Slot Attention')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_slots', type=int, default=11)
    parser.add_argument('--num_iterations', type=int, default=3)
    parser.add_argument('--hid_dim', type=int, default=64)
    parser.add_argument('--max_objs', type=int, default=10)
    parser.add_argument('--name', type=str, default='default')
    parser.add_argument('--steps', type=int, default=400000)
    args = parser.parse_args()

    wandb_logger = WandbLogger(project='slot attention', name=args.name, log_model='all')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="max")
    train_data = CLEVR(split='train', max_objs=args.max_objs)
    val_data = CLEVR(split='val', max_objs=args.max_objs)
    model = SA_module(
        resolution=(128, 128),
        num_slots=args.num_slots,
        num_iterations=args.num_iterations,
        hid_dim=args.hid_dim,
        batch_size=args.batch_size,
        train_data=train_data,
        val_data=val_data,
        desired_steps=args.steps,
    )
    trainer = Trainer(
        max_epochs=model.n_epochs,
        logger=wandb_logger,
        callbacks=[SlotAttentionLogger(), checkpoint_callback, LearningRateMonitor(logging_interval='step')],
    )
    trainer.fit(model)
    wandb.finish()


if __name__ == '__main__':
    L.seed_everything(42)
    torch.set_float32_matmul_precision('medium')
    main()
