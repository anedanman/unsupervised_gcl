import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback

# or
# from wandb.integration.lightning.fabric import WandbLogger


class SlotAttentionLogger(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, batch, batch_idx, n_samples=10
    ):
        if batch_idx == 0:
            # recon_combinde.shape = [batch_size, num_channels, width, height], recons.shape = [batch_size, num_slots, width, height, num_channels], masks.shape = [batch_size, num_slots, width, height, 1]
            recons_combined, recons, masks, slots = pl_module.model(batch)
            recons_combined = recons_combined.permute(0, 2, 3, 1)[:n_samples]
            # reshape num_slots to width
            recons = recons[:n_samples].reshape(-1, recons.shape[2] * recons.shape[1], recons.shape[3], recons.shape[4])
            masks = masks[:n_samples].reshape(-1, masks.shape[2] * masks.shape[1], masks.shape[3], masks.shape[4])
            trainer.logger.experiment.log(
                {
                    "recon_combined": [wandb.Image(recon_combined, caption="recon_combined") for recon_combined in recons_combined],
                    "recons": [wandb.Image(recon, caption="recons") for recon in recons],
                    "masks": [wandb.Image(mask, caption="masks") for mask in masks],
                }
            )