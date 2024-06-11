import torch
import wandb
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import Callback

# or
# from wandb.integration.lightning.fabric import WandbLogger


class SlotAttentionLogger(Callback):
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx
    ):
        if batch_idx == 0:
            n_samples = 10
            # recon_combinde.shape = [batch_size, num_channels, width, height], recons.shape = [batch_size, num_slots, width, height, num_channels], masks.shape = [batch_size, num_slots, width, height, 1]
            recons_combined, recons, masks, slots, _, _ = pl_module.model(batch['image'])
            recons_combined = recons_combined[:n_samples].clip(-1, 1)
            gts = batch['image'][:n_samples].clip(-1, 1)
            # reshape num_slots to width
            recons = recons[:n_samples]
            recons = torch.cat([recons[:, i] for i in range(recons.shape[1])], dim=2)
            recons = recons.permute(0, 3, 1, 2).clip(-1, 1)
            masks = masks[:n_samples]
            masks = torch.cat([masks[:, i] for i in range(masks.shape[1])], dim=2)
            masks = masks.permute(0, 3, 1, 2)
            trainer.logger.experiment.log(
                {   
                    "gt": [wandb.Image(gts[i], caption="gt") for i in range(n_samples)],
                    "recon_combined": [wandb.Image(recon_combined, caption="recon_combined") for recon_combined in recons_combined],
                    "recons": [wandb.Image(recon, caption="recons") for recon in recons],
                    "masks": [wandb.Image(mask, caption="masks") for mask in masks],
                }
            )