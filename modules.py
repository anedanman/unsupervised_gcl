import torch
from torch import nn
import lightning as L

from sa_autoencoder import SlotAttentionAutoEncoder, SlotAttentionPropPredAE
from utils import hungarian_huber_loss, average_precision_clevr


class SA_module(L.LightningModule):
    def __init__(self, resolution=(128, 128), num_slots=11, num_iterations=3, hid_dim=64, batch_size=64, train_data=[], val_data=[], desired_steps=300000):
        super().__init__()
        self.model = SlotAttentionAutoEncoder(resolution, num_slots, num_iterations, hid_dim)
        self.loss = nn.MSELoss()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.hid_dim = hid_dim

        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.steps_per_epoch = len(self.train_data) // self.batch_size
        self.desired_steps = desired_steps
        self.n_epochs = self.desired_steps // self.steps_per_epoch + 1

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        recon_combined, _, _, _ = self.model(batch['image'])
        loss = self.loss(recon_combined, batch['image'])
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [{"scheduler": scheduler, "interval": 'step'}]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=8)

    def validation_step(self, batch, batch_idx):
        recon_combined, _, _, _ = self.model(batch['image'])
        loss = self.loss(recon_combined, batch['image'])
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss


class SA_PAE_module(L.LightningModule):
    def __init__(self, resolution=(128, 128), num_slots=11, num_iterations=3, hid_dim=64, batch_size=64, train_data=[], val_data=[], desired_steps=300000, num_components=50, use_global_concepts=True, classical_update=False, scale=0.1):
        super().__init__()
        self.model = SlotAttentionPropPredAE(resolution, num_slots, num_iterations, hid_dim, num_components, use_global_concepts, classical_update, scale)
        self.loss = nn.MSELoss()
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations
        self.hid_dim = hid_dim

        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size
        self.steps_per_epoch = len(self.train_data) // self.batch_size
        self.desired_steps = desired_steps
        self.n_epochs = self.desired_steps // self.steps_per_epoch + 1
        
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)
    
    def step(self, batch, batch_idx, mode='train'):
        images = batch['image']
        targets = batch['target']
        recon_combined, recons, masks, slots, res, log_likelihood = self.forward(images)
        hung_loss = hungarian_huber_loss(res, targets)
        rec_loss = self.loss(recon_combined, images)
        loss = rec_loss + hung_loss - log_likelihood

        metrics = {
            mode+'_loss': loss,
            mode+'_hungarian huber loss': hung_loss,
            mode+'_reconstruction loss': rec_loss,
            mode+'_log_likelihood': log_likelihood
            }
        ap_metrics = {}
        if batch_idx == 1:
            ap_metrics = {
                f'ap thr={thr}': average_precision_clevr(
                    res.detach().cpu().numpy(), 
                    targets.detach().cpu().numpy(), 
                    thr
                    )
                for thr in self.model.thrs
            }

        return metrics, ap_metrics
    
    def training_step(self, batch, batch_idx):
        metrics, ap_metrics = self.step(batch, batch_idx)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        self.log_dict(ap_metrics, on_step=False, on_epoch=True)
        return metrics['train_loss']

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], [{"scheduler": scheduler, "interval": 'step'}]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_data, batch_size=self.batch_size, num_workers=8)

    def validation_step(self, batch, batch_idx):
        metrics, ap_metrics = self.step(batch, batch_idx, mode='val')
        self.log_dict(metrics, on_step=False, on_epoch=True)
        self.log_dict(ap_metrics, on_step=False, on_epoch=True)
        return metrics['val_loss']