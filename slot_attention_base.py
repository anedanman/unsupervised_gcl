import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import lightning as L

class SlotAttention(nn.Module):
    def __init__(self, num_slots, dim, iters = 3, eps = 1e-8, hidden_dim = 128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = dim ** -0.5

        self.slots_mu = nn.Parameter(torch.randn(1, 1, dim))
        self.slots_sigma = nn.Parameter(torch.rand(1, 1, dim))

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.gru = nn.GRUCell(dim, dim)

        hidden_dim = max(dim, hidden_dim)

        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

        self.norm_input  = nn.LayerNorm(dim)
        self.norm_slots  = nn.LayerNorm(dim)
        self.norm_pre_ff = nn.LayerNorm(dim)

    def forward(self, inputs, num_slots = None):
        b, n, d = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots
        
        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_sigma.expand(b, n_s, -1)
        slots = torch.distributions.Normal(mu, sigma).rsample()

        inputs = self.norm_input(inputs)        
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots = self.perform_iter(slots, k, v)
        slots = self.perform_iter(slots.detach(), k, v)
        return slots

    def perform_iter(self, slots, k, v):
        b, _, d = slots.shape
        slots_prev = slots
        slots = self.norm_slots(slots)
        q = self.to_q(slots)
        dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
        attn = dots.softmax(dim=1) + self.eps
        attn = attn / attn.sum(dim=-1, keepdim=True)
        updates = torch.einsum('bjd,bij->bid', v, attn)
        slots = self.gru(
            updates.reshape(-1, d),
            slots_prev.reshape(-1, d)
        )
        slots = slots.reshape(b, -1, d)
        slots = slots + self.fc2(F.relu(self.fc1(self.norm_pre_ff(slots))))
        return slots

def build_grid(resolution):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))

"""Adds soft positional embedding with learnable projection."""
class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        grid = build_grid(resolution).to(self.embedding.weight.device)
        self.grid = nn.Parameter(grid, requires_grad=False)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

class Encoder(nn.Module):
    def __init__(self, resolution, hid_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, hid_dim, 5, padding = 2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, 5, padding = 2)
        self.encoder_pos = SoftPositionEmbed(hid_dim, resolution)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0,2,3,1)
        x = self.encoder_pos(x)
        x = torch.flatten(x, 1, 2)
        return x

class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super().__init__()
        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(2, 2), padding=2, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, 5, stride=(1, 1), padding=2)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, 3, stride=(1, 1), padding=1)
        self.decoder_initial_size = (8, 8)
        self.decoder_pos = SoftPositionEmbed(hid_dim, self.decoder_initial_size)
        self.resolution = resolution

    def forward(self, x):
        x = self.decoder_pos(x)
        x = x.permute(0,3,1,2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:,:,:self.resolution[0], :self.resolution[1]]
        x = x.permute(0,2,3,1)
        return x

"""Slot Attention-based auto-encoder for object discovery."""
class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim):
        """Builds the Slot Attention-based auto-encoder.
        Args:
        resolution: Tuple of integers specifying width and height of input image.
        num_slots: Number of slots in Slot Attention.
        num_iterations: Number of iterations in Slot Attention.
        """
        super().__init__()
        self.hid_dim = hid_dim
        self.resolution = resolution
        self.num_slots = num_slots
        self.num_iterations = num_iterations

        self.encoder_cnn = Encoder(self.resolution, self.hid_dim)
        self.decoder_cnn = Decoder(self.hid_dim, self.resolution)

        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)

        self.slot_attention = SlotAttention(
            num_slots=self.num_slots,
            dim=hid_dim,
            iters = self.num_iterations,
            eps = 1e-8, 
            hidden_dim = 128)
        self.norm = nn.LayerNorm(hid_dim)

    def forward(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].

        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = self.norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].

        # Slot Attention module.
        slots = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].

        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots.reshape((-1, slots.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))
        
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, num_channels, width, height].

        return recon_combined, recons, masks, slots
    

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
