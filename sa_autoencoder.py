import torch
import torch.nn as nn
import torch.nn.functional as F

from slot_attention_base import Encoder, Decoder, SlotAttention
from gmm import GMM


class SlotAttentionAutoEncoder(nn.Module):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim, num_components=200, use_global_concepts=True, classical_update=False, scale=0.1):
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
        self.gmm = GMM(num_components, hid_dim, use_classical=classical_update, scale=scale)
        if use_global_concepts:
            self.slot_upd = self.gmm.forward
        else:
            self.slot_upd = self._slot_upd

    def img2slots(self, image):
        # `image` has shape: [batch_size, num_channels, width, height].
        # Convolutional encoder with position embedding.
        x = self.encoder_cnn(image)  # CNN Backbone.
        x = self.norm(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)  # Feedforward network on set.
        # `x` has shape: [batch_size, width*height, input_size].
        slots_ = self.slot_attention(x)
        return slots_
        
    def forward(self, image):
        slots_ = self.img2slots(image)

        # resample slots with GMM
        slots, _, log_likelihood = self.slot_upd(slots_)

        # `slots` has shape: [batch_size, num_slots, slot_size].
        recon_combined, recons, masks, slots = self.proc_slots(slots, image.shape[0])
        return recon_combined, recons, masks, slots_, log_likelihood
    
    def proc_slots(self, slots_, batch_size):
        slots = slots_.reshape((-1, slots_.shape[-1])).unsqueeze(1).unsqueeze(2)
        slots = slots.repeat((1, 8, 8, 1))
        
        # `slots` has shape: [batch_size*num_slots, width_init, height_init, slot_size].
        x = self.decoder_cnn(slots)
        # `x` has shape: [batch_size*num_slots, width, height, num_channels+1].

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(batch_size, -1, x.shape[1], x.shape[2], x.shape[3]).split([3,1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0,3,1,2)
        # `recon_combined` has shape: [batch_size, num_channels, width, height].
        return recon_combined, recons, masks, slots_
    
    def _slot_upd(self, slots_):
        return slots_, None, None
    

class SlotAttentionPropPredAE(SlotAttentionAutoEncoder):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim, num_components=200, use_global_concepts=True, classical_update=False, scale=0.1):
        super().__init__(resolution, num_slots, num_iterations, hid_dim, num_components, use_global_concepts, classical_update, scale)
        self.mlp_coords = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 3),
            nn.Sigmoid()
        )
        self.mlp_prop = nn.Sequential(
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, 19 - 3),
            #nn.Sigmoid()
        )
        self.thrs = [-1, 1, 0.5, 0.25, 0.125]
        self.smax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, image):
        recon_combined, recons, masks, slots, log_likelihood = super().forward(image)
        coords = self.mlp_coords(slots.detach())*2 - 1
        props = self.mlp_prop(slots.detach())
        props = self._proc_props(props)
        res = torch.cat([coords, props], dim=-1)
        return recon_combined, recons, masks, slots, res, log_likelihood

    def _proc_props(self, props):
        props[:, :, 0:2] = self.smax(props[:, :, 0:2].clone())
        props[:, :, 2:4] = self.smax(props[:, :, 2:4].clone())
        props[:, :, 4:7] = self.smax(props[:, :, 4:7].clone())
        props[:, :, 7:15] = self.smax(props[:, :, 7:15].clone())
        props[:, :, 15:] = self.sigmoid(props[:, :, 15:].clone()) 
        return props
