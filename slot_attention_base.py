import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
import lightning as L
from scipy import optimize
from torch.distributions import Normal


class GlobalConcepts(nn.Module):
    """
    This module does the following:
    1) Initializes the global concepts. Namely, it creates N Gaussian distributions with trainable parameters.
    2) Takes slots (shape of [batch_size, num_slots, slot_size]) as input and maps them to the global concepts
    using Gaussian Mixture Model.
    3) For each initial slot, samples a new slot from the global concept to which it was assigned using the
    reparameterization trick and also computes the complete likelihood of the slot assignment.
    4) Returns new slots and likelihood.
    """
    def __init__(self, num_slots, slot_size, num_components, eps=1e-8):
        super().__init__()
        self.num_slots = num_slots
        self.slot_size = slot_size
        self.num_components = num_components
        self.eps = eps

        self.mu = nn.Parameter(torch.randn(num_components, slot_size))
        self.log_sigma = nn.Parameter(torch.randn(num_components, slot_size))  # Initialize log_sigma
        self.log_prior = nn.Parameter(torch.zeros(num_components), requires_grad=False)  # Initialize log prior probabilities
        self.decay = nn.Parameter(torch.tensor(0.9), requires_grad=False)  # Decay rate for prior mixture weights

    def forward(self, slots):
        """
        Args:
        slots: torch.Tensor of shape [batch_size, num_slots, slot_size]
        Returns:
        new_slots: torch.Tensor of shape [batch_size, num_slots, slot_size]
        likelihood: torch.Tensor of shape [batch_size, num_slots]

        The module maps slots to global concepts using a Gaussian Mixture Model.
        """
        batch_size = slots.size(0)

        # Create distributions for each component
        sigma = torch.exp(self.log_sigma)  # Ensure sigma is positive
        components = Normal(self.mu, sigma)

        # Compute log probabilities for each slot under each component
        log_probs = components.log_prob(slots.unsqueeze(2)).sum(dim=-1)  # shape: [batch_size, num_slots, num_components]

        # Add log prior probabilities to the log_probs
        log_prior_probs = torch.log_softmax(self.log_prior, dim=-1)  # Ensure priors sum to 1
        # log_probs = log_probs + log_prior_probs  # shape: [batch_size, num_slots, num_components]

        # Compute responsibilities (soft assignment)
        responsibilities = torch.softmax(log_probs, dim=-1)  # shape: [batch_size, num_slots, num_components]

        # Hard assignment of slots to components
        assignments = torch.argmax(responsibilities, dim=-1)  # shape: [batch_size, num_slots]

        # Sample new slots based on hard assignments using the reparameterization trick
        z = torch.randn(batch_size, self.num_slots, self.slot_size, device=slots.device)  # Standard normal samples
        new_slots = self.mu[assignments] + sigma[assignments] * z  # shape: [batch_size, num_slots, slot_size]

        # Compute likelihood
        log_likelihood = log_probs[torch.arange(batch_size).unsqueeze(1), torch.arange(self.num_slots).unsqueeze(0), assignments]  # shape: [batch_size, num_slots]

        # Update prior mixture weights
        self.log_prior.data = self.decay * torch.softmax(self.log_prior.data, dim=-1) + (1 - self.decay) * torch.softmax(torch.sum(responsibilities, dim=(0, 1)), dim=-1)

        return new_slots, log_likelihood


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
    def __init__(self, resolution, num_slots, num_iterations, hid_dim, num_components=50, use_global_concepts=True):
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
        self.global_concepts = GlobalConcepts(num_slots, hid_dim, num_components)
        self.use_global_concepts = use_global_concepts


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
        slots_ = self.slot_attention(x)
        # `slots` has shape: [batch_size, num_slots, slot_size].
        if self.use_global_concepts:
            slots_, log_likelihood = self.global_concepts(slots_)
        # """Broadcast slot features to a 2D grid and collapse slot dimension.""".
        slots = slots_.reshape((-1, slots_.shape[-1])).unsqueeze(1).unsqueeze(2)
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

        return recon_combined, recons, masks, slots_, log_likelihood.mean()


class SlotAttentionPropPredAE(SlotAttentionAutoEncoder):
    def __init__(self, resolution, num_slots, num_iterations, hid_dim):
        super().__init__(resolution, num_slots, num_iterations, hid_dim)
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
    def __init__(self, resolution=(128, 128), num_slots=11, num_iterations=3, hid_dim=64, batch_size=64, train_data=[], val_data=[], desired_steps=300000):
        super().__init__()
        self.model = SlotAttentionPropPredAE(resolution, num_slots, num_iterations, hid_dim)
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
        loss = rec_loss + hung_loss# - log_likelihood

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
    

def hungarian_huber_loss(x, y, coord_scale=1.):
    n_objs = x.shape[1]
    pairwise_cost = F.smooth_l1_loss(torch.unsqueeze(y, -2).expand(-1, -1, n_objs, -1), torch.unsqueeze(x, -3).expand(-1, n_objs, -1, -1), reduction='none').mean(dim=-1)
    indices = np.array(list(map(optimize.linear_sum_assignment, pairwise_cost.detach().cpu().numpy())))
    transposed_indices = np.transpose(indices, axes=(0, 2, 1))
    final_costs = torch.gather(pairwise_cost, dim=-1, index=torch.LongTensor(transposed_indices).to(pairwise_cost.device))[:, :, 1]
    return final_costs.sum(dim=1).mean()


def average_precision_clevr(pred, attributes, distance_threshold):
  """Computes the average precision for CLEVR.
  This function computes the average precision of the predictions specifically
  for the CLEVR dataset. First, we sort the predictions of the model by
  confidence (highest confidence first). Then, for each prediction we check
  whether there was a corresponding object in the input image. A prediction is
  considered a true positive if the discrete features are predicted correctly
  and the predicted position is within a certain distance from the ground truth
  object.
  Args:
    pred: Tensor of shape [batch_size, num_elements, dimension] containing
      predictions. The last dimension is expected to be the confidence of the
      prediction.
    attributes: Tensor of shape [batch_size, num_elements, dimension] containing
      ground-truth object properties.
    distance_threshold: Threshold to accept match. -1 indicates no threshold.
  Returns:
    Average precision of the predictions.
  """

  pred[:, :, :3] = (pred[:, :, :3] + 1) / 2
  attributes[:, :, :3] = (attributes[:, :, :3] + 1) / 2

  [batch_size, _, element_size] = attributes.shape
  [_, predicted_elements, _] = pred.shape

  def unsorted_id_to_image(detection_id, predicted_elements):
    """Find the index of the image from the unsorted detection index."""
    return int(detection_id // predicted_elements)

  flat_size = batch_size * predicted_elements
  flat_pred = np.reshape(pred, [flat_size, element_size])
  sort_idx = np.argsort(flat_pred[:, -1], axis=0)[::-1]  # Reverse order.

  sorted_predictions = np.take_along_axis(
      flat_pred, np.expand_dims(sort_idx, axis=1), axis=0)
  idx_sorted_to_unsorted = np.take_along_axis(
      np.arange(flat_size), sort_idx, axis=0)

  def process_targets(target):
    """Unpacks the target into the CLEVR properties."""
    coords = target[:3]
    object_size = np.argmax(target[3:5])
    material = np.argmax(target[5:7])
    shape = np.argmax(target[7:10])
    color = np.argmax(target[10:18])
    real_obj = target[18]
    return coords, object_size, material, shape, color, real_obj

  true_positives = np.zeros(sorted_predictions.shape[0])
  false_positives = np.zeros(sorted_predictions.shape[0])

  detection_set = set()

  for detection_id in range(sorted_predictions.shape[0]):
    # Extract the current prediction.
    current_pred = sorted_predictions[detection_id, :]
    # Find which image the prediction belongs to. Get the unsorted index from
    # the sorted one and then apply to unsorted_id_to_image function that undoes
    # the reshape.
    original_image_idx = unsorted_id_to_image(
        idx_sorted_to_unsorted[detection_id], predicted_elements)
    # Get the ground truth image.
    gt_image = attributes[original_image_idx, :, :]

    # Initialize the maximum distance and the id of the groud-truth object that
    # was found.
    best_distance = 10000
    best_id = None

    # Unpack the prediction by taking the argmax on the discrete attributes.
    (pred_coords, pred_object_size, pred_material, pred_shape, pred_color,
     _) = process_targets(current_pred)

    # Loop through all objects in the ground-truth image to check for hits.
    for target_object_id in range(gt_image.shape[0]):
      target_object = gt_image[target_object_id, :]
      # Unpack the targets taking the argmax on the discrete attributes.
      (target_coords, target_object_size, target_material, target_shape,
       target_color, target_real_obj) = process_targets(target_object)
      # Only consider real objects as matches.
      if target_real_obj:
        # For the match to be valid all attributes need to be correctly
        # predicted.
        pred_attr = [pred_object_size, pred_material, pred_shape, pred_color]
        target_attr = [
            target_object_size, target_material, target_shape, target_color]
        match = pred_attr == target_attr
        if match:
          # If a match was found, we check if the distance is below the
          # specified threshold. Recall that we have rescaled the coordinates
          # in the dataset from [-3, 3] to [0, 1], both for `target_coords` and
          # `pred_coords`. To compare in the original scale, we thus need to
          # multiply the distance values by 6 before applying the norm.
          distance = np.linalg.norm((target_coords - pred_coords) * 3.)

          # If this is the best match we've found so far we remember it.
          if distance < best_distance:
            best_distance = distance
            best_id = target_object_id
    if best_distance < distance_threshold or distance_threshold == -1:
      # We have detected an object correctly within the distance confidence.
      # If this object was not detected before it's a true positive.
      if best_id is not None:
        if (original_image_idx, best_id) not in detection_set:
          true_positives[detection_id] = 1
          detection_set.add((original_image_idx, best_id))
        else:
          false_positives[detection_id] = 1
      else:
        false_positives[detection_id] = 1
    else:
      false_positives[detection_id] = 1
  accumulated_fp = np.cumsum(false_positives)
  accumulated_tp = np.cumsum(true_positives)
  recall_array = accumulated_tp / np.sum(attributes[:, :, -1])
  precision_array = np.divide(accumulated_tp, (accumulated_fp + accumulated_tp))

  return compute_average_precision(
      np.array(precision_array, dtype=np.float32),
      np.array(recall_array, dtype=np.float32))


def compute_average_precision(precision, recall):
  """Computation of the average precision from precision and recall arrays."""
  recall = recall.tolist()
  precision = precision.tolist()
  recall = [0] + recall + [1]
  precision = [0] + precision + [0]

  for i in range(len(precision) - 1, -0, -1):
    precision[i - 1] = max(precision[i - 1], precision[i])

  indices_recall = [
      i for i in range(len(recall) - 1) if recall[1:][i] != recall[:-1][i]
  ]

  average_precision = 0.
  for i in indices_recall:
    average_precision += precision[i + 1] * (recall[i + 1] - recall[i])
  return average_precision