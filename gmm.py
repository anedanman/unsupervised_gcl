import torch
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F


class GMM(nn.Module):
    def __init__(self, num_components, input_dim, scale=0.1, use_classical=False):
        super(GMM, self).__init__()
        self.num_components = num_components
        self.input_dim = input_dim
        self.means = nn.Parameter(torch.randn(num_components, input_dim)) 
        self.log_stds = nn.Parameter(torch.randn(num_components, input_dim)* 0.5)
        self.weights = nn.Parameter(torch.ones(num_components))
        self.scale = scale
        if use_classical:
            self.update_func = self.classical_update
        else:
            self.update_func = self.neural_update

    def forward(self, input):
        idxs, log_likelihood = self.update_func(input, self.means, self.log_stds, self.weights)
        resampled_x = self.sample(idxs, self.means, self.log_stds)
        return resampled_x, idxs, log_likelihood
    
    def neural_update(self, input, means, log_stds, weights):
        prior_mix = D.Categorical(logits=weights)
        components = D.Independent(D.Normal(means, log_stds.exp()), 1)
        gmm = D.MixtureSameFamily(prior_mix, components)    

        # probability of each input belonging to each cluster
        log_p_x_z = gmm.component_distribution.log_prob(input.unsqueeze(1))
        # prior probability of each cluster
        log_p_z = torch.log_softmax(
            gmm.mixture_distribution.logits, dim=-1
        )
        # posterior probability of each cluster given the input
        p_z_x = F.softmax(log_p_x_z + log_p_z + 1e-8
                          , dim=-1)
        # hard assignment
        idxs = torch.argmax(p_z_x, dim=-1)
        # log likelihood
        Q = (p_z_x.detach() * (log_p_x_z + log_p_z)).mean()
        return idxs, Q
    
    def sample(self, idxs, means, log_stds):
        resampled_x = means[idxs] + log_stds[idxs].exp() * torch.randn_like(means[idxs]) * self.scale
        return resampled_x
    
    def classical_update(self, input, *args, **kwargs):
        stds = torch.clamp(self.log_stds.exp(), max=1e6)
        components = D.Independent(D.Normal(self.means, stds), 1)
        
        prior_mix = D.Categorical(logits=self.weights)
        gmm = D.MixtureSameFamily(prior_mix, components)
        
        log_p_x_z = gmm.component_distribution.log_prob(input.unsqueeze(1))
        log_p_z = torch.log_softmax(gmm.mixture_distribution.logits, dim=-1)
        
        p_z_x = F.softmax(log_p_x_z + log_p_z, dim=-1)
        
        idxs = torch.argmax(p_z_x, dim=-1)
        new_priors = p_z_x.mean(0)
        
        # Avoid division by zero
        denominator = p_z_x.sum(0).unsqueeze(-1) + 1e-9
        new_means = (p_z_x.unsqueeze(-1) * input.unsqueeze(1)).sum(0) / denominator
        
        # Compute new stds safely
        variance = (p_z_x.unsqueeze(-1) * (input.unsqueeze(1) - new_means.unsqueeze(0)).pow(2)).sum(0) / denominator
        new_stds = torch.sqrt(torch.clamp(variance, min=1e-9))
        
        Q = (p_z_x.detach() * (log_p_x_z + log_p_z)).mean()
        
        # Update parameters safely
        self.means.data = new_means
        self.log_stds.data = torch.log(torch.clamp(new_stds, min=1e-9))
        self.weights.data = torch.log(torch.clamp(new_priors, min=1e-9))
        return idxs, Q