import torch
from torch import nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, mode='vanilla', beta=None):
        super(VAE, self).__init__()
        self.mode = mode
        self.beta = beta

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # Encoder layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder layers
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        h = F.relu(self.fc2(z))
        recon_x = torch.sigmoid(self.fc3(h))
        return recon_x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        if self.mode == 'vanilla':
            return self.vanilla_loss(recon_x, x, mu, logvar)
        elif self.mode == 'beta':
            return self.beta_loss(recon_x, x, mu, logvar)
        elif self.mode == 'mmd':
            return self.mmd_loss(recon_x, x, mu, logvar)
        else:
            print('Mode unknown')

    ### Loss functions
    def vanilla_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + kl_loss

    def beta_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        kl_loss = self.beta * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
        return recon_loss + kl_loss

    def mmd_loss(self, recon_x, x, mu, logvar):
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        mmd_loss = self.max_mean_discrapency(mu, std)
        return recon_loss + mmd_loss
    
    ### Helper functions for computing the MMD loss
    def gaussian_kernel(self, z, z_prior):
        dim1_1, dim1_2 = z.shape[0], z_prior.shape[0]
        depth = z.shape[2]
        z = z.view(dim1_1, 1, depth)
        z_prior = z_prior.view(1, dim1_2, depth)
        z_core = z.expand(dim1_1, dim1_2, depth)
        z_prior_core = z_prior.expand(dim1_1, dim1_2, depth)
        numerator = (z_core - z_prior_core).pow(2).mean(2)/depth
        return torch.exp(-numerator)

    def max_mean_discrapency(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps*logvar
        z_prior = torch.randn_like(z).to(mu.get_device())
        return self.gaussian_kernel(z, z).mean() + self.gaussian_kernel(z_prior, z_prior).mean() - 2*self.gaussian_kernel(z, z_prior).mean()
