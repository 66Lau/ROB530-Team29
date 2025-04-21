import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.modules.cnn1d import CNN1dEstimator

class VAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 output_dims, 
                 hidden_dims, 
                 latent_dim = 16,
                 history_len = 20,
                 feet_contact_dim = 3,
                 prior_mu=None):
        """
        Args:
            input_dim (int): Input dimension.
            hidden_dims (list of int): List of hidden layer dimensions.
            latent_dim (int): Dimension of the latent space.
        """
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.output_dims = output_dims
        
        # Encoder
        # encoder_layers = []
        # prev_dim = input_dim
        # for h_dim in hidden_dims:
        #     encoder_layers.append(nn.Linear(prev_dim, h_dim))
        #     encoder_layers.append(nn.ReLU())
        #     prev_dim = h_dim
        # self.encoder = nn.Sequential(*encoder_layers)

        self.encoder = CNN1dEstimator(nn.ReLU(), int(input_dim//history_len), history_len, hidden_dims[-1])
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(prev_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(hidden_dims[0], output_dims))
        # decoder_layers.append(nn.Sigmoid())  # For normalized outputs
        self.decoder = nn.Sequential(*decoder_layers)

        # Feet contact decoder
        FC_decoder_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            FC_decoder_layers.append(nn.Linear(prev_dim, h_dim))
            FC_decoder_layers.append(nn.ReLU())
            prev_dim = h_dim
        FC_decoder_layers.append(nn.Linear(hidden_dims[0], feet_contact_dim))
        FC_decoder_layers.append(nn.Sigmoid())
        self.FC_decoder = nn.Sequential(*FC_decoder_layers)


        self.prior_mu = prior_mu


    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) using N(0,1).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def get_representation(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        return self.decoder(z), self.FC_decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x, recon_contact= self.decode(z)
        return recon_x, recon_contact, mu, logvar

    def loss_function(self, recon_x, x, recon_contact, contact, mu, logvar):
        """
        Compute the VAE loss function.

        Args:
            recon_x: Reconstructed input.
            x: Original input.
            mu: Mean from the encoder's latent space.
            logvar: Log variance from the encoder's latent space.

        Returns:
            Total loss, reconstruction loss, and KL divergence.
        """
        recon_loss = F.mse_loss(recon_x, x, reduction='none')
        recon_loss = recon_loss.sum(dim=1) # shape[batch]

        recon_contact_loss = F.binary_cross_entropy(recon_contact, contact, reduction='none')
        recon_contact_loss = recon_contact_loss.sum(dim=1)

        kl_div = -0.5 * torch.sum(1 + logvar - ((mu - self.prior_mu) ** 2) - logvar.exp(), dim=1)

        total_loss = recon_loss + kl_div * 0.01 + recon_contact_loss# Shape: [batch]
        
        return total_loss, recon_loss.detach(), recon_contact_loss.detach(), kl_div.detach()