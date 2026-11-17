import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    def __init__(self, action_dim=131, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, out_dim)

        self.fc4 = nn.Linear(out_dim, 32)
        self.fc5 = nn.Linear(32, 64)
        self.fc6 = nn.Linear(64, action_dim)

        self.relu = nn.ReLU()

    def encode(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

    def decode(self, z):
        x = self.relu(self.fc4(z))
        x = self.relu(self.fc5(x))
        return torch.tanh(self.fc6(x))

    def forward(self, x):
        return self.decode(self.encode(x))


class VariationalAutoencoder(nn.Module):
    def __init__(self, action_dim=131, out_dim=2):
        super().__init__()
        self.fc1 = nn.Linear(action_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mu = nn.Linear(32, out_dim)
        self.fc_logvar = nn.Linear(32, out_dim)
        self.relu = nn.ReLU()

        self.fc3 = nn.Linear(out_dim, 32)
        self.fc4 = nn.Linear(32, 64)
        self.fc5 = nn.Linear(64, action_dim)

    def encode(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.relu(self.fc3(z))
        x = self.relu(self.fc4(x))
        return torch.tanh(self.fc5(x))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if self.training else mu  # deterministic in eval mode
        return self.decode(z), mu, logvar


class SparseAutoencoder(nn.Module):
    """Standard sparse autoencoder: z = ReLU(W_enc (x - b_dec) + b_enc), x_hat = W_dec z + b_dec.

    Single linear layer each way, non-negative code, linear decoder (no tanh, so x_hat can leave [-1, 1]; the env clips
    actions). Each decoder column (one latent's direction in action space) is kept at unit norm by normalize_decoder(),
    which the training loop calls after every optimiser step, so the L1 penalty on z can't be cheated by shrinking z
    and growing W_dec. out_dim is the number of latents and is usually wider than a dense bottleneck.
    """

    def __init__(self, action_dim=131, out_dim=32):
        super().__init__()
        self.b_dec = nn.Parameter(torch.zeros(action_dim))
        self.enc = nn.Linear(action_dim, out_dim)
        self.dec = nn.Linear(out_dim, action_dim, bias=False)
        self.relu = nn.ReLU()
        self.normalize_decoder()

    @torch.no_grad()
    def normalize_decoder(self):
        w = self.dec.weight  # (action_dim, out_dim): column i is latent i's direction
        w /= w.norm(dim=0, keepdim=True).clamp_min(1e-8)

    def encode(self, x):
        return self.relu(self.enc(x - self.b_dec))

    def decode(self, z):
        return self.dec(z) + self.b_dec

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z
