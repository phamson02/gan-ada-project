import torch.nn as nn
import numpy as np
from base import BaseGAN


class Generator(nn.Module):
    """
    Generator network
    """
    def __init__(self, latent_dim, img_shape):
        super().__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.latent_dim = latent_dim
        self.img_shape = img_shape

        self.model = nn.Sequential(
            *block(latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img

class Discriminator(nn.Module):
    """
    Discriminator network
    """
    def __init__(self, img_shape):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        img_flat = z.view(z.size(0), -1)
        validity = self.model(img_flat)
        
        return validity

class GAN(BaseGAN):
    """
    Vanilla GAN
    """
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.channels_img = 1
        self.img_shape = (1, 28, 28)

        self.generator = Generator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)