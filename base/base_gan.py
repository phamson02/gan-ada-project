from itertools import chain
import torch
import numpy as np
from abc import ABC


class BaseGAN(ABC):
    """
    Base class for all GANs
    """
    def __init__(self, generator, discriminator, latent_dim=None):
        self.latent_dim = latent_dim
        self.generator = generator
        self.discriminator = discriminator

    def parameters(self):
        return chain(self.generator.parameters(), self.discriminator.parameters())    

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])

        return f'{self.generator}\n{self.discriminator}\nTrainable parameters: {params}'

    def to(self, device):
        self.generator.to(device)
        self.discriminator.to(device)
        
        return self

    def DataParallel(self, **kwargs):
        self.generator = torch.nn.DataParallel(self.generator, **kwargs)
        self.discriminator = torch.nn.DataParallel(self.discriminator, **kwargs)

        return self

    def state_dict(self):
        return {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.generator.load_state_dict(state_dict['generator'])
        self.discriminator.load_state_dict(state_dict['discriminator'])