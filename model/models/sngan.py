# ResNet generator and discriminator
from torch import nn
import torch.nn.functional as F
from base import BaseGAN
from .spec_norm import SpectralNorm
import numpy as np


class ResBlockGenerator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockGenerator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        self.model = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            self.conv1,
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            self.conv2
            )
        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.model(x) + self.bypass(x)


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)

        if stride == 1:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2)
                )
        else:
            self.model = nn.Sequential(
                nn.ReLU(),
                SpectralNorm(self.conv1),
                nn.ReLU(),
                SpectralNorm(self.conv2),
                nn.AvgPool2d(2, stride=stride, padding=0)
                )
        self.bypass = nn.Sequential()
        if stride != 1:

            self.bypass_conv = nn.Conv2d(in_channels,out_channels, 1, 1, padding=0)
            nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

            self.bypass = nn.Sequential(
                SpectralNorm(self.bypass_conv),
                nn.AvgPool2d(2, stride=stride, padding=0)
            )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

# special ResBlock just for the first layer of the discriminator
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding=1)
        self.bypass_conv = nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)
        nn.init.xavier_uniform_(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform_(self.conv2.weight.data, 1.)
        nn.init.xavier_uniform_(self.bypass_conv.weight.data, np.sqrt(2))

        # we don't want to apply ReLU activation to raw image before convolution transformation.
        self.model = nn.Sequential(
            SpectralNorm(self.conv1),
            nn.ReLU(),
            SpectralNorm(self.conv2),
            nn.AvgPool2d(2)
            )
        self.bypass = nn.Sequential(
            nn.AvgPool2d(2),
            SpectralNorm(self.bypass_conv),
        )

    def forward(self, x):
        return self.model(x) + self.bypass(x)

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.GEN_SIZE = 128
        self.latent_dim = latent_dim
        self.out_channels = img_shape[0]
        n_upsamples = int(np.log2(img_shape[-1]//4))

        self.dense = nn.Linear(self.latent_dim, 4 * 4 * self.GEN_SIZE)
        self.final = nn.Conv2d(self.GEN_SIZE, self.out_channels, 3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.dense.weight.data, 1.)
        nn.init.xavier_uniform_(self.final.weight.data, 1.)
        
        self.upsamples = nn.Sequential(*[ResBlockGenerator(self.GEN_SIZE, self.GEN_SIZE, stride=2) for _ in range(n_upsamples)])
        self.model = nn.Sequential(
            self.upsamples,
            nn.BatchNorm2d(self.GEN_SIZE),
            nn.ReLU(),
            self.final,
            nn.Tanh())

    def forward(self, z):
        return self.model(self.dense(z).view(-1, self.GEN_SIZE, 4, 4))

class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.DISC_SIZE = 128
        self.out_channels = img_shape[0]
        n_downsamples = int(np.log2(img_shape[-1]//16))
        self.downsamples = nn.Sequential(*[ResBlockDiscriminator(self.DISC_SIZE, self.DISC_SIZE, stride=2) for _ in range(n_downsamples)])
        self.model = nn.Sequential(
                FirstResBlockDiscriminator(self.DISC_SIZE, self.DISC_SIZE, stride=2),
                self.downsamples,
                ResBlockDiscriminator(self.DISC_SIZE, self.DISC_SIZE),
                ResBlockDiscriminator(self.DISC_SIZE, self.DISC_SIZE),
                nn.ReLU(),
                nn.AvgPool2d(8),
            )
        self.fc = nn.Linear(self.DISC_SIZE, 1)
        nn.init.xavier_uniform_(self.fc.weight.data, 1.)
        self.fc = SpectralNorm(self.fc)

    def forward(self, x):
        return self.fc(self.model(x).view(-1, self.DISC_SIZE))

class SNGAN64(BaseGAN):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.channels_img = 3
        self.img_shape = (3, 64, 64)

        self.generator = Generator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)
class SNGAN32(BaseGAN):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.channels_img = 3
        self.img_shape = (3, 32, 32)

        self.generator = Generator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)

class SNGAN(BaseGAN):
    def __init__(self, latent_dim):
        self.latent_dim = latent_dim
        self.channels_img = 1
        self.img_shape = (1, 28, 28)

        self.generator = Generator(self.latent_dim, self.img_shape)
        self.discriminator = Discriminator(self.img_shape)