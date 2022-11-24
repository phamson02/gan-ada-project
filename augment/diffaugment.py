import torch.nn as nn
from augment.transforms import AUGMENT_FNS


class DiffAugment(nn.Module):
    def __init__(self, policy='', channels_first=True):
        self.policy = policy
        self.channels_first = channels_first

    def forward(self, x):
        if self.policy:
            if not self.channels_first:
                x = x.permute(0, 3, 1, 2)
            for p in self.policy.split(','):
                for f in AUGMENT_FNS[p]:
                    x = f(x)
            if not self.channels_first:
                x = x.permute(0, 2, 3, 1)
            x = x.contiguous()
        return x


