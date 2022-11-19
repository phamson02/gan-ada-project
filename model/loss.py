import torch.nn.functional as F


def adversarial_loss(output, target):
    return F.binary_cross_entropy(output, target)