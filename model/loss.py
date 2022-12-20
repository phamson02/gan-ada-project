import torch.nn.functional as F
import torch
import torch.nn as nn

def adversarial_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)
def wasserstein_loss(output,target):
    return torch.mean(output) - torch.mean(target)
def leastsquare_loss(output, target) -> Tensor:
    return 1/2. * F.mse_loss(nn.Sigmoid()(output),target)


