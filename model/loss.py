import torch.nn.functional as F
import torch

def adversarial_loss(output, target):
    return F.binary_cross_entropy_with_logits(output, target)
def wasserstein_loss(output,target):
    return torch.mean(output) - torch.mean(target)
def leastsquare_loss(output, target):
    return F.mse_loss(output,target)


