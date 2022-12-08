import torch
import torch.nn.functional as F
import augment.base_augment as BAug
import numpy as np

def step(values):
    # negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + torch.sign(values))


class Ada(BAug.AugmentPipe):
    def __init__(self, ada_kimg=500, ada_target=0.6, integration_steps=4, *args, **kwargs):
        super(Ada, self).__init__(*args, **kwargs)
        self.register_buffer('p', torch.zeros([]))
        self.name = "ADA"
        self.ada_target = ada_target
        self.integration_steps = integration_steps
        self.ada_kimg = ada_kimg

    def update_p(self, lambda_t, batch_size_D):
        # the augmentation probability is updated based on the dicriminator's
        # accuracy on real images
        accuracy_error = lambda_t.cpu() - self.ada_target
        self.p.copy_(torch.as_tensor(torch.clamp(self.p + np.sign(accuracy_error) * \
                                                 batch_size_D * self.integration_steps / \
                                                 (1000 * self.ada_kimg), 0., 1.)))
