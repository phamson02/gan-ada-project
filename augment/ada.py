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
        assert integration_steps > 0, "Integration step must be a positive integer!"
        self.ada_kimg = ada_kimg
        self.register_buffer('lambda_t', torch.zeros([1]))

    def update_p(self, batch_size_D):
        # the augmentation probability is updated based on the dicriminator's
        # accuracy on real images
        accuracy_error = self.lambda_t[1:].mean().cpu() - self.ada_target
        self.p.copy_(torch.as_tensor(torch.clamp(self.p + np.sign(accuracy_error) * \
                                                 batch_size_D * self.integration_steps / \
                                                 (1000 * self.ada_kimg), 0., 1.)))

    def update_lambda(self, lambda_t):
        self.lambda_t.data = torch.cat((self.lambda_t.data, lambda_t), 0)

    def reset_lambda(self):
        self.lambda_t.data = torch.zeros((1,))
