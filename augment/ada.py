import torch
import torch.nn.functional as F
import augment.base_augment as BAug


def step(values):
    # negative values -> 0.0, positive values -> 1.0
    return 0.5 * (1.0 + torch.sign(values))


class Ada(BAug.AugmentPipe):
    def __init__(self, ada_target, integration_steps, *args, **kwargs):
        super(Ada, self).__init__(*args, **kwargs)
        self.register_buffer('p', torch.zeros([]))
        self.name = "ADA"
        self.target_accuracy = ada_target
        self.integration_steps = integration_steps

    def update_p(self, real_logits: torch.Tensor):
        current_accuracy = step(real_logits).mean()

        # the augmentation probability is updated based on the dicriminator's
        # accuracy on real images
        accuracy_error = current_accuracy - self.target_accuracy
        self.p.copy_(torch.as_tensor(torch.clamp(self.p + accuracy_error / self.integration_steps, 0., 1.)))
