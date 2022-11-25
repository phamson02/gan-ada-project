import torch
import torch.nn.functional as F
import augment.base_augment as BAug


class DiffAugment(BAug.AugmentPipe):
    '''
    The paper suggest that
     - Cutout size should be half of image, Color: (Brightness in (-0.5, 0.5), Contrast in (0.5, 1.5), Saturation in (0, 2)),
       translation in (-1/8, 1/8)
     - They also experiment contributions of rotate90 {-90, 0, 90}, Gaussian noise (std=0.1), Geometry transformations:
       (Bilinear translation (-0.25, 0.25), bilinear scaling (0.75, 1.25), bilinear rotation (-30, 30), bilinear shearing (-0.25, 0.25)
     - Combination of Translation + Color + Cutout bring the best result.
    '''

    def __init__(self, brightness=0, saturation=0, contrast=0, brightness_std=0.2, contrast_std=0.5, saturation_std=1,
                 cutout=0, cutout_size=0.5,
                 translation=0, translation_ratio=0.125,
                 ):
        super(DiffAugment, self).__init__(xflip=0, rotate90=0, xint=0,
                                          scale=0, rotate=0, aniso=0, xfrac=0,
                                          lumaflip=0, hue=0,
                                          imgfilter=0, noise=0
                                          )
        self.translation = float(translation)
        self.translation_ratio = float(translation_ratio)
        self.name = "DiffAugment"

    def forward(self, images, debug_percentile=None):
        if self.translation > 0:
            images = rand_translation(images, ratio=self.translation_ratio)
        return super(DiffAugment, self).forward(images, debug_percentile)


def rand_translation(x, ratio=0.125):
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x

