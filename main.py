import torch
from augment import translate2d, wavelets
from torch_utils.ops import upfirdn2d_gradfix


mx0 = 4
mx1 = 2
my0 = 4
my1 = 2
G_inv = torch.eye(3)
images = torch.randn(1, 3, 48, 48)
images = torch.nn.functional.pad(input=images, pad=[mx0, mx1, my0, my1], mode='reflect')
G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv
Hz_geom = upfirdn2d_gradfix.setup_filter(wavelets['sym6'])

# Upsample.
images = upfirdn2d_gradfix.upsample2d(x=images, f=Hz_geom, up=2)
print(images)