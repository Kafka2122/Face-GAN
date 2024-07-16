import torch
import torch.nn.functional as F

def reconstruction_loss(real_img=None, gen_image=None):
    return F.mse_loss(gen_image, real_img)

def diversity_loss(style_img=None, gen_image=None):
    return F.mse_loss(gen_image, style_img)

def real_loss(disc_output=None):
    return F.binary_cross_entropy_with_logits(disc_output, torch.ones_like(disc_output))

def fake_loss(disc_output=None):
    return F.binary_cross_entropy_with_logits(disc_output, torch.zeros_like(disc_output))