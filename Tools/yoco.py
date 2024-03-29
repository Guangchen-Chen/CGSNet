import torch
import torchvision.transforms as transforms



transforms1 = torch.nn.Sequential(
    transforms.RandomHorizontalFlip(), )

def YOCO(images, aug, h, w):
    images = torch.cat((aug(images[:, :, :, 0:int(w/2)]), aug(images[:, :, :, int(w/2):w])), dim=3) if \
    torch.rand(1) > 0.5 else torch.cat((aug(images[:, :, 0:int(h/2), :]), aug(images[:, :, int(h/2):h, :])), dim=2)
    return images