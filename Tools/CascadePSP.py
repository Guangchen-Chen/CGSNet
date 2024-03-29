import cv2
import time
import matplotlib.pyplot as plt
import segmentation_refinement as refine
import torch

image = cv2.imread('/home/bj/projects/Semi-supervised/ST-PlusPlus-master/SODL/JPEGImages/001.jpg')
mask = cv2.imread('/home/bj/projects/Semi-supervised/ST-PlusPlus-master/SODL/mask/001.png', cv2.IMREAD_GRAYSCALE)


# model_path can also be specified here
# This step takes some time to load the model
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
refiner = refine.Refiner(device='cuda:0') # device can also be 'cpu'

# Fast - Global step only.
# Smaller L -> Less memory usage; faster in fast mode.
output = refiner.refine(image, mask, fast=False, L=900)

plt.imshow(output)
plt.show()