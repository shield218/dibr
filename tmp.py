import torch
import cv2
from PIL import Image
import os
import torchvision.transforms as transforms
import numpy as np
import skimage

os.environ['CUDA_VISIBLE_DEVICES']='0'

totensor = transforms.ToTensor()

# img = Image.open('a.png')
# img = cv2.imread('a.png')
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = skimage.io.imread('a.png')
arr = totensor(img)
dst = arr.cpu().detach().numpy().transpose(1,2,0)
dst = dst*255
dst = dst.astype(np.uint8)
# dst = cv2.cvtColor(dst, cv2.COLOR_RGB2BGR)
cv2.imwrite("d.png", dst)
# skimage.io.imsave('d.png', dst)