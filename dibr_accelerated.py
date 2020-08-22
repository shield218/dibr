import os
import torch
import numpy as np
import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from dataloader import prepare_dataloader
import skimage

os.environ['CUDA_VISIBLE_DEVICES']='0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def apply_disparity(img, disp):
    batch_size, _, height, width = img.size()

    # Original coordinates of pixels
    x_base = torch.linspace(0, 1, width).repeat(batch_size,
                height, 1).type_as(img)
    y_base = torch.linspace(0, 1, height).repeat(batch_size,
                width, 1).transpose(1, 2).type_as(img)

    # Apply shift in X direction
    x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
    flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
    # In grid_sample coordinates are assumed to be between -1 and 1
    output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear',
                            padding_mode='zeros')
    return output

def generate_image_left(img, disp):
    return apply_disparity(img, -disp)

def check_dirs_exist(root_dir, split, subdirs):
    for subdir in subdirs:
        if not os.path.isdir(os.path.join(root_dir, split, subdir)):
            os.makedirs(os.path.join(root_dir, split, subdir))
    
def dibr(h, w):
    data_path = '/home/chu/reproduction/dense_copy/testdata/'
    disp_path = '/home/chu/reproduction/dense_copy/testout/'
    dst_dir = 'output'
    subdirs = ['left', 'right', 'blended']
    split = 'saion'
    filename_file = split + '.txt'
    batch_size = 1000
    num_workers = 0
    n, loader = prepare_dataloader(filename_file, data_path, disp_path, batch_size, h, w, num_workers)
    check_dirs_exist(dst_dir, split, subdirs)
    
    fps = 29.917
    out = cv2.VideoWriter( '{}.{}'.format(split, 'avi'), cv2.VideoWriter_fourcc('M','J','P','G'), fps, (w, h) )

    step = 0
    cnt = 0
    for data in loader:
        imgr = data['imgr'].to(device)
        dispr = data['dispr'].to(device)
        imgl = generate_image_left(imgr, dispr*0.002)

        imgl = imgl.cpu().detach().numpy().squeeze().transpose(0, 2, 3, 1)
        imgr = imgr.cpu().detach().numpy().squeeze().transpose(0, 2, 3, 1)
        current_batch_size = np.shape(imgl)[0]
        print('batch size {}'.format(current_batch_size) )
        for i in range(current_batch_size):
            blended = cv2.addWeighted(imgl[i], 0.5, imgr[i], 0.5, 0)
            blended = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
            blended = (blended * 255).astype(np.uint8)
            out.write(blended)

        step += 1
        cnt += np.shape(imgl)[0]
    out.release()

if __name__ == "__main__":
    dibr(268, 480)
