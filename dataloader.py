import os
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn.functional as F

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class dataloader(Dataset):
    def __init__(self, filenames_file, data_path, disp_path, h, w):
        names = get_filenames(filenames_file)
        names_r = names['imgr']
        names_d = names['dispr']
        self.imgr_paths = [ os.path.join(data_path, x) for x in names_r ]
        self.dispr_paths = [ os.path.join(disp_path, x) for x in names_d ]
        assert len(self.imgr_paths) == len(self.dispr_paths)
        self.totensor = transforms.ToTensor()
        self.resize = transforms.Resize( (h, w) )
        self.h = h//1
        self.w = w//1

    def __len__(self):
        return len(self.imgr_paths)

    def __getitem__(self, idx):

        right_image = Image.open(self.imgr_paths[idx])
        # right_image = cv2.imread(self.imgr_paths[idx])
        right_disp = np.load(self.dispr_paths[idx])

        # C x H x W tensor will be treated as 1D
        right_image = self.totensor(right_image)
        right_disp = self.totensor(right_disp)
        right_image = torch.unsqueeze(right_image, 0)
        right_disp = torch.unsqueeze(right_disp, 0)

        # right_image.to(device)
        # right_disp.to(device)

        right_image = F.interpolate(right_image, size=(self.h, self.w))
        right_disp = F.interpolate(right_disp, size=(self.h, self.w))

        right_image = torch.squeeze(right_image, 0)
        right_disp = torch.squeeze(right_disp, 0)

        sample = {"imgr": right_image, "dispr": right_disp}
        # sample.to(device)
        return sample


def get_filenames(filename_file, prefix=None):
    f = list(open(filename_file, 'r'))
    names = {}
    names["dispr"] = []
    names["imgr"] = []

    count = len(f)
    for i in range(count):
        line = f[i].rstrip()
        left, right = line.split(' ')
        if prefix is not None:
            left = os.path.join(prefix, left)
            right = os.path.join(prefix, right)
        names["dispr"].append(left)
        names["imgr"].append(right)
    return names

def prepare_dataloader(filenames_file, data_path, disp_path, batch_size, h, w, num_workers):

    dataset = dataloader(filenames_file, data_path, disp_path, h, w)
    n_img = len(dataset)
    print('Use a dataset with', n_img, 'images')

    loader = DataLoader(dataset, batch_size=batch_size,
                        shuffle=False, num_workers=num_workers,
                        pin_memory=True)

    return n_img, loader
