import torch
from torch.utils.data import Dataset
import nibabel as nib
from data_utils import *

class CancerNetPCa(Dataset):
    def __init__(self, img_path, mask_path, img_size=(256, 256), transform=None):
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_size = img_size
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_path = self.img_path[idx]
        mask_path = self.mask_path[idx]

        img = nib.load(img_path).dataobj
        img = np.array(img).astype(np.uint8)

        mask = nib.load(mask_path).dataobj
        mask = np.array(mask).astype(np.uint8)

        slice_num = 9
        img_slice = img[:, :, slice_num]
        mask_slice = mask[:, :, slice_num]

        img_std = resize_image(normalize_image(img_slice), self.img_size)
        mask_std = resize_image(normalize_image(mask_slice), self.img_size)

        img_tensor = torch.from_numpy(img_std).float()
        mask_tensor = torch.from_numpy(mask_std).long()

        img_tensor = img_tensor.unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0) 

        # apply transformations if provided (add later)
        #if self.transform:

        return img_tensor, mask_tensor