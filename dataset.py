import torch
from torch.utils.data import Dataset
import nibabel as nib
from data_utils import *

class CancerNetPCa(Dataset):
        def __init__(self, img_path, mask_path, img_size=(256, 256), slice_num=9, transform=None):
            self.img_path = img_path
            self.mask_path = mask_path
            self.img_size = img_size
            self.slice = slice_num
            self.transform = transform

        def __len__(self):
            print(len(self.img_path))
            return len(self.img_path)

        def __getitem__(self, idx):   
            img_path = self.img_path[idx]
            img = nib_to_numpy(img_path)
            prostate_path, lesion_path = self.mask_path
            prostate = np.load(prostate_path[idx])
            lesion = np.load(lesion_path[idx])
            
            if len(self.mask_path.shape) > 1:
                lesion *= prostate
            # align prostate mask with image
            mask_t= np.transpose(lesion, (2, 1, 0))
            mask = np.flip(mask_t, axis=1)

            img_slice = img[:, :, self.slice]
            mask_slice = mask[:, :, self.slice]

            if self.transform is not None:
                img_slice = self.transform(img_slice)
                mask_slice = self.transform(mask_slice)

            #img_tensor = torch.from_numpy(img_slice).float()
            #mask_tensor = torch.from_numpy(mask_slice).long()

            #mg_tensor = img_tensor.unsqueeze(0)
            #mask_tensor = mask_tensor.unsqueeze(0) 

            return img_slice, mask_slice