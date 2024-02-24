import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from data_utils import nib_to_numpy

class CancerNetPCa:
    def __init__(self, img_path, mask_path, batch_size=10, val_split=0.15, test_split=0.15, num_workers=2, img_size=(256, 256), slice_num=9, prostate=False, transform=None):
        self.dataset = CancerNetPCaDataset(img_path, mask_path, img_size, slice_num, prostate, transform)

        dataset_size = len(self.dataset)
        val_size = int(val_split * dataset_size)
        test_size = int(test_split * dataset_size)
        train_size = dataset_size - val_size - test_size

        print(f'Using {train_size}/{val_size}/{test_size} train/val/test split')

        generator = torch.Generator()
        generator.manual_seed(42)

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(self.dataset, [train_size, val_size, test_size], generator=generator)

        self.train = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val = DataLoader(self.val_dataset, batch_size=batch_size, num_workers=num_workers)
        self.test = DataLoader(self.test_dataset, batch_size=batch_size, num_workers=num_workers)
     
class CancerNetPCaDataset(Dataset):
        def __init__(self, img_path, mask_path, img_size=(256, 256), slice_num=9, prostate=False, transform=None):
            self.img_path = img_path
            self.mask_path = mask_path
            self.img_size = img_size
            self.slice = slice_num
            self.prostate = prostate
            self.transform = transform

        def __len__(self):
            return len(self.img_path)

        def __getitem__(self, idx):   
            img_path = self.img_path[idx]
            img = nib_to_numpy(img_path)
            prostate_path, lesion_path = self.mask_path
            prostate = np.load(prostate_path[idx])
            lesion = np.load(lesion_path[idx])
            
            if self.prostate:
                lesion *= prostate
            # align mask with image
            mask_t = np.transpose(lesion, (2, 1, 0))
            mask = np.flip(mask_t, axis=1)

            img_slice = img[:, :, self.slice]
            mask_slice = mask[:, :, self.slice]

            if self.transform is not None:
                img_slice = self.transform(img_slice)
                mask_slice = self.transform(mask_slice)

            return img_slice, mask_slice