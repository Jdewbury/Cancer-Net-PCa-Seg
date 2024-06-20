import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from data_utils import nib_to_numpy

class CancerNetPCa:
    def __init__(self, img_path, mask_path, seed=42, batch_size=10, train_split=0.7, test_split=0.15, prostate=False, transform=None):
        np.random.seed(seed)
        num_workers = os.cpu_count()
        num_workers = max(1, num_workers - 2)
        
        dataset_size = len(img_path)
        idxs = np.arange(dataset_size)
        np.random.shuffle(idxs)

        train_size = int(train_split * dataset_size)
        test_size = int(test_split * dataset_size)
        val_size = dataset_size - train_size - test_size

        train_idx = idxs[:train_size]
        test_idx = idxs[train_size:train_size + test_size]
        val_idx = idxs[dataset_size - val_size:]
 
        print(f'Using {train_size}/{val_size}/{test_size} train/val/test split')

        train_dataset = CancerNetPCaDataset(img_path[train_idx], mask_path[:, train_idx], prostate, transform)
        val_dataset = CancerNetPCaDataset(img_path[val_idx], mask_path[:, val_idx], prostate, transform)
        test_dataset = CancerNetPCaDataset(img_path[test_idx], mask_path[:, test_idx], prostate, transform)

        self.train = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers)
        self.test = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers)
     
class CancerNetPCaDataset(Dataset):
        def __init__(self, img_path, mask_path, prostate=False, transform=None):
            self.img_path = img_path
            self.mask_path = mask_path
            self.prostate = prostate
            self.transform = transform
            self.data = self._prepare_data()

        def _prepare_data(self):
            data = []
            for img, lesion, prostate in zip(self.img_path, self.mask_path[0], self.mask_path[1]):
                img_np = nib_to_numpy(img)
                prostate_np = np.load(lesion)
                lesion_np = np.load(prostate)

                if self.prostate:
                    lesion_np *= prostate_np
                # align mask with image
                mask_t = np.transpose(lesion_np, (2, 1, 0))
                mask = np.flip(mask_t, axis=1)
                mask = (mask > 0).astype(np.float32)
                
                num_slices = min(img_np.shape[2], mask.shape[2])

                for slice in range(num_slices):
                    data.append((img_np[:, :, slice], mask[:, :, slice]))

            return data

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):   
            img_slice, mask_slice = self.data[idx]

            if self.transform is not None:
                img_slice = self.transform(img_slice)
                mask_slice = self.transform(mask_slice)

            return img_slice, mask_slice