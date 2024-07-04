import os
import numpy as np
import torch
from scipy.signal import resample
from typing import Tuple
from termcolor import cprint


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", augs = None, resampling_rate = None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        self.augs = augs
        self.resampling_rate = resampling_rate
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.__standardize(self.X[i])
        if self.resampling_rate:
            X = self.__resample(X, self.resampling_rate)
        if self.augs:
            X = self.__augment(X, self.augs)
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
        
    def __standardize(self, img):
        # Log transformation
        #img = np.clip(img,np.exp(-4),np.exp(8))
        #img = np.log(img)

        # Standarize per image
        ep = 1e-6
        m = np.nanmean(img.flatten())
        s = np.nanstd(img.flatten())
        img = (img-m)/(s+ep)
        img = np.nan_to_num(img, nan=0.0)

        return img
    
    def __resample(self, img, new_freq):
        num_samples = int(img.shape[-1] * new_freq / 200)  # 元のサンプリングレートは200Hz
        resampled_data = resample(img.numpy(), num_samples, axis=-1)
        return torch.tensor(resampled_data)
    
    def __random_transform(self, img, transform):
        return transform(image=img)['image']

    def __augment(self, img_batch, transform):
        for i in range(img_batch.shape[0]):
              img_batch[i,] = self.__random_transform(img_batch[i,],  transform)
        return img_batch
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]