import os
import numpy as np
import torch
from typing import Tuple
from termcolor import cprint


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", augs = None) -> None:
        super().__init__()
        
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        
        self.X = torch.load(os.path.join(data_dir, f"{split}_X.pt"))
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        self.augs = augs
        
        if split in ["train", "val"]:
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        X = self.X[i]
        if self.augs:
            X = self._augment(X, self.augs)
        if hasattr(self, "y"):
            return X, self.y[i], self.subject_idxs[i]
        else:
            return X, self.subject_idxs[i]
    
    def _random_transform(self, img, transform):
        return transform(image=img)['image']

    def _augment(self, img_batch, transform):
        for i in range(img_batch.shape[0]):
              img_batch[i,] = self._random_transform(img_batch[i,],  transform)
        return img_batch
        
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]
    
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]