from pathlib import Path

from typing import Callable, Optional

import pandas as pd
from PIL import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms.v2 import PILToTensor

class PlantDataset(Dataset):
    train_images = "train_images"
    test_images = "test_images"
    train_csv = "train.csv"
    test_csv = "test.csv"

    def __init__(
        self, 
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        _root = Path(root)

        if _root.is_dir():
            self.root = _root
        else: 
            raise RuntimeError("Could not find provided path to data...")
        
        self.root = root
        self.train = train
        self.transform = transform or PILToTensor()

        if not self._check_for_data():
            raise RuntimeError("Could not find all expected dataset artifacts...")
        
        self.mu = None
        self.sigma = None
        
        self.images, self.metadata, self.traits = self._load_data()

        return

    def _check_for_data(self):
        if self.train:
            if (
                not Path(self.root, self.train_images).is_dir() or
                not Path(self.root, self.train_csv).is_file()
            ):
                return False
        else:
            if (
                not Path(self.root, self.test_images).is_dir() or
                not Path(self.root, self.test_csv).is_file()
            ):
                return False
        
        return True

    def _normalize_tensor(self, x: Tensor):
        mu = x.mean(0, keepdim=True)
        sigma = x.std(0, keepdim=True)

        return (x-mu)/sigma, mu, sigma

    def _load_data(self):
        img_dir, csv_loc = (
            (Path(self.root, self.train_images), Path(self.root, self.train_csv))
            if self.train else 
            (Path(self.root, self.test_images), Path(self.root, self.test_csv))
        )

        self.csv = pd.read_csv(csv_loc)

        images = []

        for _, row in self.csv.iterrows():
            id = int(row['id'])
            img_loc = Path(img_dir, f"{id}.jpeg")

            if not img_loc.is_file():
                msg = f"Could not find {str(img_loc)} ..."
                raise RuntimeError(msg)
            
            img = self.transform(
                Image
                    .open(img_loc)
                    .convert('RGB')
            )

            images.append(img)

        csv_tensor = torch.tensor(self.csv.values, dtype=torch.float32)

        if self.train:
            metadata = csv_tensor[:, 1:-6]
            metadata = self._normalize_tensor(metadata)[0]
            targets = csv_tensor[:, -6:]
            targets, self.mu, self.sigma = self._normalize_tensor(targets)
        else:
            metadata = csv_tensor[:, 1:]
            metadata = self._normalize_tensor(metadata)[0]
            targets = None
        return (images, metadata, targets)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        if self.train:
            return (self.images[idx], self.metadata[idx]), self.traits[idx]   # type: ignore
        else:
            return self.images[idx], self.metadata[idx]
