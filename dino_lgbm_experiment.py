from PlantDataset import PlantDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import numpy as np
import pandas as pd

from tqdm import tqdm
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor

from PlantModels.utils import generate_submission

if __name__ == "__main__":
    dataset = PlantDataset("data/", transform=transforms.Compose(
    [
        transforms.RandomHorizontalFlip(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ]
))
    train, val = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train, batch_size=256, num_workers=14)
    val_loader = DataLoader(val, batch_size=256, num_workers=14)

    tr_all_x = []
    val_all_x= []

    tr_all_y = []
    val_all_y = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitg14_reg')
    dino.norm = nn.Identity()
    dino = dino.to(device)
    dino.eval()

    with torch.no_grad():
        for (img, meta), traits in tqdm(train_loader):
            img = img.to(device)
            dino_embed = dino(img)
            tr_all_x.append(torch.cat((meta, dino_embed.cpu()), dim=1))
            tr_all_y.append(traits)

        for (img, meta), traits in tqdm(val_loader):
            img = img.to(device)
            dino_embed = dino(img)
            val_all_x.append(torch.cat((meta, dino_embed.cpu()), dim=1))
            val_all_y.append(traits)

        tr_x = torch.cat(tr_all_x, dim=0).numpy()
        tr_y = torch.cat(tr_all_y, dim=0).numpy()

        val_x = torch.cat(val_all_x, dim=0).numpy()
        val_y = torch.cat(val_all_y, dim=0).numpy()

        model = MultiOutputRegressor(LGBMRegressor(num_leaves=250, n_estimators=200, learning_rate=0.05, reg_lambda=0.1)) # type: ignore
        model.fit(tr_x, tr_y)
        score = model.score(val_x, val_y) # type: ignore

        test_ds = PlantDataset("data/", train=False, transform=transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ]
        ))

        test_dl = DataLoader(test_ds, batch_size=256, num_workers=14)
        test_all_x = []

        for img, meta in tqdm(test_dl):
            img = img.to(device)
            dino_embed = dino(img)
            test_all_x.append(torch.cat((meta, dino_embed.cpu()), dim=1))
        
        test_x = torch.cat(test_all_x, dim=0)

    generate_submission(
        lambda: model.predict(test_x),
        dataset.mu,
        dataset.sigma,
        test_ds.csv,
        f"dino-g14-drop-layer-norm-lgbm-leaves-250-est-200-lr-005-lmbda-01-r2-{score:4f}"
    )
