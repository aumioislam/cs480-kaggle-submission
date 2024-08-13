from typing import Callable
import lightning as L
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from .utils import compute_r2

class DinoBackbone(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

        # load pretrained vit model
        self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg')
        
        # discard pretrained head
        out_dim = self.vit.norm.normalized_shape[0]

        # freeze everything in vit
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False

            self.vit.eval()

        self.head = nn.Sequential( # type: ignore
            nn.Linear(out_dim, 256),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.Dropout1d(p=0.5),
            nn.Linear(256, 64),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU()
        )

    def forward(self , x):
        return self.head(self.vit(x))
    
class DinoMLP(nn.Module):
    def __init__(self, freeze):
        super().__init__()

        self.backbone = DinoBackbone(freeze=freeze)
        self.metadata_fc = nn.Sequential(
            nn.Linear(163, 256),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.Dropout1d(p=0.5),
            nn.Linear(256, 64),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU()
        )

        self.mlp_head = nn.Sequential(
            nn.Linear(64+64, 256),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 128),
            nn.LazyBatchNorm1d(),
            nn.LeakyReLU(),
            nn.Linear(128, 6)
        )

    def forward(self, input):
        img, metadata = input
        vit_out = self.backbone(img)
        metadata_out = self.metadata_fc(metadata)

        mlp_in = torch.cat((metadata_out, vit_out), dim=1)
        return self.mlp_head(mlp_in)

class DinoModel(L.LightningModule):
    def __init__(self, loss_fn: Callable = nn.MSELoss(), scheduler=None, lr=0.001, freeze=True):
        super().__init__()

        self.model = DinoMLP(freeze=freeze)

        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.lr = lr

        self.save_hyperparameters(ignore=['loss_fn', 'scheduler'])

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01) # type: ignore

        if not self.scheduler:
            return [opt], []
        
        return {
            "optimizer": opt, 
            "lr_scheduler": {
                "scheduler": self.scheduler(opt),
                "monitor": "val_loss"
            }
        }
    
    def _calculate_loss(self, batch, mode="train"):
        input, traits = batch
        preds = self.model(input)
        loss = self.loss_fn(preds, traits)

        self.log(f"{mode}_loss", loss.item())

        if mode == "val":
            r2 = compute_r2(preds, traits)
            self.log("val_r2", r2, logger=True)

        return loss
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def predict_step(self, batch, batch_idx):
        return self.model(batch)
