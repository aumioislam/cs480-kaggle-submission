from typing import Callable
import lightning as L
import torch.nn as nn
import torch.optim as optim

from .utils import compute_r2

class MetadataFeedForward(nn.Module):
    def __init__(self, hidden_dims=[256, 128, 64]):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(163, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[2], 6)
        )

    def forward(self , x):
        return self.model(x)

class MetadataModel(L.LightningModule):
    def __init__(self, loss_fn: Callable = nn.HuberLoss(), scheduler=None, lr=0.001):
        super().__init__()

        self.model = MetadataFeedForward()

        self.loss_fn = loss_fn
        self.scheduler = scheduler

        self.lr = lr

        self.save_hyperparameters(ignore=['loss_fn', 'scheduler'])

    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=self.lr) # type: ignore

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
        (imgs, metadata), traits = batch
        preds = self.model(metadata)
        loss = self.loss_fn(preds, traits)

        self.log(f"{mode}_loss", loss.item())

        if mode == "val":
            r2 = compute_r2(preds, traits)
            self.log("val_r2", r2)

        return loss
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        loss = self._calculate_loss(batch)
        return loss
    
    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode="val")

    def predict_step(self, batch, batch_idx):
        _, metadata = batch
        return self.model(metadata)
