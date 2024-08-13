from PlantDataset import PlantDataset
from PlantModels import MetadataModel

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau

from pathlib import Path

from PlantModels.utils import generate_submission, r2_loss

if __name__ == "__main__":
    dataset = PlantDataset("data/")
    train, val = random_split(dataset, [0.8, 0.2])

    train_loader = DataLoader(train, batch_size=256, num_workers=15)
    val_loader = DataLoader(val, batch_size=256, num_workers=15)

    trainer = L.Trainer(
        devices=1,
        max_epochs=1,
        accelerator="auto",
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_r2"),
            LearningRateMonitor("epoch"),
        ],
    )
    
    sch = lambda x: ReduceLROnPlateau(x, min_lr=0.0005, factor=0.3)
    model = MetadataModel(loss_fn=nn.MSELoss(), scheduler=None, lr=0.001)
    trainer.fit(model, train_loader, val_loader)

    # generate submission file
    best = MetadataModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # type: ignore

    val_result = trainer.validate(best, dataloaders=val_loader, verbose=False)
    print(f"val_r2: {val_result[0]['val_r2']}")
    submission_name = f"submission-v{trainer.logger.version}-r2-{val_result[0]['val_r2']:.4f}" # type: ignore

    test_ds = PlantDataset("data/", train=False)
    test_dl = DataLoader(test_ds, batch_size=256, num_workers=15)

    generate_submission(
        lambda: trainer.predict(best, dataloaders=test_dl),
        dataset.mu,
        dataset.sigma,
        test_ds.csv,
        submission_name
    )
    