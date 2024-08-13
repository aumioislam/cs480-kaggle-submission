from PlantDataset import PlantDataset
from PlantModels import DinoModel

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR

from torchvision import transforms

from PlantModels.utils import generate_submission, r2_loss

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

    train_loader = DataLoader(train, batch_size=512, num_workers=14)
    val_loader = DataLoader(val, batch_size=512, num_workers=14)


    trainer = L.Trainer(
        devices=1,
        max_epochs=17,
        accelerator="auto",
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_r2"),
            LearningRateMonitor("epoch"),
        ],
    )
    
    sch = lambda x: MultiStepLR(x, milestones=[25, 50, 75], gamma=0.2)
    model = DinoModel(loss_fn=r2_loss, scheduler=None, lr=0.001, freeze=True)
    trainer.fit(model, train_loader, val_loader) # type: ignore

    # generate submission file
    best = DinoModel.load_from_checkpoint(trainer.checkpoint_callback.best_model_path) # type: ignore

    val_result = trainer.validate(best, dataloaders=val_loader, verbose=False)
    print(f"val_r2: {val_result[0]['val_r2']}")
    submission_name = f"lmbda-dinov2-submission-v{trainer.logger.version}-r2-{val_result[0]['val_r2']:.4f}" # type: ignore

    test_ds = PlantDataset("data/", train=False, transform=transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    ))
    test_dl = DataLoader(test_ds, batch_size=256, num_workers=14)

    generate_submission(
        lambda: trainer.predict(best, dataloaders=test_dl),
        dataset.mu,
        dataset.sigma,
        test_ds.csv,
        submission_name
    )
    
