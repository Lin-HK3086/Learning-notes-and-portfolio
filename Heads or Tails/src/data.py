from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os

def creat_dataloader(train_dir: str,
                     valid_dir: str,
                     transform: transforms.Compose,
                     batch_size: int,
                     num_workers: int = 2):
    train_data = datasets.ImageFolder(root=train_dir,
                                      transform=transform,
                                      target_transform=None)
    valid_data = datasets.ImageFolder(root=valid_dir,
                                      transform=transform)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    valid_loader = DataLoader(
        dataset=valid_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, valid_loader, train_data.classes
