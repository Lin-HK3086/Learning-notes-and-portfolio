import torch
from torchvision import transforms
from pathlib import Path

import data, engine, utils, model_build

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

image_path = Path('D:\Python Program\Kaggle Competetion\Heads or Tails\data\images')
train_dir = image_path / 'train'
valid_dir = image_path / 'valid'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
manual_transform = transforms.Compose([
    transforms.Resize((400, 400)),
    transforms.ToTensor(),
    #normalize
])

if __name__ == "__main__":
    train_dataloader, test_dataloader, classnames = data.creat_dataloader(
        train_dir=str(train_dir),
        valid_dir=str(valid_dir),
        transform=manual_transform,
        batch_size=20,
        num_workers=2
    )
    model = model_build.model_resnet18.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00003)
    set_seed(33)
    print(f"Training on {device}........")
    result = engine.train(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        train_loader=train_dataloader,
        valid_loader=test_dataloader,
        device=device,
        epochs=20
    )
    utils.save_model(model=model,
                     target_dir='/Kaggle Competetion/Heads or Tails/models',
                     model_name='resnet18_lr00003_400x400.pth', )
