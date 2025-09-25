import torch
from torchvision import models
from torch import nn
from torchinfo import summary

weights = models.ResNet18_Weights.DEFAULT
model_resnet18 = models.resnet18(weights=weights)
torch.manual_seed(33)
model_resnet18.fc=nn.Linear(in_features=512, out_features=2)

if __name__ == '__main__':
    # Get a summary of the model (uncomment for full output)
    summary(model=model_resnet18,
            input_size=(32, 3, 128, 128),
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
            )