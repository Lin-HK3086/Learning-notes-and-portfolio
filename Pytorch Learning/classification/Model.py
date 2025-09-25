import torch
from torch import nn

device = "cuda"

class CircleModelV0(nn.Module):
    def __init__(self):
        super().__init__()
        # creat 2 nn.Linear layers capable of handing X and y input and output shape
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=5)
        self.layer_3 = nn.Linear(in_features=5, out_features=1)

    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))

#You can also do the same as above using nn.Sequential
# Replicate CircleModelV0 with nn.Sequential
model = nn.Sequential(
    nn.Linear(in_features=2, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=10),
    nn.ReLU(),
    nn.Linear(in_features=10, out_features=1),
    nn.Sigmoid()
).to(device)
# which is much simpler, but it always runs in sequent order
# the model is ready on the GPU

# Set loss function and optimizer
torch.manual_seed(33)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# it a function used to evaluate how your model is going
# Calculate accuracy (a classification metric)
def accuracy_fn(y_pred,y_true):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100
    return acc

