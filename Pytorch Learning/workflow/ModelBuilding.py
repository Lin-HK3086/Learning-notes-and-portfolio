import torch
from torch import nn
import GetDataReady

# Pytorch has 4 essential models you can use to create almost any kind of neural network you can imagine.
# torch.nn, torch,optim, torch.utils.data.Dataset and torch.utils.data.DataLoader
# We will focus on the first two

# torch.nn: Contains all building blocks for computational graphs (essentially a series of computations executed in a particular way).
# almost everything in a PyTorch neural network comes from torch.nn:
# nn.Module contains the larger building blocks (layers)
# nn.Parameter contains the smaller parameters like weights and biases (put these together to make nn.Module(s))
# forward() tells the larger blocks how to make calculations on inputs (tensors full of data) within nn.Module(s)
# torch.optim contains optimization methods on how to improve the parameters within nn.Parameter to better represent input data


# Build the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        #inicialize model parameters
        self.weights = nn.Parameter(torch.randn(
            1,
            requires_grad=True,
            dtype=torch.float
        ))
        self.bias = nn.Parameter(torch.randn(
            1,
            requires_grad=True,
            dtype=torch.float
        ))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights * x + self.bias
    # Any subclass of nn.model needs to override forward()(this defines the forward computation of the model)

# Set manual seed since nn.Parameter are randomly initialized
torch.manual_seed(42)

# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

# Check the nn.Parameter(s) within the nn.Module subclass we created
print(list(model_0.parameters()))

# List named parameters
print(model_0.state_dict())


#Making predictions using torch.inference_mode()
# Make predictions with model
with torch.inference_mode():
    y_preds = model_0(GetDataReady.X_test)

# Note: in older PyTorch code you might also see torch.no_grad()
# with torch.no_grad():
#   y_preds = model_0(X_test)

# Check the predictions
print(f"Number of testing samples: {len(GetDataReady.X_test)}")
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

# show the result by plot
GetDataReady.plot_predictions(predictions=y_preds)
# you can see that this predication is pretty bad, but we can make it better via training it.