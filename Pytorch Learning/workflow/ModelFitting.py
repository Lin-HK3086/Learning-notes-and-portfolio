import torch
from torch import nn
from ModelBuilding import LinearRegressionModel
import GetDataReady
import matplotlib.pyplot as plt

# Train(fit): find the most correct pattern to predicate with the help of datas

# Set manual seed since nn.Parameter are randomly initialized
torch.manual_seed(42)
# Create an instance of the model (this is a subclass of nn.Module that contains nn.Parameter(s))
model_0 = LinearRegressionModel()

# Creating a loss function and optimizer in PyTorch
# For our model to update its parameters on its own, we will add a loss function as well as an optimizer

# Loss function: Measures how wrong your model's predictions. Lower the better
# Optimizer: Tells your model how to update its internal parameters to best lower the loss.

# Create the loss function
loss_fn = nn.L1Loss() # MAE loss is same as L1Loss

# Create the optimizer
optimizer = torch.optim.SGD(params=model_0.parameters(), # parameters of target model to optimize
                            lr=0.01)
# learning rate (how much the optimizer should change parameters at each step, higher=more (less stable), lower=less (might take a long time))


# Pytorch training loop

# Create empty loss lists to track values
train_loss_values = []
test_loss_values = []
epoch_count = []

# creat a for loop for 10 training epoch
for epoch in range(100):
    ### Training

    #put the model in training model(this is the default state of a model)
    model_0.train()

    # 1. Forward pass on train data using the forward() method inside
    y_pred = model_0(GetDataReady.X_train)

    # 2. Calculate the loss (how different are our models predictions to the ground truth)
    loss = loss_fn(y_pred, GetDataReady.y_train)

    # 3. Zero grad of the optimizer
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Progress the optimizer
    optimizer.step()

    ### Testing

    # Put the model in evaluation mode
    model_0.eval()

    with torch.inference_mode():
        # 1. Forward pass on test data
        test_pred = model_0(GetDataReady.X_test)

        # 2. Caculate loss on test data
        test_loss = loss_fn(test_pred, GetDataReady.y_test.type(
            torch.float))  # predictions come in torch.float datatype, so comparisons need to be done with tensors of the same type

        # Print out what's happening
        if epoch % 1 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")



# Plot the loss curves
plt.plot(epoch_count, train_loss_values, label="Train loss")
plt.plot(epoch_count, test_loss_values, label="Test loss")
plt.title("Training and test loss curves")
plt.ylabel("Loss")
plt.xlabel("Epochs")
plt.legend()
plt.show()