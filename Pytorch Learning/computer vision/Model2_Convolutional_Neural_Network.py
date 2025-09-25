from Dataset import *
import torch
from torch import nn

device = torch.device("cuda")
# CNN is known for their capabilities to find patterns in visual data.
# Create a convolutional neural network
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from:
    https://poloclub.github.io/cnn-explainer/
    """

    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,  # how big is the square that's going over the image?
                      stride=1,  # default
                      padding=1),
            # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2)  # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units * 7 * 7,
                      out_features=output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x
torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1,
                            hidden_units=10,
                            output_shape=len(train_data.classes)).to(device)


if __name__ == '__main__':
    # Setup loss and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model_2.parameters(),
                                lr=0.1)

    torch.manual_seed(42)

    # Measure time
    from timeit import default_timer as timer

    train_time_start_model_2 = timer()

    # Train and test model
    from classification.helper_functions import accuracy_fn
    from tqdm import tqdm


    def train_step(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   optimizer: torch.optim.Optimizer,
                   accuracy_fn,
                   device: torch.device = device):
        train_loss, train_acc = 0, 0
        model.to(device)
        for batch, (X, y) in enumerate(data_loader):
            # Send data to GPU
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            y_pred = model(X)

            # 2. Calculate loss
            loss = loss_fn(y_pred, y)
            train_loss += loss
            train_acc += accuracy_fn(y_true=y,
                                     y_pred=y_pred.argmax(dim=1))  # Go from logits -> pred labels

            # 3. Optimizer zero grad
            optimizer.zero_grad()

            # 4. Loss backward
            loss.backward()

            # 5. Optimizer step
            optimizer.step()

        # Calculate loss and accuracy per epoch and print out what's happening
        train_loss /= len(data_loader)
        train_acc /= len(data_loader)
        print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")


    def test_step(data_loader: torch.utils.data.DataLoader,
                  model: torch.nn.Module,
                  loss_fn: torch.nn.Module,
                  accuracy_fn,
                  device: torch.device = device):
        test_loss, test_acc = 0, 0
        model.to(device)
        model.eval()  # put model in eval mode
        # Turn on inference context manager
        with torch.inference_mode():
            for X, y in data_loader:
                # Send data to GPU
                X, y = X.to(device), y.to(device)

                # 1. Forward pass
                test_pred = model(X)

                # 2. Calculate loss and accuracy
                test_loss += loss_fn(test_pred, y)
                test_acc += accuracy_fn(y_true=y,
                                        y_pred=test_pred.argmax(dim=1)  # Go from logits -> pred labels
                                        )

            # Adjust metrics and print out
            test_loss /= len(data_loader)
            test_acc /= len(data_loader)
            print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")


    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch: {epoch}\n---------")
        train_step(data_loader=train_dataloader,
                   model=model_2,
                   loss_fn=loss_fn,
                   optimizer=optimizer,
                   accuracy_fn=accuracy_fn,
                   device=device
                   )
        test_step(data_loader=test_dataloader,
                  model=model_2,
                  loss_fn=loss_fn,
                  accuracy_fn=accuracy_fn,
                  device=device
                  )
    train_time_end_model_2 = timer()
    total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                                end=train_time_end_model_2,
                                                device=device)


    # Get model_2 results
    def eval_model(model: torch.nn.Module,
                   data_loader: torch.utils.data.DataLoader,
                   loss_fn: torch.nn.Module,
                   accuracy_fn,
                   device: torch.device = device):
        """Evaluates a given model on a given dataset.

        Args:
            model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
            data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
            loss_fn (torch.nn.Module): The loss function of model.
            accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
            device (str, optional): Target device to compute on. Defaults to device.

        Returns:
            (dict): Results of model making predictions on data_loader.
        """
        loss, acc = 0, 0
        model.eval()
        with torch.inference_mode():
            for X, y in data_loader:
                # Send data to the target device
                X, y = X.to(device), y.to(device)
                y_pred = model(X)
                loss += loss_fn(y_pred, y)
                acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

            # Scale loss and acc
            loss /= len(data_loader)
            acc /= len(data_loader)
        return {"model_name": model.__class__.__name__,  # only works when model was created with a class
                "model_loss": loss.item(),
                "model_acc": acc}


    torch.save(model_2.state_dict(), "model_2_checkpoint.pth")

    model_2_results = eval_model(
        model=model_2,
        data_loader=test_dataloader,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn
    )
    print(model_2_results)