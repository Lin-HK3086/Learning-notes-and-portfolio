import torch
from torch import nn
from torchvision import transforms, datasets
from pathlib import Path
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from typing import Dict, List

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Create simple transform
simple_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])


# Build TinyVGG model
class TinyVGG(nn.Module):
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int)-> None:
        super().__init__()
        self.conv_block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3, # how big the square that is going over the image?
                      stride=1, # default
                      padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2) # default stride value is same as kernel_size
        )

        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_feature come from?
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*16*16,
                      out_features=output_shape),
        )

    def forward(self, x:torch.Tensor):
        x=self.conv_block_1(x)
        #print(x.shape)
        x=self.conv_block_2(x)
        #print(x.shape)
        x=self.classifier(x)
        #print(x.shape)
        return x

def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer):
    # Put model in train mode
    model.train() 
    # Setup train loss and train accuracy values
    train_loss, train_acc=0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # Forward pass
        y_pred = model(X)

        # Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # Optimizer zero grad
        optimizer.zero_grad()

        # Loss backward
        loss.backward()

        # Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module):
    # Put model in eval mode
    model.eval()
    # Setup test loss and test accuracy
    test_loss, test_acc=0, 0

    # Turn for inference context manager
    with torch.inference_mode():
        # Loop through Dataloader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # Forward pass
            test_pred_logits = model(X)

            # Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels==y).sum().item()/len(test_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int = 5):

    # Creat empty results dictionary
    results = {'train_loss': [],
               'test_loss': [],
               'train_acc': [],
               'test_acc': []}

    # Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn)

        # Print out what is happening
        print(
            f"Epoch: {epoch + 1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # Return the filled results at the end of the epochs
    return results

# 1. Load and transform data
train_dir = Path('D:\Python Program\Pytorch Learning\custom dataset\data\pizza_steak_sushi\\train')
test_dir = Path('D:\Python Program\Pytorch Learning\custom dataset\data\pizza_steak_sushi\\test')
train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir, transform=simple_transform)

# 2. Turn data into DataLoaders
# Setup batch size and number of workers
BATCH_SIZE = 32
NUM_WORKERS = 2   # Do not need too much cpu
print(f"Creating DataLoader's with batch size {BATCH_SIZE} and {NUM_WORKERS} workers.")

    # Create DataLoader's
train_dataloader_simple = DataLoader(train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)

test_dataloader_simple = DataLoader(test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    num_workers=NUM_WORKERS)

if __name__ == "__main__":

    torch.manual_seed(42)
    print(f"Creating DataLoader's with seed {42}")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_0 = TinyVGG(3, 10, len(train_data_simple.classes))
    model_0.to(device)
    print(model_0)
    print(f"Creating DataLoader's with device {device}")

    ## Try a forward pass on a single image
    # 1. Get a batch of images and labels from the DataLoader
    img_batch, label_batch = next(iter(train_dataloader_simple))
    print(img_batch.shape)

    # 2. Get a single image from the batch and unsqueeze the image so its shape fits the model
    img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
    print(f"Single image shape: {img_single.shape}\n")

    # 3. Perform a forward pass on a single image
    model_0.eval()
    with torch.inference_mode():
        pred = model_0(img_single.to(device))

    # 4. Print out what's happening and convert model logits -> pred probs -> pred label
    print(f"Output logits:\n{pred}\n")
    print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
    print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1), dim=1)}\n")
    print(f"Actual label:\n{label_single}")

    summary(model_0, input_size=[1, 3, 64, 64])  # do a test pass through of an example input size

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 5

    # Recreate an instance of TinyVGG
    model_0 = TinyVGG(input_shape=3,  # number of color channels (3 for RGB)
                      hidden_units=10,
                      output_shape=len(train_data_simple.classes)).to(device)

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(), lr=0.001)

    # Start the timer
    from timeit import default_timer as timer

    start_time = timer()

    # Train model_0
    model_0_results = train(model=model_0,
                            train_dataloader=train_dataloader_simple,
                            test_dataloader=test_dataloader_simple,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")


    def plot_loss_curves(results: Dict[str, List[float]]):
        """Plots training curves of a results dictionary.

        Args:
            results (dict): dictionary containing list of values, e.g.
                {"train_loss": [...],
                 "train_acc": [...],
                 "test_loss": [...],
                 "test_acc": [...]}
        """

        # Get the loss values of the results dictionary (training and test)
        loss = results['train_loss']
        test_loss = results['test_loss']

        # Get the accuracy values of the results dictionary (training and test)
        accuracy = results['train_acc']
        test_accuracy = results['test_acc']

        # Figure out how many epochs there were
        epochs = range(len(results['train_loss']))

        # Setup a plot
        plt.figure(figsize=(15, 7))

        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, loss, label='train_loss')
        plt.plot(epochs, test_loss, label='test_loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()

        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, accuracy, label='train_accuracy')
        plt.plot(epochs, test_accuracy, label='test_accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.legend()

    plot_loss_curves(model_0_results)
    plt.show()