import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path

import data_setup, engine, utils, model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setup path to data folder
data_path = Path('D:\Python Program\Pytorch Learning\modular\going_modular\data')
image_path = data_path / "pizza_steak_sushi"

# Setup Dirs
train_dir = image_path / "train"
test_dir = image_path / "test"

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# Create a transform pipline manually
manual_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


if __name__ == "__main__":
    # Create training and testing DataLoaders as well as get a list of class names
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(train_dir=str(train_dir),
                                                                                   test_dir=str(test_dir),
                                                                                   transform=model.auto_transforms,
                                                                                   # resize, convert images to between 0 & 1 and normalize them
                                                                                   batch_size=32)  # set mini-batch size to 32
    print(train_dataloader, test_dataloader, class_names, sep='\n')

    # Get the model from model.py
    model1 = model.model
    # Freeze all base layers in the "features" section of the model (the feature extractor) by setting requires_grad=False
    for param in model1.features.parameters():
        param.requires_grad = False
    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Get the length of class_names (one output unit for each class)
    output_shape = len(class_names)

    # Recreate the classifier layer and seed it to the target device
    model1.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True),
        torch.nn.Linear(in_features=1280,
                        out_features=output_shape,  # same number of output units as our number of classes
                        bias=True))

    model1.to(device)

    # TRAINING
    # Define loss and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model1.parameters(), lr=0.001)

    # Training with train() method
    # Set the random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Start the timer
    from timeit import default_timer as timer

    start_time = timer()

    # Setup training and save the results
    results = engine.train(model=model1,
                           train_dataloader=train_dataloader,
                           test_dataloader=test_dataloader,
                           optimizer=optimizer,
                           loss_fn=loss_fn,
                           epochs=5,
                           device=device)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"[INFO] Total training time: {end_time - start_time:.3f} seconds")

    # Get the plot_loss_curves() function from helper_functions.py, download the file if we don't have it
    from helper_functions import plot_loss_curves

    # Plot the loss curves of our model
    plot_loss_curves(results)
    plt.show()

    # Get a random list of image paths from test set
    import random

    num_images_to_plot = 3
    test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))  # get list all image paths from test data
    test_image_path_sample = random.sample(population=test_image_path_list,  # go through all of the test image paths
                                           k=num_images_to_plot)  # randomly select 'k' image paths to pred and plot

    # Make predictions on and plot the images
    for image_path in test_image_path_sample:
        utils.pred_and_plot_image(model=model1,
                            image_path=image_path,
                            class_names=class_names,
                            transform=model.auto_transforms, # optionally pass in a specified transform from our pretrained model weights
                            image_size=(224, 224))
