import torch
from torch import nn
from torchvision import transforms

import model,utils,data,engine

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Download 10 percent and 20 percent training data (if necessary)# Creat a helper function to set seeds
def set_seed(seed: int=33):
    # Set the seed for general torch operations.
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":

    data_10_percent_path = data.download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
        destination="pizza_steak_sushi")

    data_20_percent_path = data.download_data(
        source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
        destination="pizza_steak_sushi_20_percent")

    # Setup training directory paths
    train_dir_10_percent = data_10_percent_path / "train"
    train_dir_20_percent = data_20_percent_path / "train"

    # Setup testing directory paths (note: use the same test dataset for both to compare the results)
    test_dir = data_10_percent_path / "test"

    # Check the directories
    print(f"Training directory 10%: {train_dir_10_percent}")
    print(f"Training directory 20%: {train_dir_20_percent}")
    print(f"Testing directory: {test_dir}")

    # Create a transform to normalize data distribution to be inline with ImageNet
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # values per colour channel [red, green, blue]
                                     std=[0.229, 0.224, 0.225])  # values per colour channel [red, green, blue]

    # Compose transforms into a pipeline
    simple_transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 1. Resize the images
        transforms.ToTensor(),  # 2. Turn the images into tensors with values between 0 & 1
        normalize  # 3. Normalize the images so their distributions match the ImageNet dataset
    ])

    BATCH_SIZE = 32

    # Create 10% training and test DataLoaders
    train_dataloader_10_percent, test_dataloader, class_names = data.create_dataloaders(
        train_dir=str(train_dir_10_percent),
        test_dir=str(test_dir),
        transform=simple_transform,
        batch_size=BATCH_SIZE
        )

    # Create 20% training and test data DataLoders
    train_dataloader_20_percent, _, _ = data.create_dataloaders(train_dir=str(train_dir_20_percent),
                                                                test_dir=str(test_dir),
                                                                transform=simple_transform,
                                                                batch_size=BATCH_SIZE
                                                                )

    # Setup experiment variable
    # Create epochs list
    num_epochs = [5, 10]

    # Create models list (need to create a new model for each experiment)
    models = ["effnetb0", "effnetb2"]

    # Create dataloaders dictionary for various dataloaders
    train_dataloaders = {"data_10_percent": train_dataloader_10_percent,
                         "data_20_percent": train_dataloader_20_percent}

    # 1. Set the random seeds
    set_seed()

    # 2. Keep track of experiment numbers
    experiment_number = 0

    # 3. Loop through each DataLoader
    print("traoning .........................")
    for dataloader_name, train_dataloader in train_dataloaders.items():

        # 4. Loop through each number of epochs
        for epochs in num_epochs:

            # 5. Loop through each model name and create a new model based on the name
            for model_name in models:

                # 6. Create information print outs
                experiment_number += 1
                print(f"[INFO] Experiment number: {experiment_number}")
                print(f"[INFO] Model: {model_name}")
                print(f"[INFO] DataLoader: {dataloader_name}")
                print(f"[INFO] Number of epochs: {epochs}")

                # 7. Select the model
                if model_name == "effnetb0":
                    model_ = model.create_effnetb0() # creates a new model each time (important because we want each experiment to start from scratch)
                else:
                    model_ = model.create_effnetb2()  # creates a new model each time (important because we want each experiment to start from scratch)

                # 8. Create a new loss and optimizer for every model
                loss_fn = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(params=model_.parameters(), lr=0.001)

                # 9. Train target model with target dataloaders and track experiments
                engine.train_with_write(model=model_,
                      train_dataloader=train_dataloader,
                      test_dataloader=test_dataloader,
                      optimizer=optimizer,
                      loss_fn=loss_fn,
                      epochs=epochs,
                      device=device,
                      writer=utils.create_writer(experiment_name=dataloader_name,
                                                model_name=model_name,
                                                extra=f"{epochs}_epochs"))

                # 10. Save the model to file so we can get back the best model
                save_filepath = f"D:\Python Program\Pytorch Learning\experiment tracking\more_models_training\models\\07_{model_name}_{dataloader_name}_{epochs}_epochs.pth"
                utils.save_model(model=model_,
                           target_dir="models",
                           model_name=save_filepath)
                print("-" * 50 + "\n")