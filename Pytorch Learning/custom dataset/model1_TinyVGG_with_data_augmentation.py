from model0_TinyVGG_without_data_augmentation import *

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Create training transform with TrivialAugment
    train_transform_trivial_augment = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    # Create testing transform (no data augmentation)
    test_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Turn image folders into Datasets
    train_dir = Path('D:\Python Program\Pytorch Learning\custom dataset\data\pizza_steak_sushi\\train')
    test_dir = Path('D:\Python Program\Pytorch Learning\custom dataset\data\pizza_steak_sushi\\test')
    train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
    test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

    # Turn Datasets into DataLoader's
    BATCH_SIZE = 32
    NUM_WORKERS = 2

    torch.manual_seed(42)
    train_dataloader_augmented = DataLoader(train_data_augmented,
                                            batch_size=BATCH_SIZE,
                                            shuffle=True,
                                            num_workers=NUM_WORKERS)

    test_dataloader_simple = DataLoader(test_data_simple,
                                        batch_size=BATCH_SIZE,
                                        shuffle=False,
                                        num_workers=NUM_WORKERS)

    # Create model_1 and send it to the target device
    torch.manual_seed(42)
    model_1 = TinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=len(train_data_augmented.classes)).to(device)

    # Set random seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    # Set number of epochs
    NUM_EPOCHS = 20

    # Setup loss function and optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_1.parameters(), lr=0.001)

    # Start the timer
    from timeit import default_timer as timer

    start_time = timer()

    # Train model_1
    model_1_results = train(model=model_1,
                            train_dataloader=train_dataloader_augmented,
                            test_dataloader=test_dataloader_simple,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)

    # End the timer and print out how long it took
    end_time = timer()
    print(f"Total training time: {end_time - start_time:.3f} seconds")
    torch.save(model_1.state_dict(), "model_1.pth")

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
    plot_loss_curves(model_1_results)
    plt.show()