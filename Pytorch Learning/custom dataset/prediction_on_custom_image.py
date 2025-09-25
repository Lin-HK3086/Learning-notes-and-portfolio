from model0_TinyVGG_without_data_augmentation import*
import requests
import torchvision

if __name__ == "__main__":
    # Setup custom image path
    data_path = Path("D:\Python Program\Pytorch Learning\custom dataset\data")
    custom_image_path = data_path / "04-pizza-dad.jpeg"

    # Download the image if it doesn't already exist
    if not custom_image_path.is_file():
        with open(custom_image_path, "wb") as f:
            # When downloading from GitHub, need to use the "raw" file link
            request = requests.get(
                "https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
            print(f"Downloading {custom_image_path}...")
            f.write(request.content)
    else:
        print(f"{custom_image_path} already exists, skipping download.")

    # Read in custom image
    custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))
    # Print out image data
    print(f"image tensor:\n{custom_image_uint8}")
    print(f"image shape: {custom_image_uint8.shape}")
    print(f"image datatype: {custom_image_uint8.dtype}")

    # Now our image is tensor format, however, is this image format compatible with our model?
    # Our custom_image tensor is of datatype torch.uint8 and its values are between [0, 255].
    # But our model takes image tensors of datatype torch.float32 and with values between [0, 1].
    # We need to convert it to the same format as the data our model is trained on
    custom_image = custom_image_uint8.type(torch.float32)
    # Drive the image pixel value by 255 to get them between [0,1]
    custom_image = custom_image / 255

    # Plot custom image
    plt.imshow(custom_image.permute(1, 2,
                                    0))  # need to permute image dimensions from CHW -> HWC otherwise matplotlib will error
    plt.title(f"Image shape: {custom_image.shape}")
    plt.axis(False)
    plt.show()

    # Get the same size as the images our model was trained on
    # Creat transform to resize image
    custom_image_transform = transforms.Compose([
        transforms.Resize((64, 64))
    ])
    # Transform target image
    custom_image_transformed = custom_image_transform(custom_image)
    # It is not enough now
    # There is one dimension we forgot about, the batch size
    # Add an extra dimension to image
    custom_image_transformed_with_batch_size = custom_image_transformed.unsqueeze(dim=0)

    model_1 = TinyVGG(3, 10, 3).to(torch.device('cuda:0'))
    model_1.load_state_dict(torch.load("model_1.pth"))
    model_1.eval()
    with torch.inference_mode():
        custom_image_pred = model_1(custom_image_transformed_with_batch_size.to(torch.device('cuda:0')))

    # Print out predication logits
    print(f"Prediction logits: {custom_image_pred}")
    # Convert logits ->predication probabilities (using softmax() for multi-class classification)
    custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
    print(f"Prediction probabilities: {custom_image_pred_probs}")
    # Convert predication probabilities -> predication labels
    custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
    print(f"Prediction label: {custom_image_pred_label}")

    # Find the predicted label
    class_names = train_data_simple.classes
    custom_image_pred_class = class_names[custom_image_pred_label.cpu()]  # put pred label to CPU, otherwise will error
    print(f"Prediction class: {custom_image_pred_class}")


# Doing all of the above steps every time you'd like to make a prediction on a custom image would quickly become tedious.
# Let's put them all together in a function we can easily use over and over again
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str,
                        class_names: List[str] = None,
                        transform=None,
                        device: torch.device = device):
    """Makes a prediction on a target image and plots the image with its prediction."""

    # 1. Load in image and convert the tensor values to float32
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)

    # 2. Divide the image pixel values by 255 to get them between [0, 1]
    target_image = target_image / 255.

    # 3. Transform if necessary
    if transform:
        target_image = transform(target_image)

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
        # Add an extra dimension to the image
        target_image = target_image.unsqueeze(dim=0)

        # Make a prediction on image with an extra dimension and send it to the target device
        target_image_pred = model(target_image.to(device))

    # 6. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 7. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 8. Plot the image alongside the prediction and prediction probability
    plt.imshow(target_image.squeeze().permute(1, 2, 0))  # make sure it's the right size for matplotlib
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else:
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)
    plt.show()
