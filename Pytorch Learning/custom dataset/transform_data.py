import random
from pathlib import Path
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

# Set up train and test dataset path
image_path = Path("data/pizza_steak_sushi")
train_dir = image_path / "train"
test_dir = image_path / "test"

# Transform dataset
# 1. Resize the images using transforms.
#   Resize() (from about 512x512 to 64x64, the same shape as the images on the CNN Explainer website).
# 2. Flip our images randomly on the horizontal using transforms.
#   RandomHorizontalFlip() (this could be considered a form of data augmentation because it will artificially change our image data).
# 3. Turn our images from a PIL image to a PyTorch tensor using transforms.
#   ToTensor().

# Write transform for image
data_transform = transforms.Compose([
    # Resize the images to 64x64
    transforms.Resize(size=(64, 64)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
    # Turn the image into a torch.Tensor
    transforms.ToTensor() # this also converts all pixel values from 0 to 255 to be between 0.0 and 1.0
])

def plot_transformed_images(image_paths, transform, n=3, seed=42):
    """Plots a series of random images from image_paths.

    Will open n image paths from image_paths, transform them
    with transform and plot them side by side.

    Args:
        image_paths (list): List of target image paths.
        transform (PyTorch Transforms): Transforms to apply to images.
        n (int, optional): Number of images to plot. Defaults to 3.
        seed (int, optional): Random seed for the random generator. Defaults to 42.
    """
    random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSize: {f.size}")
            ax[0].axis("off")

            # Transform and plot image
            # Note: permute() will change shape of image to suit matplotlib
            # (PyTorch default is [C, H, W] but Matplotlib is [H, W, C])
            transformed_image = transform(f).permute(1, 2, 0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize: {transformed_image.shape}")
            ax[1].axis("off")

            fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)


if __name__ == "__main__" :
    print(train_dir.is_dir())
    print(test_dir.is_dir())
    # Get all image paths (* means "any combination")
    image_path_list = list(image_path.glob("*/*/*.jpg"))
    plot_transformed_images(image_path_list,
                            transform=data_transform,
                            n=3)
    plt.show()