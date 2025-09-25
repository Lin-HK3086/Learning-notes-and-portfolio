from torchvision import transforms
from transform_data import *

# Machine learning is all about harnessing the power of randomness
# and research shows that random transforms
# (like transforms.RandAugment() and transforms.TrivialAugmentWide())
# generally perform better than hand-picked transforms.

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    # The main parameter to pay attention to in transforms.TrivialAugmentWide() is num_magnitude_bins
    # It defines how much of a range an intensity value will be picked to apply a certain transform,
    # 0 being no range and 31 being maximum range (highest chance for highest intensity).
    transforms.ToTensor() # use ToTensor() last to get everything between 0 & 1
])

# Don't need to perform augmentation on the test data
test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Get all image paths
image_path_list = list(image_path.glob("*/*/*.jpg"))

# Plot random images
plot_transformed_images(
    image_paths=image_path_list,
    transform=train_transforms,
    n=3,
    seed=None
)
plt.show()