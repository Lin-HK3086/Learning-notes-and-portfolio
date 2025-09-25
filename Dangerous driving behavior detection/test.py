# Imports
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


# Define the custom classification head
class CustomClassifier(nn.Module):
    def __init__(self, num_classes=10, in_features=2048):
        super(CustomClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


# Combine pretrained base and custom classifier (for old model)
class ResNet152Transfer(nn.Module):
    def __init__(self, pretrained_model, num_classes=10):
        super(ResNet152Transfer, self).__init__()
        # 注意: 这里的结构必须与你训练时使用的完全一致
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        self.classifier = CustomClassifier(num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def test_model(model_path, num_classes=10, batch_size=128):
    """
    加载并测试已保存的模型。
    Args:
        model_path (str): 模型文件（.pt）的路径。
        num_classes (int): 数据集的类别数量。
        batch_size (int): 测试时的批处理大小。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Using device for testing: {device}")

    # **核心修复**：根据模型保存时的结构构建模型
    try:
        # 尝试使用ResNet152Transfer结构加载
        pretrained_base = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)
        model = ResNet152Transfer(pretrained_base, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(">> Loaded weights for ResNet152Transfer model.")
    except RuntimeError as e:
        # 如果加载失败，则尝试使用完整ResNet152结构加载
        print(f"Failed to load as ResNet152Transfer: {e}")
        model = models.resnet152(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(">> Loaded weights for a fully fine-tuned ResNet-152 model.")

    model = model.to(device)

    ds_test_ = datasets.ImageFolder(
        'D:\Python Program\Summer Reasurch\DriverImage\imgs\\test',
        transform=transform
    )
    print(">> test data is ready.")

    ds_test = DataLoader(
        ds_test_,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True
    )

    model.eval()
    correct_predictions = 0
    total_samples = 0
    running_loss = 0.0

    criterion = nn.CrossEntropyLoss()

    test_bar = tqdm(ds_test, desc="Testing Model", leave=True)
    with torch.no_grad():
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    accuracy = 100 * correct_predictions / total_samples
    print(f"\n>> Test Accuracy of the model on the validation images: {accuracy:.4f}%")
    print(f"\n>> Test Loss of the model on the test images: {running_loss / total_samples:.4f}\n")


if __name__ == '__main__':
    model_file = 'D:\Python Program\Summer Reasurch\ResNet-152\\train_all_lr_00003\\best resut\\best_resnet152_model.pt'
    num_classes = 10

    if os.path.exists(model_file):
        test_model(model_file, num_classes=num_classes)
    else:
        print(f"Error: Model file '{model_file}' not found. Please check the path.")