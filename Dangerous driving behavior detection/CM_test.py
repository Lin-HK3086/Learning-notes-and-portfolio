# Imports
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

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

    # 加载测试数据集
    ds_test_ = datasets.ImageFolder(
        'D:\Python Program\Summer Reasurch\DriverImage\imgs\\test',
        transform=transform
    )
    print(">> Test data is ready.")

    # 获取类别名称
    class_names = ds_test_.classes

    # 创建DataLoader
    ds_test = DataLoader(
        ds_test_,
        batch_size=batch_size,
        shuffle=False,
        num_workers=3,
        pin_memory=True
    )

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
    model.eval()

    criterion = nn.CrossEntropyLoss()

    # 用于存储所有预测和真实标签
    all_preds = []
    all_labels = []

    correct_predictions = 0
    total_samples = 0
    running_loss = 0.0

    test_bar = tqdm(ds_test, desc="Testing Model", leave=True)
    with torch.no_grad():
        for inputs, labels in test_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)

            # 收集预测和标签
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    # 计算并打印测试结果
    accuracy = 100 * correct_predictions / total_samples
    avg_loss = running_loss / total_samples

    print(f"\n>> Test Accuracy of the model on the validation images: {accuracy:.4f}%")
    print(f">> Test Loss of the model on the test images: {avg_loss:.4f}\n")

    # 生成混淆矩阵
    print(">> Generating confusion matrix...")
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('ResNet-152 Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # 保存混淆矩阵图像
    plt.tight_layout()
    cm_path = os.path.join(os.path.dirname(model_path), 'confusion_matrix.png')
    plt.savefig(cm_path)
    print(f">> Confusion matrix saved to: {cm_path}")
    plt.show()

    # 计算并打印每个类别的准确率
    print("\n>> Per-class accuracy:")
    class_correct = np.zeros(num_classes)
    class_total = np.zeros(num_classes)

    for label, pred in zip(all_labels, all_preds):
        class_correct[label] += (pred == label)
        class_total[label] += 1

    for i in range(num_classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"{class_names[i]:<15}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
        else:
            print(f"{class_names[i]:<15}: No samples")


if __name__ == '__main__':
    model_file = 'D:\Python Program\Summer Reasurch\ResNet-152\\train_all_lr_00003\\best resut\\best_resnet152_model.pt'
    num_classes = 10

    if os.path.exists(model_file):
        test_model(model_file, num_classes=num_classes)
    else:
        print(f"Error: Model file '{model_file}' not found. Please check the path.")