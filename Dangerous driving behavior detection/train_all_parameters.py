 # Imports
import os
import warnings
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import pickle
import pandas as pd
from tqdm import tqdm


# Reproducibility
def set_seed(seed=31415):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


set_seed(31415)

# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)
plt.rc('image', cmap='magma')
warnings.filterwarnings("ignore")

# Data Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

if __name__ == '__main__':
    # --- *** 可调整的参数 *** ---
    train_batch_size = 64
    valid_batch_size = 16
    num_data_workers = 3
    num_classes = 10  # 您的数据集类别数
    # --- ****************** ---

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">> Using device: {device}")
    if torch.cuda.is_available():
        print(f"CUDA可用: {torch.cuda.is_available()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name(0)}")

    # Load training and validation sets
    ds_train_ = datasets.ImageFolder(
        'D:\Python Program\Summer Reasurch\DriverImage\imgs\\train',
        transform=transform
    )
    print(">>train data is ready.")

    ds_valid_ = datasets.ImageFolder(
        'D:\Python Program\Summer Reasurch\DriverImage\imgs\\valid',
        transform=transform
    )
    print(">>valid data is ready.")

    # DataLoaders using the adjustable parameters
    ds_train = DataLoader(
        ds_train_,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_data_workers,
        pin_memory=True
    )

    ds_valid = DataLoader(
        ds_valid_,
        batch_size=valid_batch_size,
        shuffle=False,
        num_workers=num_data_workers,
        pin_memory=True
    )

    # --- *** 核心修改 1: 加载整个模型并替换最后的分类器 *** ---
    model = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

    print(model.fc)
    # 获取最后一个全连接层的输入特征数
    num_ftrs = model.fc.in_features
    # 替换最后一个全连接层，以适应新的类别数
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(model.fc)

    # 移除之前冻结参数的代码
    # for param in pretrained_base.parameters():
    #     param.requires_grad = False
    model = model.to(device)
    print(">> 训练整个ResNet-152模型已准备就绪。")
    # -----------------------------------------------------------

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # --- *** 核心修改 2: 优化器针对整个模型的参数 *** ---
    optimizer = optim.Adam(model.parameters(), lr=0.0003)
    # ---------------------------------------------------

    print("training-------------------------------------->")

    # Training loop
    num_epochs = int(input("epochs: "))
    total_epochs = num_epochs
    epoch = 0
    pre_acc = 0
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    while num_epochs > 0:
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        train_bar = tqdm(ds_train, desc=f"Epoch {epoch + 1}/{total_epochs} [Train]", leave=True)
        for i, (inputs, labels) in enumerate(train_bar):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            true_labels = labels
            correct_predictions += (predicted == true_labels).sum().item()
            total_samples += labels.size(0)

            train_bar.set_postfix(loss=f"{running_loss / total_samples:.4f}",
                                  acc=f"{correct_predictions / total_samples:.4f}")

        epoch_loss = running_loss / total_samples
        epoch_accuracy = correct_predictions / total_samples
        history['loss'].append(epoch_loss)
        history['accuracy'].append(epoch_accuracy)

        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct_predictions = 0
        val_total_samples = 0

        val_bar = tqdm(ds_valid, desc=f"Epoch {epoch + 1}/{total_epochs} [Valid]", leave=True)
        with torch.no_grad():
            for inputs, labels in val_bar:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                true_labels = labels
                val_correct_predictions += (predicted == true_labels).sum().item()
                val_total_samples += labels.size(0)

                val_bar.set_postfix(loss=f"{val_running_loss / val_total_samples:.4f}",
                                    acc=f"{val_correct_predictions / val_total_samples:.4f}")

        val_epoch_loss = val_running_loss / val_total_samples
        val_epoch_accuracy = val_correct_predictions / val_total_samples
        history['val_loss'].append(val_epoch_loss)
        history['val_accuracy'].append(val_epoch_accuracy)

        print(f'Epoch [{epoch + 1}/{total_epochs}] Summary: '
              f'Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, '
              f'Val Loss: {val_epoch_loss:.4f}, Val Accuracy: {val_epoch_accuracy:.4f}\n')
        if (val_epoch_accuracy > pre_acc):
            torch.save(model.state_dict(), 'best_resnet152_model.pt')
            print(f"Best model is saved. acc:{val_epoch_accuracy:.4f}")
            pre_acc = val_epoch_accuracy

        epoch += 1
        num_epochs -= 1
        if num_epochs == 0:
            num_epochs = int(input("more epochs: "))
            total_epochs += num_epochs

    # Save the trained model
    torch.save(model.state_dict(), 'final_resnet152_model.pt')
    print("Model saved as final_resnet152_model.pt")

    history_frame = pd.DataFrame(history)
    print(history_frame)
    history_frame.loc[:, ['loss', 'val_loss']].plot()
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('loss_plot.png')

    history_frame.loc[:, ['accuracy', 'val_accuracy']].plot()
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.savefig('accuracy_plot.png')

    print("\nPlots saved as loss_plot.png and accuracy_plot.png")

    # Save training history
    history_frame = pd.DataFrame(history)
    history_frame.to_csv('training_history.csv', index=False)
    print("Training history saved as training_history.csv")