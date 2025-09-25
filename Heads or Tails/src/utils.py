import os
import random
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
import torch
from pathlib import Path

def split_files_into_two_folders(source_folder, output_folder_1, output_folder_2, split_ratio=0.5):
    """
    随机将一个文件夹中的文件分成两个新文件夹。

    Args:
        source_folder (str): 原始文件夹的路径。
        output_folder_1 (str): 第一个输出文件夹的名称。
        output_folder_2 (str): 第二个输出文件夹的名称。
        split_ratio (float): 第一个文件夹的文件比例，默认值为 0.5。
    """
    # 确保源文件夹存在
    if not os.path.isdir(source_folder):
        print(f"错误：源文件夹 '{source_folder}' 不存在。")
        return

    # 获取所有文件的列表
    all_files = [f for f in os.listdir(source_folder) if os.path.isfile(os.path.join(source_folder, f))]

    if not all_files:
        print("源文件夹中没有文件。")
        return

    # 随机打乱文件列表
    random.shuffle(all_files)

    # 计算两个文件夹的文件数量
    split_index = int(len(all_files) * split_ratio)
    files_folder_1 = all_files[:split_index]
    files_folder_2 = all_files[split_index:]

    # 创建输出文件夹
    os.makedirs(output_folder_1, exist_ok=True)
    os.makedirs(output_folder_2, exist_ok=True)

    print(f"开始划分文件...")

    # 将文件移动到第一个文件夹
    for file_name in tqdm(files_folder_1, desc=f"移动到 '{output_folder_1}'"):
        src_path = os.path.join(source_folder, file_name)
        dst_path = os.path.join(output_folder_1, file_name)
        shutil.move(src_path, dst_path)

    # 将文件移动到第二个文件夹
    for file_name in tqdm(files_folder_2, desc=f"移动到 '{output_folder_2}'"):
        src_path = os.path.join(source_folder, file_name)
        dst_path = os.path.join(output_folder_2, file_name)
        shutil.move(src_path, dst_path)

    print("\n文件划分完成！")
    print(f"总文件数: {len(all_files)}")
    print(f"'{output_folder_1}' 中的文件数: {len(files_folder_1)}")
    print(f"'{output_folder_2}' 中的文件数: {len(files_folder_2)}")


def plot_transformed_images(image_path, transform, n=3, seed=42):
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
    image_paths = list(image_path.glob("*/*.jpg"))
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
    plt.show()

def walk_through_dir(dir_path):
    """
    Walks through dir_path returning its contents.
    Args:
      dir_path (str or pathlib.Path): target directory

    Returns:
      A print out of:
        number of subdiretories in dir_path
        number of images (files) in each subdirectory
        name of each subdirectory
    """
    for dirpath, dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.

  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)

  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)