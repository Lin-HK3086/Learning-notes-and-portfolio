import torch

print(torch.__version__)  # pytorch版本
print(torch.version.cuda)  # cuda版本
print(torch.cuda.is_available())  # 查看cuda是否可用