import torch

# check for GPU
print(torch.cuda.is_available())

# Set device type
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
print("the num of GPUs:"+str(torch.cuda.device_count()))

# creating a tensor and putting it on the GPU
# Create tensor (default on CPU)
tensor = torch.tensor([1, 2, 3])

# Tensor not on GPU
print(tensor, tensor.device)

# Move tensor to GPU (if available)
tensor_on_gpu = tensor.to(device)
print(tensor_on_gpu)

# if you want to interact with you tensor with numpy(numpy does not leverage the GPU), you have to move the tensor back to CPU.
# Instead, copy the tensor back to cpu
tensor_back_on_cpu = tensor_on_gpu.cpu().numpy()
print(tensor_back_on_cpu,tensor_on_gpu)