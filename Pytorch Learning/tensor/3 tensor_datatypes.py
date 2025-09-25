import torch

#There are many different tensor datatypes available in PyTorch. Some are specific for CPU and some are better for GPU.
#32-bit floating point:torch.float32 or torch.float
#16-bit floating point:torch.float16 or torch.half
#64-bit floating point:torch.float64 or torch.double

#use parameter "dtype" to creat tensors with specific datatypes.
# Default datatype for tensors is float32
float_32_tensor = torch.tensor([3.0, 6.0, 9.0],
                               dtype=None, # defaults to None, which is torch.float32 or whatever datatype is passed
                               device=None, # defaults to None, which uses the default tensor type
                               requires_grad=False) # if True, operations performed on the tensor are recorded
print(float_32_tensor.shape, float_32_tensor.dtype, float_32_tensor.device)

#creat a tensor with dtype=torch.float16
float_16_tensor = torch.tensor([[3.0, 6.0, 9.0]],dtype=torch.float16)
print(float_16_tensor.dtype)