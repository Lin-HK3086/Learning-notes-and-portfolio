import torch

# Create a tensor and check its datatype
Tensor = torch.arange(10., 100., 10.)
print(Tensor)
print(Tensor.dtype)#the default datatype is torch.float32

#creat a torch.int8 tensor by Tensor
tensor_int8=Tensor.type(torch.int8)
print(tensor_int8)
print(tensor_int8.dtype)