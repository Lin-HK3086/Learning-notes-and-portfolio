import torch
import numpy as np

# two major method used for numpy to pytorch:
# torch.from_numpy(ndarray) - NumPy array -> PyTorch tensor.
# torch.Tensor.numpy() - PyTorch tensor -> NumPy array.

# NumPy array to tensor
array = np.arange(1.0, 8.0)#By default, NumPy arrays are created with the datatype float64
# NumPy array (float64) -> PyTorch tensor (float64) -> PyTorch tensor (float32)
tensor = torch.from_numpy(array).type(torch.float32)
print(array, tensor)

# Tensor to NumPy array
tensor = torch.ones(7) # create a tensor of ones with dtype=float32
numpy_tensor = tensor.numpy() # will be dtype=float32 unless changed
print(tensor, numpy_tensor)