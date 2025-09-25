import torch

# Create a random tensor of size (3, 4)
random_tensor = torch.rand(size=(3, 4))
print(random_tensor, random_tensor.dtype, sep='\n')

#Zeros and ones
#creat a tensor filled with 0 or 1 via torch.zeros() and torch.ones()
zeros=torch.zeros(size=(3,3))
print(zeros, zeros.dtype, sep='\n')#datatype: float32
#creat a tensor of all 0s or 1s with the same shape as a previous tensor
#torch.zeros_like(input=name_of_the_previous_tensor)
#torch.zeros_like(input=name_of_the_previous_tensor)
zeros=torch.zeros_like(random_tensor)
print(zeros)
zeros=torch.zeros(size=random_tensor.shape)#same function as the previous method
print(zeros)

#Creating a range and tensors like
#torch.arange(start, end, step)
# Create a range of values 0 to 10
zero_to_ten = torch.arange(start=0, end=10,step=1)
print(zero_to_ten, zero_to_ten.dtype, sep='\n')#datatype: int64