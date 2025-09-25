import torch

scalar=torch.tensor(3.3)#it is a scalar标量, in tensor-speak it's a zero dimension tensor.
print(scalar)#show the tensor
print(scalar.ndim)#.nidm is used to show the dimensions of a tensor

# Get the Python number within a tensor (only works with one-element tensors)
print(scalar.item())

#vector
##A vector is a single dimension tensor but can contain many numbers.
vector=torch.tensor([3,3,3])
print(vector.ndim)#a vector is a one dimension tensor

#Shape:Another important concept for tensors is their shape attribute.
#The shape tells you how the elements inside them are arranged.
print(vector.shape)

#matrix
MATRIX=torch.tensor([[3,3,3],[3,3,3],[3,3,3]])
print(MATRIX.ndim)
print(MATRIX.shape)

#tensor
TENSOR=torch.tensor([[[3,3,3],[3,3,3],[3,3,3]]])
print(TENSOR.ndim)
print(TENSOR.shape)

