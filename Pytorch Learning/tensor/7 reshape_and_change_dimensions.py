import torch

# there are some popular to reshape and change dimensions
# Create a tensor
import torch
x = torch.arange(1., 8.)
print(x, x.shape)
print("-------------------------1---------------------------")

# torch.reshape(input, shape)	Reshapes input to shape (if compatible), can also use torch.Tensor.reshape().
x_reshaped = torch.reshape(x, (1,1,7))
print(x_reshaped, x_reshaped.shape)
print("-------------------------2---------------------------")

# Tensor.view(shape)	Returns a view of the original tensor in a different shape but shares the same data as the original tensor.
z = x.view(1, 7)
print(z, z.shape)
#changing the view changes the original tensor too.
# Changing z changes x
z[:, 0] = 5
print(z, x)
print("-------------------------3---------------------------")

# torch.stack(tensors, dim=0)	Concatenates a sequence of tensors along a new dimension (dim), all tensors must be same size.
# Stack tensors on top of each other
x_stacked_dim0 = torch.stack([x, x, x, x], dim=0)
# try changing dim to dim=1 and see what happens
x_stacked_dim1 = torch.stack([x, x, x, x], dim=1)
print(f"while dim=0: {x_stacked_dim0}")
print(f"while dim=1: {x_stacked_dim1}")
print("-------------------------4---------------------------")

# torch.squeeze(input)	Squeezes input to remove all the dimentions with value 1.
print(f"Previous tensor: {x_reshaped}")
print(f"Previous shape: {x_reshaped.shape}\n")
# Remove extra dimension from x_reshaped
x_squeezed = x_reshaped.squeeze()#dim=0 default, dim is the dimension of the squeezed tensor
print(x_squeezed, x_squeezed.shape)
print(f"New tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}\n")
print("-------------------------5---------------------------")

# torch.unsqueeze(input, dim)	Returns input with a dimension value of 1 added at dim.
# Add extra dimension from x_reshaped
x_squeezed = x_reshaped.unsqueeze(dim=0)
print(x_squeezed, x_squeezed.shape)
print(f"New tensor: {x_squeezed}")
print(f"New shape: {x_squeezed.shape}\n")
print("-------------------------6---------------------------")

# torch.permute(input, dims)	Returns a view of the original input with its dimensions permuted (rearranged) to dims.
# Create tensor with specific shape
x_original = torch.rand(size=(224, 224, 3))
# Permute the original tensor to rearrange the axis order
x_permuted = x_original.permute(2, 0, 1) # shifts axis 0->1, 1->2, 2->0
print(f"Previous shape: {x_original.shape}")
print(f"New shape: {x_permuted.shape}")