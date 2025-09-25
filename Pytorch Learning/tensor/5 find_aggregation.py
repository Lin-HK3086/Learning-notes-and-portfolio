import torch

# aggregation: min, max, mean, sum, etc.

# reat a tensor
x = torch.arange(0,100,10)
print(f"creat a tensor: {x}")

# let's perform some aggregation
print(f"Minimum: {x.min()}")
print(f"Maximum: {x.max()}")
# print(f"Mean: {x.mean()}") # this will error
print(f"Mean: {x.type(torch.float32).mean()}") # won't work without float datatype
print(f"Sum: {x.sum()}")

#You can also do the same as above with torch methods.
print(torch.max(x), torch.min(x), torch.mean(x.type(torch.float32)), torch.sum(x))