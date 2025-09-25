import torch

#basic operation: addition, subtraction, multiplication
# Create a tensor of values and add/subtract/multiply
tensor = torch.tensor([1, 2, 3])
print(tensor + 10)
print(tensor - 10)
print(tensor * 10)#same as torch.mul(tensor, 10)

#Matrix multiplication (is all you need)
#Note: "@" in Python is the symbol for matrix multiplication.
tensor1 = torch.tensor([[1, 2, 3], [4, 5, 6]])
tensor2 = torch.tensor([[7, 8], [9, 10], [11, 12]])
print(tensor1 @ tensor2)
print(torch.matmul(tensor1, tensor2))#faster than @, same as torch.mm()
print()

#make matrix multiplication work between tensor_A and tensor_B by making their inner dimensions match

#one of the ways to do this is with a transpose T
tensor_A = torch.tensor([[1, 2],
                         [3, 4],
                         [5, 6]], dtype=torch.float32)

tensor_B = torch.tensor([[7, 10],
                         [8, 11],
                         [9, 12]], dtype=torch.float32)
#using torch.matmul(tensor_A, tensor_B) will error

#let's use tensor.T to switch the dimensions of tensor_B
print(f"Original shapes: tensor_A = {tensor_A.shape}, tensor_B = {tensor_B.shape}")
print(f"New shapes: tensor_A = {tensor_A.shape} (same as above), tensor_B.T = {tensor_B.T.shape}")
print(f"Multiplying: {tensor_A.shape} * {tensor_B.T.shape} <- inner dimensions match")
print("Output:")
output = torch.matmul(tensor_A, tensor_B.T)
print(output)
print(f"Output shape: {output.shape}")
print()

print("-------------A sample of linear layers---------------")
#Neural networks are full of matrix multiplications and dot products
# Since the linear layer starts with a random weights matrix, let's make it reproducible (more on this later)
torch.manual_seed(42)
# This uses matrix multiplication
linear = torch.nn.Linear(in_features=2, # in_features = matches inner dimension of input
                         out_features=6) # out_features = describes outer value
x = tensor_A
output = linear(x)
print(f"Input shape: {x.shape}\n")
print(f"Output:\n{output}\n\nOutput shape: {output.shape}")