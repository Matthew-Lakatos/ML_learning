import torch as t

print(t.__version__)

# pytorch tensors are created through torch.tesor -> https://pytorch.org/docs/stable/tensors.html

# scalars
scalar = t.tensor(7)  # created a scalar tensor
print(scalar)
print(scalar.ndim) # outputs the dimensions of the tensor (n-1 dimensions output)
#print(scalar.shape()) breaks as cant get shape of scalar
scalar.item() # converts the scalar back into primitive data type

# vector
vector = t.tensor([7, 7]) # 2 dimensional item
print(vector.ndim) # outputs 1
#print(vector.shape()) breaks as the scalar cannot be represented as a shape

# MATRIX
MATRIX = t.tensor([[1, 4],
                  [2, 5]])
print(MATRIX[0]) # outputs the first list  (torch.tensor([1, 4]))

# just generally the first horizontal list

print(MATRIX.shape) # outputs the size of the matrix (e.g. [2, 2] represents a 2x2 matrix)
print(MATRIX.ndim) # outputs 2
# CANNOT CONVERT MATRIX BACK INTO PRIMITIVE

mytensor = t.tensor([[[8, 2, 3, 7], 
                      [2, 3, 4, 6], 
                      [3, 4, 5, 4]], 
                     [[8, 2, 3, 7],    # second matrix in the tensor, changes the shape[0] to 2, not 1
                      [2, 3, 4, 6], 
                      [3, 4, 5, 4]]])
print(mytensor[0])
print(mytensor.ndim)
print(mytensor.shape) #outputs the size of the tensor, e.g. [1, 3, 3] represents one matrix, size of 3rows x 4columns
