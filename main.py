import torch as t

#print(t.__version__)

# pytorch tensors are created through torch.tesor -> https://pytorch.org/docs/stable/tensors.html

# scalars
#scalar = t.tensor(7)  # created a scalar tensor
#print(scalar)
#print(scalar.ndim) # outputs the dimensions of the tensor (n-1 dimensions output)
#print(scalar.shape()) breaks as cant get shape of scalar
#scalar.item() # converts the scalar back into primitive data type

# vector
#vector = t.tensor([7, 7]) # 2 dimensional item
#print(vector.ndim) # outputs 1
#print(vector.shape()) breaks as the scalar cannot be represented as a shape

# MATRIX
#MATRIX = t.tensor([[1, 4],
                  #[2, 5]])
#print(MATRIX[0]) # outputs the first list  (torch.tensor([1, 4]))

# just generally the first horizontal list

#print(MATRIX.shape) # outputs the size of the matrix (e.g. [2, 2] represents a 2x2 matrix)
#print(MATRIX.ndim) # outputs 2
# CANNOT CONVERT MATRIX BACK INTO PRIMITIVE

#mytensor = t.tensor([[[8, 2, 3, 7], 
                     # [2, 3, 4, 6], 
                     # [3, 4, 5, 4]], 
                     #[[8, 2, 3, 7],    # second matrix in the tensor, changes the shape[0] to 2, not 1
                     # [2, 3, 4, 6], 
                     # [3, 4, 5, 4]]])
#print(mytensor[0])
#print(mytensor.ndim)
#print(mytensor.shape) #outputs the size of the tensor, e.g. [1, 3, 3] represents one matrix, size of 3rows x 4columns



# RANDOM TENSORS

# why random tensors?
# random tensors are important as the way many neural networks learn is from heuristics, and update these heuristics to fit better

# creating a random tensor of size (3, 4)
#random_tensor = t.rand(3, 4)

# torch random tensors: https://pytorch.org/docs/stable/generated/torch.rand.html
#print(random_tensor)

#creating a random tensor with similar shape to an image tensor
#random_image_size_tensor = t.rand(size=(224,224,3)) # height, width, colour channels (red green blue)
#print(random_image_size_tensor.shape, random_image_size_tensor.ndim)


# Zeros and ones

#creating tensors of ones and zeros (useful for masks -> getting a model to ignore some values by multiplication)
#zeros = t.zeros(size=(3, 4))
#print(zeros) # outputs a tensor filled with 0

#ones = t.ones(size=(3, 4))
#print(ones) # outputs a tensor filled with 1

#print(ones.dtype) # outputs the data type of all values in the tensor


# creating a range of tensors and tensors-like (1d tensor creation)

 # creating tensors:
  # one_to_five = t.arange(start = 1, end = 6, step = 2) (end is exclusive here)
 # creating tensors-like - replicating another tensor without explicitly defining the shape
  # ten_zeros = t.zeros_like(input = one_to_five()) (creates an equally sized tensor full of 0)


# tensor datatypes -> one of the 3 big errors with pytorch and deep learning:
    # tensors not right datatype
    # tensors not right shape
    # tensors not on right device

float_32_tensor = t.tensor([[3.0, 6.0, 7.0], [2.0, 7.0, 9.0, 1.0, 2.0]], 
                           dtype = None, # what the data type of each item is (e.g. float32, float16, boolean)
                           device = None, # default of "cpu", can be "gpu" or "cuda" -> what device the tensor is on
                           requires_grad = False) # if we want the tensor to keep track of the gradients within numerical calculations

print(float_32_tensor) # default type is always torch.float -> each datapoint is stored using 32 bits


print(float_32_tensor * float_32_tensor) # each item is multiplied by its counterpart?
