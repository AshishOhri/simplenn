''' Building a basic neural network'''

import numpy as np

# We use the sigmoid function as the activation function
def sigmoid_function(L):
    return 1/(1+np.exp(-L))

# Derivative (slope)=y/x
def deriv(L):
    return L*(1-L)

# START READING FROM HERE

# Take input and output
X=np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])
Y=np.array([[0,0,1,1]]).T

# Seeding
np.random.seed(1)

'''
    we have the input matrix as 4X3 matrix and we create one hidden layer for this example

    input maxtrix X (4*3=12)       Synapse matrix (hidden layer)        Output Matrix (4*1=4)
            O
            O                                     O                           O
            O                                     O                           O
            O                                     O                           O
            O                                                                 O
            O
            O
            O
            O
            O
            O
            O
    Synapse matrix has dimensions as 3*1 so as to get the output matrix on the dot product with the input matrix
'''

#initializing the weights randomly with mean 0
syn0=np.random.random([3,1])

# Loop for repeating the process of forward proppgation, calculating the loss and then updating the weights using back propogation
for i in range(10000):

    #forward propogation
    l0=X                                   # 4*3
    dot_product=np.dot(l0,syn0)            # 4*3 . 3*1 = 4*1
    l1=sigmoid_function(dot_product)       # 4*1

    #Calculating the error
    l1_error=Y-l1                          # 4*1

    #Calculating the derivative
    d=deriv(l1)                            # 4*1

    #getting the delta i.e the value by which the weights must be updated
    l1_delta=l1_error*d                   # 4*1

    #update the weights
    syn0+=np.dot(l0.T,l1_delta)           # 3*4 . 4*1 = 3*1

print("Output after training:")
print(l1)