import torch
import torch.optim as optim
import numpy as np

import scipy.linalg
import sys
import functions

#create a loop that generates circuits with different x values and combination of functions.Gx(x) gates 
#and test the results of the circuits using generate_matrix(x_var) function

# define a function that generates a random matrix with a given value of x
def generate_matrix(x_var):
    Gm = []
    # loop over the x values to generate the corresponding G matrices
    for i in range(x_var.size(dim=0)):
        Gx_i = torch.zeros(11, 8, 8, dtype=torch.complex64)
        Gx_i = functions.Gx(x_var[i].item())
        Gm.append(Gx_i)
    
    gates = []
    #generate random gates for the 2-qubit gates and switch them with the corresponding single qubit gates in the Gm list randomly
    num_gates = 0  #number of gates in the gate list

    while(num_gates < 6):
        for i, j, k, i in torch.randint(0, len(Gm), (1,)), j in torch.randint(5, Gm[i].size(dim = 0), (1,)), k in torch.randint(0,5, (1,)): #loop over the different G matrices with different x values
            gates.append(Gm[i][j][k])
            num_gates+=1
    
    for i in range(len(gates)):
        G_final*=gates[i]

    return G_final

x_var = torch.rand(18, dtype=torch.float32)
print(generate_matrix(x_var))


#generate random x values between 0 and 2pi

# x = torch.rand(10, dtype=torch.float32)
# #generate random circuits: matrix product of functions.Gx(x)(combination of i < 5 and k >=5)
# #and test the results of the circuits
# for i in range(x.size(dim = 0)):
#     functions.Gx_i = functions.Gx((x[i].item()))
#     for j in range(5, functions.Gx_i.size(dim = 0)):
#             for k in range(functions.Gx_i.size(dim = 0)):
#                 if k < 5:

#not single qubit gates after another

#starting with 2-qubit gates(i>5 in G matrix)  + 
# 2 single qubit gates(i<5 in G matrix) afte every 2-qubit gate


#να μην εχουν το 4 στην ιδια θεση(2qubits)

# for vv3 in range(functions.Gx_i.size(dim = 0)):
#     for 2quit_gate in range(functions.Gx_i)


#blocks per 3 and permute them to get the final G matrix
