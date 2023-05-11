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
    
    gates = torch.zeros(11, 8, 8, dtype=torch.complex64)
    #generate random gates for the 2-qubit gates and switch them with the corresponding single qubit gates in the Gm list randomly
    for i in range(len(Gm)):
        for j in range(5, Gm[i].size(dim = 0)): #loop over the 2-qubit gates
            for k in range(Gm[i].size(dim = 0)):    #loop over the single qubit gates
                if k < 5:   #if the gate is single qubit
                    num_gates = 0
                    while(num_gates < 6):
                        gates[i][j][k] = torch.rand(2, 2, dtype=torch.complex64)   #generate a random gate
                        num_gates += 1

    #G_final is the matrix multiplication of the gates
    for i in range(len(gates)):
        G_final*=gates[i]

        #G_final = G1@G2@G3@G4@G5@G6@G7@G8@G9@G10@G11@G12@G13@G14@G15@G16@G17@G18

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
