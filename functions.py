import torch
import numpy as np
import scipy.linalg
import sys

ss1 = [[0,1],[1,0]]
ss1 = torch.tensor(ss1)
#print(ss1)

#complex matrix with real and imaginary parts
ss2 = [[0,-1j],[1j,0]]
ss2 = torch.tensor(ss2)
#print(ss2)

ss3 = [[1,0],[0,-1]]
ss3 = torch.tensor(ss3)
#print(ss3)

ss4 = torch.eye(2)
#print(ss4)

#fill a tensor 1*8 with 1/(2*sqrt(2)
QF3 = torch.full((1,8),1/(2*torch.sqrt(torch.tensor(2.0))))
#QF3 = torch.tensor(QF3)
#print(QF3)

#fill a tensor 1*8 with 1/(2*sqrt(2), exponential(1j*pi/4)/2*sqrt(2), (1j)/2*sqrt(2), exponential(3j*pi/4)/2*sqrt(2), - 1/(2*sqrt(2), exponential(-3j*pi/4)/2*sqrt(2), -(1j)/2*sqrt(2), exponential(-1j*pi/4)/2*sqrt(2)
QF4 = [[1/(2*torch.sqrt(torch.tensor(2.0))), (torch.exp(1j*torch.tensor(np.pi/4)))/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0))) , torch.exp(3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))),-1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))) , -1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0)))]]
QF4 = torch.tensor(QF4)
#print(QF4)

#fill a tensor 1*8, with 1/(2*sqrt(2), (1j)/2*sqrt(2), - 1/(2*sqrt(2),  -(1j)/2*sqrt(2), 1/(2*sqrt(2), (1j)/2*sqrt(2), - 1/(2*sqrt(2),  -(1j)/2*sqrt(2)
QF5 = [[1/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0))), 1/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0)))]]
QF5 = torch.tensor(QF5)
#print(QF5)

#fill a tensor 1*8, with 1/(2*sqrt(2), exponential(3j*pi/4)/2*sqrt(2), -(1j)/2*sqrt(2), exponential(1j*pi/4)/2*sqrt(2), - 1/(2*sqrt(2),  exponential(-1j*pi/4)/2*sqrt(2), (1j)/2*sqrt(2), exponential(-3j*pi/4)/2*sqrt(2)
QF6 = [[1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))),-1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))) , 1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0)))]]
QF6 = torch.tensor(QF6)
#print(QF6)

#fill a tensor 1*8 with 1/(2*sqr(2) and -1/(2*sqr(2)
QF7 = [[1/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), 1/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), 1/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), 1/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0)))]]
QF7 = torch.tensor(QF7)
#print(QF7)

QF8 = [[1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))) , -1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0)))]]
QF8 = torch.tensor(QF8)
#print(QF8)

#fill a tensor 1*8, with 1/(2*sqrt(2), (-1j)/2*sqrt(2), - 1/(2*sqrt(2),  (1j)/2*sqrt(2), 1/(2*sqrt(2), (-1j)/2*sqrt(2), - 1/(2*sqrt(2),  (1j)/2*sqrt(2)
QF9 = [[1/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0))), 1/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0))), -1/(2*torch.sqrt(torch.tensor(2.0))), 1j/(2*torch.sqrt(torch.tensor(2.0)))]]
QF9 = torch.tensor(QF9)
#print(QF9)


#fill a tensor 1*8 with 1/(2*sqrt(2), exponential(-1j*pi/4)/2*sqrt(2), (-1j)/2*sqrt(2), exponential(-3j*pi/4)/2*sqrt(2), - 1/(2*sqrt(2), exponential(3j*pi/4)/2*sqrt(2), (1j)/2*sqrt(2), exponential(1j*pi/4)/2*sqrt(2)
QF10 = [[1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))), -1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(-3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))),-1/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(3j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0))) , 1j/(2*torch.sqrt(torch.tensor(2.0))), torch.exp(1j*torch.tensor(np.pi/4))/(2*torch.sqrt(torch.tensor(2.0)))]]
QF10 = torch.tensor(QF10)
#print(QF10)

#make tensor with all the above tensors in it
QF = torch.cat((QF3, QF4, QF5, QF6, QF7, QF8, QF9, QF10), 0)

#find conjugate transpose of QF3
B = torch.conj(torch.transpose(QF, 0,1))
B.requires_grad = True

#find kronecker products
c1 = torch.kron(ss1, ss4)
c1 = torch.kron(c1, ss4)

c2 = torch.kron(ss4, ss2)
c2 = torch.kron(c2, ss4)

c3 = torch.kron(ss4, ss3)
c3 = torch.kron(c3, ss4)

c4 = torch.kron(ss4, ss1)
c4 = torch.kron(c4, ss4)

c5 = torch.kron(ss4, ss4)
c5 = torch.kron(c5, ss3)

c6 = torch.kron(ss4, ss3)
c6 = torch.kron(c6, ss3)

c7 = torch.kron(ss4, ss1)
c7 = torch.kron(c7, ss3)

c8 = torch.kron(ss1, ss1)
c8 = torch.kron(c8, ss4)

c9 = torch.kron(ss1, ss2)
c9 = torch.kron(c9, ss4)

c10 = torch.kron(ss3, ss4)
c10 = torch.kron(c10, ss2)

c11 = torch.kron(ss1, ss4)
c11 = torch.kron(c11, ss3)

#c1 - c5 are single qubit gates, c6 - c11 are two qubit gates
vv3 = torch.stack((c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11))
#print("vv3: ", vv3)

#Gi_, j_, k_, x_ := MatrixExp[I x KroneckerProduct[Assi, ssj, ssk]]
G = torch.zeros(11, 8, 8, dtype=torch.complex64)
for i in range(G.size(dim = 0)):
    G[i]= torch.linalg.matrix_exp(1j*vv3[i])

# def check_if_unitary(G):
#     for i in range(G.size(dim = 0)):
#         if(torch.allclose(torch.eye(8, dtype = torch.complex64), G[i]@torch.conj(torch.transpose(G[i], 0,1)))):
#             print("G is unitary")
#         else:
#             #end the program
#             print("G is not unitary")
#             sys.exit()


# # define your cost function
# def cost_function(G_final):
#     cost = 1 - 1/64*((torch.abs(torch.trace(G_final@B)))**2)
#     return cost

# # define a function that generates a random matrix with a given value of x
# def generate_matrix(x_var):
#     Gm = []
#     # loop over the x values to generate the corresponding G matrices
#     for i in range(x_var.size(dim=0)):
#         Gx_i = torch.zeros(11, 8, 8, dtype=torch.complex64)
#         Gx_i = Gx(x_var[i].item())
#         Gm.append(Gx_i)

#     # multiply the 18 G matrices to get the final G matrix/circuit
#     #G_final = torch.eye(8, dtype=torch.complex64)
#     for i in range(0, len(Gm), 18):
#         G1 = Gm[i][5]   #get the first 2-qubit gate(of the first x-modified Gx_i(i==0)), 4-3-3
#         G2 = Gm[i+1][1] #get the second single qubit gate, 4-2-4
#         G3 = Gm[i+2][4] #get the last single qubit gate 4-4-3
#         G4 = Gm[i+3][7] #1-1-4
#         G5 = Gm[i+4][0] #1-4-4
#         G6 = Gm[i+5][2] #4-3-4
#         G7 = Gm[i+6][9] #3-4-2
#         G8 = Gm[i+7][4] #4-4-3
#         G9 = Gm[i+8][0] #1-4-4
#         G10 = Gm[i+9][6] #4-1-3
#         G11 = Gm[i+10][2] #4-3-4
#         G12 = Gm[i+11][4] #4-4-3
#         G13 = Gm[i+12][8] #1-2-4
#         G14 = Gm[i+13][0] #1-4-4
#         G15 = Gm[i+14][3] #4-1-4
#         G16 = Gm[i+15][10]#1-4-3
#         G17 = Gm[i+16][0] #1-4-4
#         G18 = Gm[i+17][4] #4-4-3

#         G_final = G1@G2@G3@G4@G5@G6@G7@G8@G9@G10@G11@G12@G13@G14@G15@G16@G17@G18
#     return G_final

def Gx(x):
    Gx = torch.zeros(11, 8, 8, dtype=torch.complex64)
    for i in range(Gx.size(dim = 0)):
        Gx[i] = torch.tensor(scipy.linalg.fractional_matrix_power(G[i], x)) #G to the power of x
    return Gx


def matrix_exponential_gradient(A):
# In summary, this code computes the gradient of the matrix exponential 
# of the input matrix A by numerically approximating the derivative using finite differences. 
# It perturbs each element of A separately, computes the corresponding matrix exponential, 
# and calculates the difference with the original matrix exponential. 
# This difference is then divided by the perturbation value to obtain an approximation of the derivative.

    # Get the size of the input matrix
    N = A.size(0)

    # Create an identity matrix of shape (N, N) on the same device as A
    eye = torch.eye(N, device=A.device)

    # Compute the matrix exponential of A
    exp_A = torch.matrix_exp(A)

    # Initialize the gradient matrix with zeros
    gradient = torch.zeros_like(A)

    # Iterate over each element of the gradient matrix
    for i in range(N):
        for j in range(N):
            # Perturb the (i, j) element of A by a small value
            Aj = A.clone()
            Aj[i, j] += 1e-8

            # Compute the matrix exponential of the perturbed matrix Aj
            exp_Aj = torch.matrix_exp(Aj)

            # Compute the (i, j) element of the gradient
            gradient[i, j] = torch.sum((exp_Aj - exp_A) / 1e-8)

    # Return the computed gradient matrix
    return gradient

# Example usage
#x 10 random values between 0 and 2pi
x_var = torch.rand(18, dtype=torch.float32)*2*np.pi
print("x is: \n\n", x_var)
Gm = []

# loop over the x values to generate the corresponding G matrices
for i in range(x_var.size(dim=0)):
    Gx_i = torch.zeros(11, 8, 8, dtype=torch.complex64)
    Gx_i = Gx(x_var[i].item())
    Gm.append(Gx_i)

print(len(Gm))
print(Gm[0].size())
for i in range(len(Gm)-1):
    for j in range(Gm[i].size(dim = 0)):
        print(Gm[i][j])
        gradient = matrix_exponential_gradient(Gm[i][j])
        print("grad = ", gradient)
