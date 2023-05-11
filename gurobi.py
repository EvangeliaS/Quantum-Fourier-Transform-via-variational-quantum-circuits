import random
import numpy as np
import gurobipy as gp
import scipy.linalg
import torch
import test_2

G = torch.zeros(11, 8, 8, dtype=torch.complex64)
for i in range(G.size(dim = 0)):
    G[i]= torch.linalg.matrix_exp(1j*test_2.vv3[i])

def Gx(x):
    Gx = torch.zeros(11, 8, 8, dtype=torch.complex64)
    for i in range(Gx.size(dim = 0)):
        Gx[i] = torch.tensor(scipy.linalg.fractional_matrix_power(G[i], x)) #G to the power of x
    return Gx


# define a function that generates a random matrix with a given value of x
def generate_matrix(x_var):
    Gm = []
    # loop over the x values to generate the corresponding G matrices
    for i in range(x_var.size(dim=0)):
        Gx_i = torch.zeros(11, 8, 8, dtype=torch.complex64)
        Gx_i = Gx(x_var[i].item())
        Gm.append(Gx_i)

    # multiply the 18 G matrices to get the final G matrix
    #G_final = torch.eye(8, dtype=torch.complex64)
    for i in range(0, len(Gm), 18):
        G1 = Gm[i][5]   #get the first 2-qubit gate(of the first x-modified Gx_i(i==0)), 4-3-3
        G2 = Gm[i+1][1] #get the second single qubit gate, 4-2-4
        G3 = Gm[i+2][4] #get the last single qubit gate 4-4-3
        G4 = Gm[i+3][7] #1-1-4
        G5 = Gm[i+4][0] #1-4-4
        G6 = Gm[i+5][2] #4-3-4
        G7 = Gm[i+6][9] #3-4-2
        G8 = Gm[i+7][4] #4-4-3
        G9 = Gm[i+8][0] #1-4-4
        G10 = Gm[i+9][6] #4-1-3
        G11 = Gm[i+10][2] #4-3-4
        G12 = Gm[i+11][4] #4-4-3
        G13 = Gm[i+12][8] #1-2-4
        G14 = Gm[i+13][0] #1-4-4
        G15 = Gm[i+14][3] #4-1-4
        G16 = Gm[i+15][10]#1-4-3
        G17 = Gm[i+16][0] #1-4-4
        G18 = Gm[i+17][4] #4-4-3

        G_final = G1@G2@G3@G4@G5@G6@G7@G8@G9@G10@G11@G12@G13@G14@G15@G16@G17@G18
    return G_final

def generate_x():
    return torch.rand(18, dtype=torch.float32, requires_grad=True)*100* 2 * np.pi

# create a new model
model = gp.Model()

# create a 3x3 matrix with decision variables and random initial values
n = 8
init_matrix = generate_matrix(generate_x())
matrix = model.addVars(n, n, lb=-1, ub=1, name="matrix")

# set the initial values of the decision variables
for i in range(n):
    for j in range(n):
        matrix[i,j].start = init_matrix[i,j]

# set the objective function to minimize the trace of the matrix
obj_expr = gp.quicksum(matrix[i,i] for i in range(n))
model.setObjective(obj_expr, gp.GRB.MINIMIZE)

# add constraints on the matrix entries
for i in range(n):
    for j in range(n):
        if i != j:
            model.addConstr(matrix[i,j] == -matrix[j,i], name=f"constraint_{i}{j}")
        else:
            model.addConstr(matrix[i,j] >= 0, name=f"constraint_{i}{j}")

# solve the optimization problem for the given parameter values
model.optimize()

# extract the optimal matrix
opt_matrix = np.zeros((n,n))
for i in range(n):
    for j in range(n):
        opt_matrix[i,j] = matrix[i,j].x

print("Initial matrix:")
print(init_matrix)
print("Optimal matrix:")
print(opt_matrix)
