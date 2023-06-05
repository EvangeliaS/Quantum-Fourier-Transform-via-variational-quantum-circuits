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


# define a function that generates a random matrix with a given value of x
def generate_matrix(x_var):
    Gm = []
    # loop over the x values to generate the corresponding G matrices
    for i in range(x_var.size(dim=0)):
        Gx_i = torch.zeros(11, 8, 8, dtype=torch.complex64)
        Gx_i = Gx(x_var[i].item())
        Gm.append(Gx_i)

    # multiply the 18 G matrices to get the final G matrix/circuit
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

def create_circuit(x_var, parameters_num):
    Gm = []
    # loop over the x values to generate the corresponding G matrices
    for i in range(x_var.size(dim=0)):
        Gx_i = torch.zeros(11, 8, 8, dtype=torch.complex64)
        Gx_i = Gx(x_var[i].item())
        Gm.append(Gx_i)

    # multiply the 18 G matrices to get the final G matrix/circuit
    i = 0

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

    #Initial gate indices
    gate_indices = [5, 1, 4, 7, 0, 2, 9, 4, 0, 6, 2, 4, 8, 0, 3, 10, 0, 4]  # Example gate indices
    # Additional gates
    G_additional = torch.eye(8, dtype=torch.complex64)  # identity matrix

    # Multiply additional gates based on the number of parameters
    for i in range(parameters_num - 18):
        gate_idx = gate_indices[i % 18]  # Cycle through the gate_indices list
        G_additional = G_additional @ Gm[i+18][gate_idx]
        G_final = G_final @ G_additional # Multiply G_final with G_additional

    return G_final

def Gx(x):
    Gx = torch.zeros(11, 8, 8, dtype=torch.complex64)
    for i in range(Gx.size(dim = 0)):
        Gx[i] = torch.tensor(scipy.linalg.fractional_matrix_power(G[i], x)) #G to the power of x
    return Gx

#find conjugate transpose of QF3
B = torch.conj(torch.transpose(QF, 0,1))
B.requires_grad = True

# Define your cost function
def cost_function(x_var):
    G_final = create_circuit(x_var, len(x_var))
    cost = 1 - 1/64 * ((torch.abs(torch.trace(G_final @ B)))**2)
    return cost

def learning_rate_step_scheduler(learning_rate, step_size):
    return learning_rate * step_size

#the function performs gradient descent of cost to find the optimal x values
#x_var is the initial x values, gamma is the learning rate, delta is the perturbation value
def optimize_parameters(x_var, gamma, delta):   
    print("x initial is: \n\n", x_var)         
    print("cost initial = ", cost_function(x_var))
    x_new = x_var.clone()

    for i in range(len(x_var)):
        x_var_sum = x_var.clone() #create a copy of the x_var tensor
        x_var_sum[i] = x_var[i] + delta
        #print("x sum is: \n\n", x_var_sum)
        cost_sum = cost_function(x_var_sum)
        #print("cost sum = ", cost_sum)

        x_var_diff = x_var.clone()
        x_var_diff[i] = x_var[i] - delta
        #print("x diff is: \n\n", x_var_diff)
        cost_diff = cost_function(x_var_diff)
        #print("cost diff = ", cost_diff)
        x_new[i] = x_var[i] - gamma * ((cost_sum - cost_diff) / (2* delta))

    #print("x new is: \n\n", x_new)
    return x_new, cost_function(x_new)


#create a function that calls the optimize_parameters function until the cost stops changing more than a certain value(epsilon)
def gradient_descent_cost_optimizer(x_var, learning_rate, delta, epsilon, threshold, step_size):
    iterations = 0
    x_init, cost_init = optimize_parameters(x_var, learning_rate, delta) #get the initial cost after the first optimization
    x_old = x_init.clone()
    cost_old = cost_init.clone()
    while True:
        x_new, cost_new = optimize_parameters(x_old, learning_rate, delta)
        print("new cost = ", cost_new)
        if torch.abs(cost_new - cost_old) < epsilon:
            break
        else:
            if(torch.abs(cost_new - cost_old) < threshold and iterations%20 == 0  and iterations != 0):
                learning_rate = learning_rate_step_scheduler(learning_rate, step_size)
                print("ITERATION = ", iterations, "    LEARNING RATE = ", learning_rate, "\n")
            x_old = x_new.clone()
            cost_old = cost_new.clone()
            iterations += 1
    return x_new, cost_new, iterations

# Perform optimization on a single data point -> this will be used in the stochastic gradient descent
def optimize_stochastic_parameters(x_var, learning_rate, delta, data_point):  # Perform optimization on a single data point
    x_var_sum = x_var.clone()
    x_var_sum[data_point] = x_var[data_point] + delta
    cost_sum = cost_function(x_var_sum)

    x_var_diff = x_var.clone()
    x_var_diff[data_point] = x_var[data_point] - delta
    cost_diff = cost_function(x_var_diff)

    x_new = x_var.clone()
    x_new[data_point] = x_var[data_point] - learning_rate* ((cost_sum - cost_diff) / (2 * delta))

    return x_new, cost_function(x_new)


def stochastic_gradient_descent(x_var, learning_rate, delta, epsilon, threshold, step_size, scheduler, num_epochs):
    iterations = 0
    num_data_points = len(x_var)
    x_init, cost_init = optimize_stochastic_parameters(x_var, learning_rate, delta, np.random.randint(num_data_points)) # Get the initial cost after the first optimization
    x_old = x_init.clone()
    cost_old = cost_init.clone()

    while True:
        # Randomly select a data point
        data_point = np.random.randint(num_data_points)
        x_new, cost_new = optimize_stochastic_parameters(x_old, learning_rate, delta, data_point)
        print("x new =", x_new)
        print("new cost =", cost_new)

        print("cost difference = ", torch.abs(cost_new - cost_old))
        if torch.abs(cost_new - cost_old) < epsilon and iterations > num_epochs:
            break
        else:
            if torch.abs(cost_new - cost_old) < threshold and iterations % 10 == 0 and iterations != 0:
                print("Scheduler called", scheduler)
                learning_rate = scheduler(learning_rate, step_size)
                print("ITERATION =", iterations, " LEARNING RATE =", learning_rate, "\n")

            x_old = x_new.clone()
            cost_old = cost_new.clone()
            iterations += 1

    return x_new, cost_new, iterations

##############################################################################################################

#test the functions
import matplotlib.pyplot as plt
import numpy as np
import time

start = time.time()
x_var = torch.rand(22, dtype=torch.float32)*2*np.pi
print("    X INITIAL is: \n\n", x_var) 
print("cost initial = ", cost_function(x_var))

#x, cost , iters = stochastic_gradient_descent(x_var, 0.5, 0.05, 1e-7, 0.001, 0.1, exponential_decay_scheduler, 50 )
x, cost, iters = stochastic_gradient_descent(x_var, 0.2, 0.05, 1e-7, 0.0001, 0.1, learning_rate_step_scheduler, 100)
print("    X FINAL is: \n\n", x)
print("iterations = ", iters, "FINAL COST: ", cost, "\n\n")
end = time.time()
print("time taken = ", end - start, "\n\n")


# for i in range(3):
#     #count time for each iteration
#     start = time.time()
#     x_var = torch.rand(18, dtype=torch.float32)*2*np.pi
#     x, cost , iters = stochastic_gradient_descent(x_var, 0.5, 0.05, 0.0000001, 0.01, 0.1)
#     print("    X INITIAL is: \n\n", x_var) 
#     print("cost initial = ", cost_function(x_var))
#     print("    X FINAL is: \n\n", x)
#     print("iterations = ", iters, "FINAL COST: ", cost, "\n\n")
#     end = time.time()
#     print("time taken = ", end - start, "\n\n")

# #    # Plot each x_var value against the cost
# #     x_var_detached = x_var.detach().numpy()
# #     plt.plot(x_var_detached, cost.detach().numpy(), 'o')
# #     plt.xlabel('x_var')
# #     plt.ylabel('cost')
# #     plt.show()