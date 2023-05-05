import torch
import torch.optim as optim
import functions
import numpy as np

# define your cost function
def cost_function(G_final):
    cost = 1 - 1/64*((torch.abs(torch.trace(G_final@functions.B)))**2)
    return cost

# define a function that generates a random matrix with a given value of x
def generate_matrix(x):
    Gm = []
    #for multiple real x values test the results of functions.Gx(x) and save them in functions.Gx tensor in dim = 0 
    for i in range(x.size(dim = 0)):
        functions.Gx_i = torch.zeros(11, 8, 8, dtype=torch.complex64, requires_grad=True)
        functions.Gx_i = functions.Gx((x[i].item()))
        Gm.append(functions.Gx_i) #append in Gm list

    #generate the same circuit with the same gates but using only the first x value
    G1 = Gm[0][5]   #get the first 2-qubit gate, 4-3-3
    G2 = Gm[0][1] #get the second single qubit gate, 4-2-4
    G3 = Gm[0][4] #get the last single qubit gate 4-4-3
    G4 = Gm[0][7] #1-1-4
    G5 = Gm[0][0] #1-4-4
    G6 = Gm[0][2] #4-3-4
    G7 = Gm[0][9] #3-4-2
    G8 = Gm[0][4] #4-4-3
    G9 = Gm[0][0] #1-4-4
    G10 = Gm[0][6] #4-1-3
    G11 = Gm[0][2] #4-3-4
    G12 = Gm[0][4] #4-4-3
    G13 = Gm[0][8] #1-2-4
    G14 = Gm[0][0] #1-4-4
    G15 = Gm[0][3] #4-1-4
    G16 = Gm[0][10]#1-4-3
    G17 = Gm[0][0] #1-4-4
    G18 = Gm[0][4] #4-4-3

    #multiply the 18 G matrices to get the final G matrix
    G_final = G1@G2@G3@G4@G5@G6@G7@G8@G9@G10@G11@G12@G13@G14@G15@G16@G17@G18
    return G_final

# define a function that generates a random value of x between 0 and 2pi
def generate_x():
    #x 10 random values between 0 and 2pi
    x = torch.rand(18, dtype=torch.float32, requires_grad=True)*2*np.pi
    #print("x is: \n\n", x)
    return x

# # define a function that generates the matrix with the optimal value of x
def generate_optimal_matrix():
    # initialize the optimization process
    optimal_matrix = None
    optimal_cost = np.inf
    for i in range(100): # try 100 different random values of x
        x = generate_x()
        G = generate_matrix(x)
        # convert G to a PyTorch tensor
        #G = torch.tensor(G, dtype=torch.complex64, requires_grad=True)
        # use the Adam optimizer to minimize the cost function
        optimizer = optim.Adam([G], lr=0.001)
        for j in range(100): # run the optimizer for 100 iterations
            optimizer.zero_grad()
            cost = cost_function(G)
            cost.backward()
            optimizer.step()
            print("Cost is: ", cost.item())
            print("x is: \n\n", x)
        if cost.item() < optimal_cost:
            optimal_cost = cost.item()
            optimal_matrix = G.detach().numpy()
    print("The optimal cost is: ", optimal_cost)
    return optimal_matrix


# # define a function that generates the matrix with the optimal value of x
# def generate_optimal_matrix():
#     # initialize the optimization process
#     optimal_matrix = None
#     optimal_cost = np.inf
#     for i in range(100): # try 100 different random values of x
#         x = generate_x()
#         G = generate_matrix(x)
#         #G = torch.tensor(G, dtype=torch.complex64)
#         x_var = torch.tensor(x, dtype=torch.float, requires_grad=True)
#         print("x is: \n\n", x_var)
#         optimizer = optim.Adam([x_var], lr=0.001)
#         for j in range(100): # run the optimizer for 100 iterations
#             G = generate_matrix(x_var)
#             cost = cost_function(G)
#             optimizer.zero_grad()
#             cost.backward()
#             optimizer.step()
#             print("Cost is: ", cost.item())
#         if cost.item() < optimal_cost:
#             optimal_cost = cost.item()
#             optimal_matrix = G.detach().numpy()
#     return optimal_matrix


# generate the optimal matrix
optimal_matrix = generate_optimal_matrix()

