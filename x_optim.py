import torch
import numpy as np
import functions
import torch.optim as optim

#cost function
def cost_function(G_final):
    #cost = torch.zeros(8, 8, dtype=torch.complex64)
    cost = 1 - 1/64*((torch.abs(torch.trace(G_final@functions.B)))**2)
    return cost


def create_Gm(x):
    # Compute Gm using the current value of x
    Gm = []
    for j in range(x.size(dim=0)):
        functions.Gx_i = torch.zeros(11, 8, 8, dtype=torch.complex64)
        functions.Gx_i = functions.Gx((x[j].item()))
        Gm.append(functions.Gx_i)

    # Compute the final G matrix using the current value of Gm
    G1 = Gm[0][5].clone().detach()
    G2 = Gm[1][1].clone().detach()
    G3 = Gm[2][4].clone().detach()
    G4 = Gm[3][7].clone().detach()
    G5 = Gm[4][0].clone().detach()
    G6 = Gm[5][2].clone().detach()
    G7 = Gm[6][9].clone().detach()
    G8 = Gm[7][4].clone().detach()
    G9 = Gm[8][0].clone().detach()
    G10 = Gm[9][6].clone().detach()
    G11 = Gm[10][2].clone().detach()
    G12 = Gm[11][4].clone().detach()
    G13 = Gm[12][8].clone().detach()
    G14 = Gm[13][0].clone().detach()
    G15 = Gm[14][3].clone().detach()
    G16 = Gm[15][10].clone().detach()
    G17 = Gm[16][0].clone().detach()
    G18 = Gm[17][4].clone().detach()

    G_final = G1 @ G2 @ G3 @ G4 @ G5 @ G6 @ G7 @ G8 @ G9 @ G10 @ G11 @ G12 @ G13 @ G14 @ G15 @ G16 @ G17 @ G18

    return G_final

# Initialize x as a trainable parameter
x = torch.nn.Parameter(torch.rand(18, dtype=torch.float32, requires_grad=True)*2*np.pi)

print("x is: \n\n", x)
print("x_shape", x.shape)

# Initialize the optimizer with a learning rate of 0.01
optimizer = optim.Adam([x], lr=0.01)

# Define the scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Optimize both x and G_final simultaneously for 1000 iterations
for i in range(100):
    print("x1 is: \n\n", x)

    # Compute the cost function using the current value of G_final
    #print("G_final is: \n\n", create_Gm(x))   
    cost = cost_function(create_Gm(x))
    print(cost)

    cost.backward()
    print("x grad is: \n\n", x.grad)

    # update parameters via
    optimizer.step()

    # Print current cost every 10 iterations
    if i % 10 == 0:
        print(f"Iteration {i}: Cost {cost.item()}")
        
    # # Update learning rate every 20 iterations using StepLR scheduler
    # if i % 10 == 0:
    #     scheduler.step()

# Print the optimized x, G_final, and cost function
print("Optimized x:\n", x)
print("Optimized G_final:\n", G_final)
print("Final cost:\n", cost.item())
