import torch
import numpy as np
import functions
import torch.optim as optim

# Initialize x as a trainable parameter
x = torch.nn.Parameter(torch.rand(18, dtype=torch.float32, requires_grad=True)*2*np.pi)

# Initialize G_final as a trainable parameter
G_final = torch.eye(8, 8, dtype=torch.complex64, requires_grad=True)

# Initialize the optimizer with a learning rate of 0.01
optimizer = optim.Adam([x, G_final], lr=0.1)

# Define the scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Optimize both x and G_final simultaneously for 1000 iterations
for i in range(1000):
    optimizer.zero_grad()
    Gm = []

    # Compute Gm using the current value of x
    for j in range(x.size(dim=0)):
        functions.Gx_i = torch.zeros(11, 8, 8, dtype=torch.complex64)
        functions.Gx_i = functions.Gx((x[j].item()))
        Gm.append(functions.Gx_i)

    # Compute the final G matrix using the current value of Gm
    G1 = torch.tensor(Gm[0][5])
    G2 = torch.tensor(Gm[1][1])
    G3 = torch.tensor(Gm[2][4])
    G4 = torch.tensor(Gm[3][7])
    G5 = torch.tensor(Gm[4][0])
    G6 = torch.tensor(Gm[5][2])
    G7 = torch.tensor(Gm[6][9])
    G8 = torch.tensor(Gm[7][4])
    G9 = torch.tensor(Gm[8][0])
    G10 = torch.tensor(Gm[9][6])
    G11 = torch.tensor(Gm[10][2])
    G12 = torch.tensor(Gm[11][4])
    G13 = torch.tensor(Gm[12][8])
    G14 = torch.tensor(Gm[13][0])
    G15 = torch.tensor(Gm[14][3])
    G16 = torch.tensor(Gm[15][10])
    G17 = torch.tensor(Gm[16][0])
    G18 = torch.tensor(Gm[17][4])

    G_final = G1 @ G2 @ G3 @ G4 @ G5 @ G6 @ G7 @ G8 @ G9 @ G10 @ G11 @ G12 @ G13 @ G14 @ G15 @ G16 @ G17 @ G18

    # Compute the cost function using the current value of G_final
    cost = functions.cost_function(G_final)

    # Compute gradients and update parameters
    cost.backward()
    optimizer.step()

    # Print current cost every 10 iterations
    if i % 10 == 0:
        print(f"Iteration {i}: Cost {cost.item()}")
        
    # Update learning rate every 20 iterations using StepLR scheduler
    if i % 10 == 0:
        scheduler.step()

# Print the optimized x, G_final, and cost function
print("Optimized x:\n", x)
print("Optimized G_final:\n", G_final)
print("Final cost:\n", cost.item())
