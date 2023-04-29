import torch
import numpy as np
import functions
import torch.optim as optim

#x 10 random values between 0 and 2pi
x = torch.rand(18, dtype=torch.float32)*2*np.pi
print("x is: \n\n", x)
Gm = []

#for multiple real x values test the results of functions.Gx(x) and save them in functions.Gx tensor in dim = 0 
for i in range(x.size(dim = 0)):
    functions.Gx_i = torch.zeros(11, 8, 8, dtype=torch.complex64)
    #print("xi is: \n\n", x[i].item())
    functions.Gx_i = functions.Gx((x[i].item()))
    Gm.append(functions.Gx_i) #append in Gm list

#create a specific circuit with specific gates and test the results
for i in range(0, len(Gm)-1, 18):
    G1 = Gm[i][5]   #get the first 2-qubit gate, 4-3-3
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

#multiply the 18 G matrices to get the final G matrix
G_final = G1@G2@G3@G4@G5@G6@G7@G8@G9@G10@G11@G12@G13@G14@G15@G16@G17@G18

print("G_final is: \n\n", G_final)

initial_cost = functions.cost_function(G_final)
print("Initial cost: \n\n", initial_cost)

#_________________________ADAM optimizer________________________
#define G tensor
G_final = torch.eye(8,8 , dtype=torch.complex64, requires_grad=True)
optimizer = optim.Adam([G_final], lr=0.001)

# optimize the cost function
for i in range(100):
    optimizer.zero_grad()
    cost = functions.cost_function(G_final)
    cost.backward()
    optimizer.step()
    if i % 10 == 0:
        print(f"Iteration {i}: Cost {cost.item()}")

#perform learning rate scheduling
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=0.0001, gamma=0.1)
for i in range(100):
    optimizer.zero_grad()
    cost = functions.cost_function(G_final)
    cost.backward()
    optimizer.step()
    scheduler.step()
    if i % 10 == 0:
        print(f"Iteration {i}: Cost {cost.item()}")

# print the optimized G_final
print("G_final after optimization: \n\n", G_final)
print("Initial cost: \n\n", initial_cost)
print("Cost after optimization: \n\n", functions.cost_function(G_final))
print("Difference between initial and final cost: \n\n", initial_cost - functions.cost_function(G_final))


#Note: the best result is about 0.2 reduction. We need to find a way to improve this result.
#very improved results using the learning rate scheduling
#______________________________________________________________________________
