import torch
import torch.optim as optim
import functions
import numpy as np
import pytorch_imag
#define x tensor: 18 random values between 0 and 2pi
x = torch.rand(18, dtype=torch.float32)*2*np.pi

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

# print the optimized G_final
print("G_final after optimization: \n\n", G_final)
print("Initial cost: \n\n", pytorch_imag.initial_cost)
print("Cost after optimization: \n\n", functions.cost_function(G_final))
print("Difference between initial and final cost: \n\n", pytorch_imag.initial_cost - functions.cost_function(G_final))


#Note: the best result is about 0.2 reduction. We need to find a way to improve this result.