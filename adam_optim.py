import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple linear regression model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# Create a dataset with a linear relationship
x_train = torch.randn(100, 1) * 10
y_train = x_train * 2 + torch.randn(100, 1)

# Initialize the model and optimizer
model = LinearRegression()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
for epoch in range(1000):
    # Forward pass
    y_pred = model(x_train)

    # Compute loss
    loss = nn.functional.mse_loss(y_pred, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print progress
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Evaluate the trained model
x_test = torch.Tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y_test = model(x_test)
print(f"Predictions: {y_test.detach().numpy()}")
