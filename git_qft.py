
from tqdm import tqdm
from matplotlib import pyplot as plt

def cost(x):
    target = -1
    expval = generate_matrix(x)
    return torch.abs(Gx(x) - target) ** 2, expval

x = torch.tensor([[np.pi/4]], requires_grad=True)
opt = torch.optim.Adam([x], lr=0.1)
num_epoch = 100
loss_list = []
expval_list = []

for i in tqdm(range(num_epoch)):
    opt.zero_grad()
    loss, expval = cost(x)
    loss.backward()
    opt.step()
    loss_list.append(loss.item())
    expval_list.append(expval.item())
