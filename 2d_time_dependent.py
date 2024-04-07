import torch
import numpy as np
import matplotlib.pyplot as pyplot
from custom_dataset import CustomDataset
from neural_network import ANN_net

device = "cuda:0"
dtype = torch.float32

# True solution
alpha = 3.
beta = 1.2
def true_sol(domain_points, alpha=alpha, beta=beta):
    return (1 + (domain_points[:, 0])**2 + \
        alpha * (domain_points[:, 1])**2 + \
        beta * (domain_points[:, 2])).unsqueeze(1)

# Create collocation points on domain (x, y) and in time (t)
# (x, y, t) order 0 \leq x \leq 1, 0 \leq y \leq 1, 0 \leq t \leq 2
num_points_domain = 1000
domain_points = \
    torch.hstack((torch.rand(num_points_domain, 1),
                  torch.rand(num_points_domain, 1),
                  torch.rand(num_points_domain, 1))).to(device=device).to(dtype=dtype)

# Bottom boundary y = 0
num_points_bottom_boundary = 100
bottom_bc_points = \
    torch.hstack((torch.rand(num_points_bottom_boundary, 1),
                  torch.zeros(num_points_bottom_boundary, 1),
                  torch.rand(num_points_bottom_boundary, 1))).to(device=device).to(dtype=dtype)

# Right boundary x = 1
num_points_right_boundary = 100
right_bc_points = \
    torch.hstack((torch.ones(num_points_right_boundary, 1),
                  torch.rand(num_points_right_boundary, 1),
                  torch.rand(num_points_right_boundary, 1))).to(device=device).to(dtype=dtype)

# Top boundary y = 1
num_points_top_boundary = 100
top_bc_points = \
    torch.hstack((torch.rand(num_points_top_boundary, 1),
                  torch.ones(num_points_top_boundary, 1),
                  torch.rand(num_points_top_boundary, 1))).to(device=device).to(dtype=dtype)

# Left boundary x = 0
num_points_left_boundary = 100
left_bc_points = \
    torch.hstack((torch.zeros(num_points_left_boundary, 1),
                  torch.rand(num_points_left_boundary, 1),
                  torch.rand(num_points_left_boundary, 1))).to(device=device).to(dtype=dtype)

# Initial condition t = 0
num_points_initial_condition = 100
ic_points = \
    torch.hstack((torch.rand(num_points_initial_condition, 1),
                  torch.rand(num_points_initial_condition, 1),
                  torch.zeros(num_points_initial_condition, 1))).to(device=device).to(dtype=dtype)
u_ic_val = true_sol(ic_points)


# Collect all BC points and BC solution value
bc_points = torch.vstack((bottom_bc_points,
                          right_bc_points,
                          top_bc_points,
                          left_bc_points))
u_bc_val = true_sol(bc_points)

points_range = torch.tensor([[0., 0., 0.],
                             [1., 1., 2.]]).to(device=device).to(dtype=dtype)
points_scaling_range = torch.tensor([[-1., -1., -1.],
                                     [1., 1., 1.]]).to(device=device).to(dtype=dtype)


# Solution (u) range
u_val_range = torch.tensor([[-0.2], [6.4]]).to(device=device).to(dtype=dtype)
u_val_scaling_range = torch.tensor([[-1.], [1.]]).to(device=device).to(dtype=dtype)

# Source term
f_true = ((beta - 2 - 2 * alpha) * torch.ones(num_points_domain)).unsqueeze(1).to(device=device).to(dtype=dtype)
f_true_range = (torch.tensor([[-10.], [10.]])).to(device=device).to(dtype=dtype)
f_true_scaling_range = (torch.tensor([[-10.], [10.]])).to(device=device).to(dtype=dtype)

# Create dataset and dataloader
domain_dataset = CustomDataset(domain_points, f_true,
                               points_range, f_true_range,
                               points_scaling_range,
                               f_true_scaling_range)

bc_dataset = CustomDataset(bc_points, u_bc_val,
                           points_range, u_val_range,
                           points_scaling_range,
                           u_val_scaling_range)

ic_dataset = CustomDataset(ic_points, u_ic_val,
                           points_range, u_val_range,
                           points_scaling_range,
                           u_val_scaling_range)

# domain_dataloader = torch.utils.data.DataLoader(domain_dataset, batch_size=num_points_domain, shuffle=True)
domain_dataloader = torch.utils.data.DataLoader(domain_dataset, batch_size=7, shuffle=True)
bc_dataloader = torch.utils.data.DataLoader(bc_dataset, batch_size=bc_points.shape[0], shuffle=True)
ic_dataloader = torch.utils.data.DataLoader(ic_dataset, batch_size=num_points_initial_condition, shuffle=True)

net = ANN_net([3, 50, 50, 1], torch.tanh).to(device=device).to(dtype=dtype)

for batch, *args in enumerate(domain_dataloader):
    u_pred = net(args[0][0])
    print(torch.nn.MSELoss()(u_pred, torch.vstack((args[0][1])).T))
    exit()
