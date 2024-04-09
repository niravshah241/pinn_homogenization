import torch
import numpy as np
import matplotlib.pyplot as pyplot
from custom_dataset import CustomDataset
from neural_network import ANN_net
from train_validation import train_nn, validate_nn
import time

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

domain_dataloader = torch.utils.data.DataLoader(domain_dataset, batch_size=num_points_domain, shuffle=True)
bc_dataloader = torch.utils.data.DataLoader(bc_dataset, batch_size=bc_points.shape[0], shuffle=True)
ic_dataloader = torch.utils.data.DataLoader(ic_dataset, batch_size=num_points_initial_condition, shuffle=True)

net = ANN_net([3, 10, 10, 1], torch.tanh).to(device=device).to(dtype=dtype)

'''
for batch, args in enumerate(domain_dataloader):
    print(args)
    print(args[0])
    print(args[0][1])
    exit()
    print(torch.vstack((args[0])).T)
    u_pred = net(args[0])
    print(u_pred)
    print(args[1])
    print(torch.vstack((args[1])).T)
    print(torch.nn.MSELoss()(u_pred, torch.vstack((args[1])).T))
    exit()
'''

def pde_loss(u_pred, input_args, output_args, input_scaling_range, output_scaling_range, input_range, output_range):
    x_scaled = input_args[0]
    y_scaled = input_args[1]
    t_scaled = input_args[2]
    f_true = output_args[0]
    diffusion_coefficient = torch.tensor([0.1]).to(x_scaled.device)
    # TODO Verify that pde_loss backpropagation is not slowed by split of the output columns below,
    # like previously in customdataloder case
    u_x = torch.autograd.grad(u_pred[:, 0].sum(), x_scaled, create_graph=True)[0].to(x_scaled.device)
    u_y = torch.autograd.grad(u_pred[:, 0].sum(), y_scaled, create_graph=True)[0].to(y_scaled.device)
    u_t = torch.autograd.grad(u_pred[:, 0].sum(), t_scaled, create_graph=True)[0].to(t_scaled.device)
    u_xx = torch.autograd.grad(u_x.sum(), x_scaled, create_graph=True)[0].to(x_scaled.device)
    u_yy = torch.autograd.grad(u_y.sum(), y_scaled, create_graph=True)[0].to(y_scaled.device)
    # NOTE in case of error, check loss function and also see if f_true is correctly written in the loss function
    residual = ((output_range[1][0] - output_range[0][0]) * (input_scaling_range[1][2] - input_scaling_range[0][2])) / ((output_scaling_range[1][0] - output_scaling_range[0][0]) * (input_range[1][2] - input_range[0][2])) * u_t[0] - diffusion_coefficient * ((input_scaling_range[1][0] - input_scaling_range[0][0])**2 / (input_range[1][0] - input_range[0][0])**2 * (output_range[1][0] - output_range[0][0]) / (output_scaling_range[1][0] - output_scaling_range[0][0]) * u_xx[0] + (input_scaling_range[1][1] - input_scaling_range[0][1])**2 / (input_range[1][1] - input_range[0][1])**2 * (output_range[1][0] - output_range[0][0])/ (output_scaling_range[1][0] - output_scaling_range[0][0]) * u_yy[0]) - f_true
    return torch.mean(residual**2)

max_epochs = 5000
total_loss_list = list()
residual_loss_list = list()
ic_loss_list = list()
bc_loss_list = list()

optimiser = torch.optim.Adam(net.parameters(), lr=1.e-4)

for epoch in range(max_epochs):
    start_time = time.process_time()
    total_loss, loss_bc, loss_ic, loss_residual = \
        train_nn(domain_dataloader, bc_dataloader,
                 ic_dataloader, pde_loss, optimiser, net)
    # TODO validate_nn by splitting data in test and train set
    total_loss_list.append(total_loss.item())
    ic_loss_list.append(loss_ic.item())
    bc_loss_list.append(loss_bc.item())
    residual_loss_list.append(loss_residual.item())
    if (epoch + 1) % 100 == 0:
        print(f"Epoch: {epoch+1}/{max_epochs}, Loss: {total_loss.item()}, Loss boundary: {loss_bc.item()}, Loss IC: {loss_ic.item()}, Loss residual: {loss_residual.item()}")
    if epoch == 0:
        min_loss = total_loss_list[epoch]
    else:
        if total_loss_list[epoch] <= min_loss:
            min_loss = total_loss_list[epoch]
            torch.save(net.state_dict(), "best_model.pth")
        else:
            pass
    end_time = time.process_time()
    print(f"Epoch: {epoch+1}, Loss: {total_loss.item()}, Epoch time: {end_time - start_time}")
