import torch
import numpy as np
import matplotlib.pyplot as plt
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
    return (1. + (domain_points[:, 0])**2 + \
        alpha * (domain_points[:, 1])**2 + \
        beta * (domain_points[:, 2])).unsqueeze(1)

# Create collocation points on domain (x, y) and in time (t)
# (x, y, t) order 0 \leq x \leq 1, 0 \leq y \leq 1, 0 \leq t \leq t_max
t_max = 0.2
num_points_domain = 1000
domain_points = \
    torch.hstack((torch.rand(num_points_domain, 1),
                  torch.rand(num_points_domain, 1),
                  torch.rand(num_points_domain, 1) * t_max)).to(device=device).to(dtype=dtype)

# Bottom boundary y = 0
num_points_bottom_boundary = 100
bottom_bc_points = \
    torch.hstack((torch.rand(num_points_bottom_boundary, 1),
                  torch.zeros(num_points_bottom_boundary, 1),
                  torch.rand(num_points_bottom_boundary, 1) * t_max)).to(device=device).to(dtype=dtype)

# Right boundary x = 1
num_points_right_boundary = 100
right_bc_points = \
    torch.hstack((torch.ones(num_points_right_boundary, 1),
                  torch.rand(num_points_right_boundary, 1),
                  torch.rand(num_points_right_boundary, 1) * t_max)).to(device=device).to(dtype=dtype)

# Top boundary y = 1
num_points_top_boundary = 100
top_bc_points = \
    torch.hstack((torch.rand(num_points_top_boundary, 1),
                  torch.ones(num_points_top_boundary, 1),
                  torch.rand(num_points_top_boundary, 1) * t_max)).to(device=device).to(dtype=dtype)

# Left boundary x = 0
num_points_left_boundary = 100
left_bc_points = \
    torch.hstack((torch.zeros(num_points_left_boundary, 1),
                  torch.rand(num_points_left_boundary, 1),
                  torch.rand(num_points_left_boundary, 1) * t_max)).to(device=device).to(dtype=dtype)

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
                             [1., 1., t_max]]).to(device=device).to(dtype=dtype)
points_scaling_range = torch.tensor([[-1., -1., -1.],
                                     [1., 1., 1.]]).to(device=device).to(dtype=dtype)


# Solution (u) range
u_val_range = torch.tensor([[1.], [5.3]]).to(device=device).to(dtype=dtype)
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

domain_dataloader = torch.utils.data.DataLoader(domain_dataset, batch_size=num_points_domain, shuffle=False)
bc_dataloader = torch.utils.data.DataLoader(bc_dataset, batch_size=bc_points.shape[0], shuffle=False)
ic_dataloader = torch.utils.data.DataLoader(ic_dataset, batch_size=num_points_initial_condition, shuffle=False)

net = ANN_net([3, 20, 20, 1], torch.tanh).to(device=device).to(dtype=dtype)

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
    # TODO tested only when u_pred is scalar
    u_x = torch.autograd.grad(u_pred[:, 0].sum(), x_scaled, create_graph=True)[0].to(x_scaled.device)
    u_y = torch.autograd.grad(u_pred[:, 0].sum(), y_scaled, create_graph=True)[0].to(y_scaled.device)
    u_t = torch.autograd.grad(u_pred[:, 0].sum(), t_scaled, create_graph=True)[0].to(t_scaled.device)
    u_xx = torch.autograd.grad(u_x.sum(), x_scaled, create_graph=True)[0].to(x_scaled.device)
    u_yy = torch.autograd.grad(u_y.sum(), y_scaled, create_graph=True)[0].to(y_scaled.device)
    # NOTE in case of error, check loss function and also see if f_true is correctly written in the loss function
    residual = ((output_range[1][0] - output_range[0][0]) * (input_scaling_range[1][2] - input_scaling_range[0][2])) / ((output_scaling_range[1][0] - output_scaling_range[0][0]) * (input_range[1][2] - input_range[0][2])) * u_t[0] - diffusion_coefficient * ((input_scaling_range[1][0] - input_scaling_range[0][0])**2 / (input_range[1][0] - input_range[0][0])**2 * (output_range[1][0] - output_range[0][0]) / (output_scaling_range[1][0] - output_scaling_range[0][0]) * u_xx[0] + (input_scaling_range[1][1] - input_scaling_range[0][1])**2 / (input_range[1][1] - input_range[0][1])**2 * (output_range[1][0] - output_range[0][0])/ (output_scaling_range[1][0] - output_scaling_range[0][0]) * u_yy[0]) - f_true
    return torch.mean(residual**2)

max_epochs = 30000
total_loss_list = list()
residual_loss_list = list()
ic_loss_list = list()
bc_loss_list = list()

optimiser = torch.optim.Adam(net.parameters(), lr=1.e-3)
weights = torch.tensor([1., 1., 0.01]).to(device=device).to(dtype=dtype)

for epoch in range(max_epochs):
    start_time = time.process_time()
    total_loss, loss_bc, loss_ic, loss_residual = \
        train_nn(domain_dataloader, bc_dataloader,
                 ic_dataloader, pde_loss, optimiser, net, weights)
    # TODO validate_nn by splitting data in test and train set
    total_loss_list.append(total_loss.item())
    ic_loss_list.append(loss_ic.item())
    bc_loss_list.append(loss_bc.item())
    residual_loss_list.append(loss_residual.item())
    if epoch == 0:
        min_loss = total_loss_list[epoch]
    else:
        if total_loss_list[epoch] <= min_loss:
            min_loss = total_loss_list[epoch]
            torch.save(net.state_dict(), "best_model.pth")
        else:
            pass
    end_time = time.process_time()
    if (epoch + 1) % 10 == 0:
        print(f"Epoch: {epoch+1}/{max_epochs}, Loss: {total_loss.item()}, Loss boundary: {loss_bc.item()}, Loss IC: {loss_ic.item()}, Loss residual: {loss_residual.item()}, Epoch time: {end_time - start_time}")

num_test_points_x = 25
num_test_points_y = 25
t_step_test = t_max / 2 # Test time step

x_test = torch.linspace(0., 1., num_test_points_x).to(device=device)
y_test = torch.linspace(0., 1., num_test_points_y).to(device=device)

X_test, Y_test = torch.meshgrid(x_test, y_test)
X_test_reshaped, Y_test_reshaped = X_test.reshape(-1, 1), Y_test.reshape(-1, 1)
T_test = torch.ones(X_test_reshaped.shape[0], 1).to(device=device) * t_step_test

input_scaled = bc_dataloader.dataset.scale_input(torch.hstack((X_test_reshaped, Y_test_reshaped, T_test)))
print(input_scaled.shape)
X_test_scaled, Y_test_scaled, T_test_scaled = \
    input_scaled[:, 0], input_scaled[:, 1], input_scaled[:, 2]

with torch.no_grad():
    u_test = bc_dataloader.dataset.reverse_scale_output(net([X_test_scaled, Y_test_scaled, T_test_scaled]))

u_true = true_sol(torch.hstack((X_test_reshaped, Y_test_reshaped, T_test)))

print(f"ANN pred: {u_test.T}")
print(f"True: {u_true.T}")
print(f"Error: {torch.max(abs(u_test - u_true))}")

plt.figure(figsize=[8, 8])
ann_colorbar = \
    plt.contourf(X_test.cpu(), Y_test.cpu(),
                 u_test.reshape(num_test_points_x, num_test_points_y).cpu(),
                 levels=50, cmap='viridis')
plt.colorbar(ann_colorbar)
plt.title("ANN Prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("ann_prediction.png")

plt.figure(figsize=[8, 8])
true_colorbar = \
    plt.contourf(X_test.cpu(), Y_test.cpu(),
                 u_true.reshape(num_test_points_x, num_test_points_y).cpu(),
                 levels=50, cmap='viridis')
plt.colorbar(true_colorbar)
plt.title("True")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("true_solution.png")

plt.figure(figsize=[8, 8])
true_colorbar = \
    plt.contourf(X_test.cpu(), Y_test.cpu(),
                 (abs(u_test - u_true)).reshape(num_test_points_x, num_test_points_y).cpu(),
                 levels=50, cmap='viridis')
plt.colorbar(true_colorbar)
plt.title("Absolute error")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("absolute_error.png")

plt.show()
