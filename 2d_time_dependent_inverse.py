import torch
import numpy as np
import matplotlib.pyplot as plt
from custom_dataset import CustomDataset
from neural_network import ANN_net
from train_validation import train_nn, validate_nn
import time

device = "cuda:0"
dtype = torch.float32

c_p = torch.tensor([2.3]).to(device=device).to(dtype=dtype)
k = torch.tensor([5.5]).to(device=device).to(dtype=dtype)

# True solution
def true_solution(domain_points):
    return (torch.sin(torch.pi * domain_points[:, 0]) *
            torch.cos(torch.pi * domain_points[:, 1]) *
            torch.sin(torch.pi * domain_points[:, 2])).unsqueeze(1)

# Create collocation points on domain (x, y) and in time (t)
# (x, y, t) order 0 \leq x \leq 1, 0 \leq y \leq 1, 0 \leq t \leq t_max
t_max = 0.7
num_points_domain = 200
domain_points = \
    torch.hstack((torch.rand(num_points_domain, 1),
                  torch.rand(num_points_domain, 1),
                  torch.rand(num_points_domain, 1) * t_max)
                ).to(device=device).to(dtype=dtype)

# Source term
f_true = (c_p * torch.pi * torch.sin(torch.pi * domain_points[:, 0]) *
          torch.cos(torch.pi * domain_points[:, 1]) *
          torch.cos(torch.pi * domain_points[:, 2]) +
          k * torch.pi**2 * torch.sin(torch.pi * domain_points[:, 0]) *
          torch.cos(torch.pi * domain_points[:, 1]) *
          torch.sin(torch.pi * domain_points[:, 2])
          ).unsqueeze(1).to(device=device).to(dtype=dtype)

# Create data points with known solution
num_points_data = 100
data_points = \
    torch.hstack((torch.rand(num_points_data, 1),
                  torch.rand(num_points_data, 1),
                  torch.rand(num_points_data, 1) * t_max)
                ).to(device=device).to(dtype=dtype)
# Solution data
u_val_data = true_solution(data_points)

# Scaling ranges
# source term range
f_true_range = (torch.tensor([[-10.], [10.]])
                ).to(device=device).to(dtype=dtype)
f_true_scaling_range = (torch.tensor([[-10.], [10.]])
                        ).to(device=device).to(dtype=dtype)
# data point range
points_range = torch.tensor([[0., 0., 0.],
                             [1., 1., t_max]]
                            ).to(device=device).to(dtype=dtype)
points_scaling_range = torch.tensor([[-1., -1., -1.],
                                     [1., 1., 1.]]
                                    ).to(device=device).to(dtype=dtype)
# solution (u) range
u_val_range = torch.tensor([[-1.], [1.]]).to(device=device).to(dtype=dtype)
u_val_scaling_range = torch.tensor([[-1.], [1.]]).to(device=device).to(dtype=dtype)

# Create dataset and dataloader
domain_dataset = \
    CustomDataset(domain_points, f_true, points_range, f_true_range,
                  points_scaling_range, f_true_scaling_range)
data_dataset = \
    CustomDataset(data_points, u_val_data, points_range, u_val_range,
                  points_scaling_range, u_val_scaling_range)

domain_dataloader = torch.utils.data.DataLoader(domain_dataset, batch_size=num_points_domain, shuffle=False)
data_dataloader = torch.utils.data.DataLoader(data_dataset, batch_size=num_points_data, shuffle=False)

# TODO ANN, Training, Comparison with true solution (use below plotting routine)










# Plotting and visualization
num_test_points_x = 100
num_test_points_y = 100
t_step_test = 0.5 # Test time step

x_test = torch.linspace(0., 1., num_test_points_x
                        ).to(device=device).to(dtype=dtype)
y_test = torch.linspace(0., 1., num_test_points_y
                        ).to(device=device).to(dtype=dtype)

X_test, Y_test = torch.meshgrid(x_test, y_test, indexing="ij")
X_test_reshaped, Y_test_reshaped = X_test.reshape(-1, 1), Y_test.reshape(-1, 1)
T_test = torch.ones(X_test_reshaped.shape[0], 1
                    ).to(device=device).to(dtype=dtype) * t_step_test
u_true = true_solution(torch.hstack((X_test_reshaped,
                                     Y_test_reshaped, T_test)))

plt.figure(figsize=[8, 8])
true_colorbar = \
    plt.contourf(X_test.cpu(), Y_test.cpu(),
                 u_true.reshape(num_test_points_x,
                                num_test_points_y).cpu(),
                 levels=50, cmap='viridis')
plt.colorbar(true_colorbar)
plt.title("True")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("true_solution_inverse.png")
plt.show()
