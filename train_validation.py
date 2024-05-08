import torch
import numpy as np

def train_nn(dataloader_pinn, dataloader_bc, dataloader_ic,
             loss_func_pinn, optimiser, ann_model, weights):
    # TODO Document loss_func_pinn has specific function call.
    # NOTE Same optimiser for bc, ic and pinn loss functions
    # NOTE Assumed MSE loss function for BC and IC
    # TODO Only 1 batch of data is considered
    ann_model.train()
    input_range = dataloader_pinn.dataset.input_range
    input_scaling_range = dataloader_pinn.dataset.input_scaling_range
    output_range = dataloader_pinn.dataset.output_range
    output_scaling_range = dataloader_pinn.dataset.output_scaling_range
    output_range_bc = dataloader_bc.dataset.output_range
    output_scaling_range_bc = dataloader_bc.dataset.output_scaling_range
    output_range_ic = dataloader_ic.dataset.output_range
    output_scaling_range_ic = dataloader_ic.dataset.output_scaling_range
    for batch, args in enumerate(dataloader_bc):
        label_pred_bc = ann_model(args[0])
        loss_bc = \
            (output_range_bc[1][0] - output_range_bc[0][0])**2 / \
            (output_scaling_range_bc[1][0] - output_scaling_range_bc[0][0])**2 * \
            torch.mean((label_pred_bc - torch.vstack((args[1])).T)**2)
    for batch, args in enumerate(dataloader_ic):
        label_pred_ic = ann_model(args[0])
        loss_ic = \
            (output_range_ic[1][0] - output_range_ic[0][0])**2 / \
            (output_scaling_range_ic[1][0] - output_scaling_range_ic[0][0])**2 * \
            torch.mean((label_pred_ic - torch.vstack((args[1])).T)**2)
    for batch, args in enumerate(dataloader_pinn):
        label_pred_pinn = ann_model(args[0])
        loss_residual = loss_func_pinn(label_pred_pinn, args[0], args[1],
                                       input_scaling_range,
                                       output_scaling_range_bc,
                                       input_range, output_range_bc)
        # NOTE Notice bc in output_range_bc and output_scaling_range_bc
        # Because in dataloader_pinn, output will be f_true and not u.
        # However, u is included in dataloader_bc or dataloader_ic.
    total_loss = weights[0] * loss_bc + weights[1] *loss_ic + weights[2] * loss_residual
    total_loss.backward()
    optimiser.step()
    optimiser.zero_grad()
    return total_loss, loss_bc, loss_ic, loss_residual

def validate_nn(dataloader_pinn, dataloader_bc, dataloader_ic,
                loss_func_pinn, ann_model):
    # TODO Only 1 batch of data is considered
    # TODO Verify vector case especially torch.vstack((args[1])).T
    ann_model.eval()
    output_range = dataloader_pinn.dataset.output_range
    output_scaling_range = dataloader_pinn.dataset.output_scaling_range
    with torch.no_grad():
        for batch, args in enumerate(dataloader_bc):
            label_pred_bc = ann_model(args[0])
            loss_bc = \
                (output_range[1][0] - output_range[0][0])**2 / \
                (output_scaling_range[1][0] - output_scaling_range[0][0])**2 * \
                torch.mean((label_pred_bc - torch.vstack((args[1])).T)**2)
        for batch, args in enumerate(dataloader_ic):
            label_pred_ic = ann_model(args[0])
            loss_ic = \
                (output_range[1][0] - output_range[0][0])**2 / \
                (output_scaling_range[1][0] - output_scaling_range[0][0])**2 * \
                torch.mean((label_pred_ic - torch.vstack((args[1])).T)**2)
    for batch, args in enumerate(dataloader_pinn):
        label_pred_pinn = ann_model(args[0])
        loss_residual = loss_func_pinn(label_pred_pinn, args[0], args[1],
                                       input_scaling_range,
                                       output_scaling_range_bc,
                                       input_range, output_range_bc)
        # NOTE Notice bc in output_range_bc and output_scaling_range_bc
        # Because in dataloader_pinn, output will be f_true and not u.
        # However, u is included in dataloader_bc or dataloader_ic.
        total_loss = loss_bc + loss_ic + loss_residual
    return total_loss, loss_bc, loss_ic, loss_residual
