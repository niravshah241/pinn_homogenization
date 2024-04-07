import torch
import numpy as np

def train_nn(dataloader_pinn, dataloader_bc, dataloader_ic,
             loss_func_pinn, loss_func_bc, loss_func_ic,
             optimiser_pinn, optimiser_bc, optimiser_ic,
             ann_model, *pinn_loss_args):
    ann_model.train()
    # TODO Verify 3 different opmtimiser implementation
    # TODO Only 1 batch data is considered
    for batch, *args in enumerate(dataloader_bc):
        label_pred_bc = ann_model(args[0][0])
        loss_bc = loss_func_bc(label_pred_bc, torch.vstack((args[0][1])).T)
    for batch, *args in enumerate(dataloader_ic):
        label_pred_ic = ann_model(args[0][0])
        loss_ic = loss_func_ic(label_pred_ic, torch.vstack((args[0][1])).T)
    # TODO args with pinn loss
    for batch, (x_scaled, y_scaled, t_scaled, source_label) in enumerate(dataloader_pinn):
        label_pred_pinn = ann_model(x_scaled, y_scaled, t_scaled)
        loss_residual = loss_func_pinn(label_pred_pinn, x_scaled, y_scaled,
                                       source_label, *pinn_loss_args)
    total_loss = loss_bc + loss_ic + loss_residual
    total_loss.backward()
    optimiser_bc.step()
    optimiser_ic.step()
    optimiser_pinn.step()
    optimiser_bc.zero_grad()
    optimiser_ic.zero_grad()
    optimiser_pinn.zero_grad()
    return total_loss, loss_bc, loss_func_ic, loss_residual

def validate_nn(dataloader_pinn, dataloader_bc, dataloader_ic,
                loss_func_pinn, loss_func_bc, loss_func_ic,
                ann_model, *pinn_loss_args):
    ann_model.eval()
    # TODO Only 1 batch of data is considered
    with torch.no_grad():
        for batch, *args in enumerate(dataloader_bc):
            label_pred_bc = ann_model(args[0][0])
            loss_bc = loss_func_bc(label_pred_bc, torch.vstack((args[0][1])).T)
        for batch, *args in enumerate(dataloader_ic):
            label_pred_ic = ann_model(args[0][0])
            loss_ic = loss_func_ic(label_pred_ic, torch.vstack((args[0][1])).T)
        # TODO args with pinn loss
        for batch, (x_scaled, y_scaled, t_scaled, source_label) in enumerate(dataloader_pinn):
            label_pred_pinn = ann_model(x_scaled, y_scaled, t_scaled)
            loss_residual = loss_func_pinn(label_pred_pinn, x_scaled, y_scaled,
                                           source_label, *pinn_loss_args)
        total_loss = loss_bc + loss_ic + loss_residual
    return total_loss, loss_bc, loss_ic, loss_residual
