import torch
import numpy as np

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, input_data, output_data, input_range,
                 output_range, input_scaling_range, output_scaling_range):
        self.input_range = input_range
        self.output_range = output_range
        self.input_scaling_range = input_scaling_range
        self.output_scaling_range = output_scaling_range
        self.input_data_scaled = self.scale_input(input_data)
        self.output_data_scaled = self.scale_output(output_data)
    
    def __len__(self):
        return self.input_data_scaled.shape[0]
    
    def __getitem__(self, idx):
        list_in = list()
        list_out = list()
        for i in range(self.input_data_scaled.shape[1]):
            list_in.append(self.input_data_scaled[idx, i])
            list_in[i].requires_grad = True
        for i in range(self.output_data_scaled.shape[1]):
            list_out.append(self.output_data_scaled[idx, i])
            list_out[i].requires_grad = True
        return list_in, list_out
    
    def scale_input(self, input_data):
        input_scaling_range = self.input_scaling_range
        input_range = self.input_range
        input_data_scaled = \
            (input_data - self.input_range[0, :]) / \
            (input_range[1, :] - input_range[0, :]) * \
            (input_scaling_range[1, :] - input_scaling_range[0, :]) + \
            input_scaling_range[0, :]
        return input_data_scaled

    def scale_output(self, output_data):
        output_scaling_range = self.output_scaling_range
        output_range = self.output_range
        output_data_scaled = \
            (output_data - self.output_range[0, :]) / \
            (output_range[1, :] - output_range[0, :]) * \
            (output_scaling_range[1, :] - output_scaling_range[0, :]) + \
            output_scaling_range[0, :]
        return output_data_scaled

    def reverse_scale_input(self, input_data_scaled):
        input_scaling_range = self.input_scaling_range
        input_range = self.input_range
        input_data = \
            (input_data_scaled - self.input_scaling_range[0, :]) / \
            (input_scaling_range[1, :] - input_scaling_range[0, :]) * \
            (input_range[1, :] - input_range[0, :]) + \
            input_range[0, :]
        return input_data

    def reverse_scale_output(self, output_data_scaled):
        output_scaling_range = self.output_scaling_range
        output_range = self.output_range
        output_data = \
            (output_data_scaled - self.output_scaling_range[0, :]) / \
            (output_scaling_range[1, :] - output_scaling_range[0, :]) * \
            (output_range[1, :] - output_range[0, :]) + \
            output_range[0, :]
        return output_data

if __name__ == "__main__":
    # TODO Test examples for each of them
    exit()
