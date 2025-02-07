from utils import visualize_results_grid, calculate_absolute_difference, calculate_accuracy, create_three_channel_tensor, load_grids_from_csv, merge_three_channels
from uNet512 import UNet

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import matplotlib.colors as mcolors

from sklearn.metrics import mean_squared_error


# Caricamento del modello salvato
model_path = r"model_UNet\best_model\FINAL.pth"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)

# Scegli la funzione di perdita: 'mse', 'smooth_l1'
loss_function_choice = 'mse'

if loss_function_choice == 'mse':
    criterion = nn.MSELoss()
elif loss_function_choice == 'smooth_l1':
    criterion = nn.SmoothL1Loss()
	
	
optimizer = optim.Adam(model.parameters(), lr=0.001)
 
model.load_state_dict(torch.load(model_path))
 
# Caricamento dati dai file CSV
file_path_input = 'griglie_input.csv'
frames, grid_data_input = load_grids_from_csv(file_path_input)
 
file_path_output = 'griglie_outputPOLY.csv'
frames, grid_data_output = load_grids_from_csv(file_path_output)
 
grid_tensors_input = [torch.tensor(grid.reshape(70, 70), dtype=torch.float32) for grid in grid_data_input]
grid_tensors_output = [torch.tensor(grid.reshape(70, 70), dtype=torch.float32) for grid in grid_data_output]
# Trasforma i dati di input e output
grid_tensors_input = [
    create_three_channel_tensor(grid) for grid in grid_tensors_input
]
grid_tensors_output = [
    create_three_channel_tensor(grid) for grid in grid_tensors_output
]
# Dividi i dati in training e validation set
x_train, x_test, y_train, y_test = train_test_split(
    grid_tensors_input, grid_tensors_output, test_size=0.2, random_state=12#ex seed 42
)
 
x_test = torch.stack(x_test)
y_test = torch.stack(y_test)
test_dataset = TensorDataset(x_test, y_test)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
 
model.eval()
val_loss = 0
output_eval = []
target_eval =[]
with torch.no_grad():
    for data in test_loader:
        inputs, targets = data[0].to(device), data[1].to(device)
        outputs= model(inputs)
        #INIZIO PARTE NUOVA
        output_eval.extend(outputs.squeeze(1).cpu().numpy())
        target_eval.extend(targets.squeeze(1).cpu().numpy())
        out = merge_three_channels(torch.tensor(outputs.squeeze(1).cpu().numpy()))
        tar = merge_three_channels(torch.tensor(targets.squeeze(1).cpu().numpy()))
        #FINE PARTE NUOVA
        loss = criterion(outputs, targets)
        val_loss += loss.item()
 
mse_list = []
abs_diff_list = []
acc_red_list = []
acc_yellow_list = []
acc_blue_list = []
 
#inizio metriche
for frame_id in range(len(output_eval)):
    #print(f"out eval: {output_eval[frame_id].shape}") #(3,70,70)
    out = merge_three_channels(torch.tensor(output_eval[frame_id]))
    tar = merge_three_channels(torch.tensor(target_eval[frame_id]))
   
    mse = mean_squared_error(tar, out)
    mse_list.append(mse)
    abs_diff_value = calculate_absolute_difference(out, tar)
    abs_diff_list.append(abs_diff_value)
 
    accuracy_yellow, accuracy_red , accuracy_blue= calculate_accuracy(out, tar)
    acc_blue_list.append(accuracy_blue)
    acc_red_list.append(accuracy_red)
    acc_yellow_list.append(accuracy_yellow)
 
print(f"Test Loss: {val_loss / len(test_loader):.4f}")
mean_mse = np.mean(mse_list)
print(f"Mean Squared Error: {mean_mse:.4f}")
mean_abs_diff = np.mean(abs_diff_list)
print(f"Mean Absolute Difference: {mean_abs_diff:.4f}")
mean_accuracy_blue = np.mean(acc_blue_list)
print(f"Mean Accuracy Blue: {mean_accuracy_blue:.4f}")
mean_accuracy_red = np.mean(acc_red_list)
print(f"Mean Accuracy Red: {mean_accuracy_red:.4f}")
mean_accuracy_yellow = np.mean(acc_yellow_list)
print(f"Mean Accuracy Yellow: {mean_accuracy_yellow:.4f}")
 
 
visualize_results_grid(model, test_loader, device)
