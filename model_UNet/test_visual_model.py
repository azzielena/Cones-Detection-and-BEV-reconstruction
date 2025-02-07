from utils import visualize_results_grid, calculate_absolute_difference, calculate_recall, create_three_channel_tensor, load_grids_from_csv, merge_three_channels
from model_nets.uNet512 import UNet512
from model_nets.uNet1024 import UNet1024

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
model_path = r"model_UNet\model_result\best_modelUnet.pth"

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
file_path_input = 'dataset/grid_input.csv'
frames, grid_data_input = load_grids_from_csv(file_path_input)
 
file_path_output = 'dataset/grid_output.csv'
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
    grid_tensors_input, grid_tensors_output, test_size=0.2, random_state=12 #ex seed 42
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
        
        output_eval.extend(outputs.squeeze(1).cpu().numpy())
        target_eval.extend(targets.squeeze(1).cpu().numpy())

        loss = criterion(outputs, targets)
        val_loss += loss.item()
 
mse_list = []
abs_diff_list = []
recall_red_list = []
recall_yellow_list = []
recall_blue_list = []
 
# inizio delle metriche
for frame_id in range(len(output_eval)):
    # applico le metriche riunendo i tre canali
    out = merge_three_channels(torch.tensor(output_eval[frame_id]))
    tar = merge_three_channels(torch.tensor(target_eval[frame_id]))
   
    mse = mean_squared_error(tar, out)
    mse_list.append(mse)
    abs_diff_value = calculate_absolute_difference(out, tar)
    abs_diff_list.append(abs_diff_value)
 
    recall_yellow, recall_red , recall_blue= calculate_recall(out, tar)
    recall_blue_list.append(recall_blue)
    recall_red_list.append(recall_red)
    recall_yellow_list.append(recall_yellow)
 
print(f"Test Loss: {val_loss / len(test_loader):.4f}")
mean_mse = np.mean(mse_list)
print(f"Mean Squared Error: {mean_mse:.4f}")
mean_abs_diff = np.mean(abs_diff_list)
print(f"Mean Absolute Difference: {mean_abs_diff:.4f}")
mean_recall_blue = np.mean(recall_blue_list)
print(f"Mean Recall Blue: {mean_recall_blue:.4f}")
mean_recall_red = np.mean(recall_red_list)
print(f"Mean Recall Red: {mean_recall_red:.4f}")
mean_recall_yellow = np.mean(recall_yellow_list)
print(f"Mean Recall Yellow: {mean_recall_yellow:.4f}")

# Risultati visivi
visualize_results_grid(model, test_loader, device)
