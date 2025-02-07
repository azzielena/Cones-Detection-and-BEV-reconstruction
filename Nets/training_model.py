from utils import load_grids_from_csv,create_three_channel_tensor, save_best_model
from models_net.uNet512 import UNet
from models_net.encoderDecoder import EncoderDecoder
from models_net.convolutionalNetwork import CNNModel

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim

# Caricamento dati
file_path_input = 'dataset/grid_input.csv'
file_path_output = 'dataset/grid_output.csv'
frames, grid_data_input = load_grids_from_csv(file_path_input)
frames, grid_data_output = load_grids_from_csv(file_path_output)

grid_tensors_input = [create_three_channel_tensor(torch.tensor(grid.reshape(70, 70), dtype=torch.float32)) for grid in grid_data_input]
grid_tensors_output = [create_three_channel_tensor(torch.tensor(grid.reshape(70, 70), dtype=torch.float32)) for grid in grid_data_output]

# Divisione training, validation e test set
x_train, x_test, y_train, y_test = train_test_split(grid_tensors_input, grid_tensors_output, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.25, random_state=42)

x_train = torch.stack(x_train)
x_val = torch.stack(x_val)
x_test = torch.stack(x_test)
y_train = torch.stack(y_train)
y_val = torch.stack(y_val)
y_test = torch.stack(y_test)

train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32, shuffle=False)
test_loader = DataLoader(TensorDataset(x_test, y_test), batch_size=32, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Configurazione training, selezionare quella desiderata e modificare di conseguenza anche il model_save_path 
#model = EncoderDecoder().to(device)
model = UNet().to(device)
#model = CNNModel().to(device)


# Scegli la funzione di perdita: 'mse', 'smooth_l1'
loss_function_choice = 'mse'

if loss_function_choice == 'mse':
    criterion = nn.MSELoss()
elif loss_function_choice == 'smooth_l1':
    criterion = nn.SmoothL1Loss()



optimizer = optim.Adam(model.parameters(), lr=0.001)
num_epochs = 75
patience = 3
best_val_loss = float('inf')
early_stopping_counter = 0
model_save_path = "Nets/model_result/best_modelUnet.pth"

# Training con early stopping
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_best_model(model, model_save_path)
        early_stopping_counter = 0
    else:
        early_stopping_counter += 1

    if early_stopping_counter >= patience:
        print("Early stopping triggered.")
        break



# Caricamento del modello migliore per il test finale
model.load_state_dict(torch.load(model_save_path))
model.eval()

test_loss = 0
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")
