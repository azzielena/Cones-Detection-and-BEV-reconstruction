import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
from torch import tensor
import torch.nn as nn
import torch.optim as optim
import matplotlib.colors as mcolors

# Funzione per caricare i dati dal CSV
def load_grids_from_csv(file_path):
    df = pd.read_csv(file_path)
    frames = df['Frame'].values
    grids = df.drop(columns=['Frame']).values
    return frames, grids

def create_three_channel_tensor(single_channel_tensor):
    if single_channel_tensor.dim() == 3 and single_channel_tensor.shape[0] == 1:
        single_channel_tensor = single_channel_tensor.squeeze(0)
    channel_blue = (single_channel_tensor == -1).float()
    channel_yellow = (single_channel_tensor == 1).float()
    channel_center = (single_channel_tensor == 0.5).float()
    return torch.stack([channel_blue, channel_yellow, channel_center], dim=0)

def merge_three_channels(tensor):
    """
    Ricompone un tensore a 3 canali in un'unica mappa con i valori originali.
    """
    output = torch.zeros_like(tensor[0])
    output[tensor[0] > 0.5] = -1       # Valori per il primo canale (blu)
    output[tensor[1] > 0.5] = 1      # Valori per il secondo canale (giallo)
    output[tensor[2] > 0.5] = 0.5     # Valori per il terzo canale (centro linea)
    return output
    
# Salvataggio del modello migliore
def save_best_model(model, path):
    torch.save(model.state_dict(), path)
    
# PRESTAZIONI
 
def calculate_absolute_difference(pred, target):
    return torch.abs(pred - target).mean().item()
 
#recall per categoria
def calculate_recall(output, target): #se out == tar and out !=0: count_right++; a prescindere totale++;
    # Crea una maschera per escludere i casi in cui output e target sono entrambi zero
    mask = ~((output == 0) & (target == 0))  # Mantiene solo i valori validi
    filtered_output = output[mask]
    filtered_target = target[mask]
 
    # Inizializzazione delle variabili per ogni categoria
    right_yellow = torch.sum((filtered_output == filtered_target) & (filtered_target == 1)).item()
    total_yellow = torch.sum(filtered_target == 1).item()
 
    right_blue = torch.sum((filtered_output == filtered_target) & (filtered_target == -1)).item()
    total_blue = torch.sum(filtered_target == -1).item()
 
    right_red = torch.sum((filtered_output == filtered_target) & (filtered_target == 0.5)).item()
    total_red = torch.sum(filtered_target == 0.5).item()
 
    # Calcolo dell'recall per ogni categoria
    recall_yellow = (right_yellow / total_yellow * 100) if total_yellow > 0 else 0
    recall_blue = (right_blue / total_blue * 100) if total_blue > 0 else 0
    recall_red = (right_red / total_red * 100) if total_red > 0 else 0
 
    return recall_yellow, recall_red, recall_blue
    
    
def visualize_results_grid(model, test_loader, device):

 
    # Creazione della colormap personalizzata
    colors = ['blue', 'white', 'red','yellow']  # Colori per i valori specificati
    bounds = [-1.5, -0.3, 0.4, 0.51, 1.5]
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
 
    def plot_grid(frame_id, input_data, target_data, output_data, ax_input, ax_target, ax_output):
        ax_input.clear()
        grid_input = merge_three_channels(torch.tensor(input_data[frame_id]))
        img_input = ax_input.imshow(grid_input, cmap=cmap, norm=norm, extent=[0, 21, -10.5, 10.5])
        ax_input.set_title(f"Input - frame{frame_id}")
        ax_input.set_xlabel('X (m)')
        ax_input.set_ylabel('Y (m)')
        ax_input.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
        ax_input.set_xticks(np.arange(0, 21, 0.3))
        ax_input.set_yticks(np.arange(-10.5, 10.5, 0.3))
 
        # Target
        ax_target.clear()
        grid_target = merge_three_channels(torch.tensor(target_data[frame_id]))
        img_target = ax_target.imshow(grid_target, cmap=cmap, norm=norm, extent=[0, 21, -10.5, 10.5])
        ax_target.set_title(f"Target - frame{frame_id}")
        ax_target.set_xlabel('X (m)')
        ax_target.set_ylabel('Y (m)')
        ax_target.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
        ax_target.set_xticks(np.arange(0, 21, 0.3))
        ax_target.set_yticks(np.arange(-10.5, 10.5, 0.3))
 
        # Output
        ax_output.clear()
        grid_output = merge_three_channels(torch.tensor(output_data[frame_id]))
        img_output = ax_output.imshow(grid_output, cmap=cmap, norm=norm, extent=[0, 21, -10.5, 10.5])
        ax_output.set_title(f"Output - frame{frame_id}")
        ax_output.set_xlabel('X (m)')
        ax_output.set_ylabel('Y (m)')
        ax_output.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
        ax_output.set_xticks(np.arange(0, 21, 0.3))
        ax_output.set_yticks(np.arange(-10.5, 10.5, 0.3))
 
        plt.draw()
 
    def on_key(event):
        nonlocal current_frame_idx
        if event.key == 'right' and current_frame_idx < num_frames - 1:
            current_frame_idx += 1
            plot_grid(current_frame_idx, input_data, target_data, output_data, ax_input, ax_target, ax_output)
        elif event.key == 'left' and current_frame_idx > 0:
            current_frame_idx -= 1
            plot_grid(current_frame_idx, input_data, target_data, output_data, ax_input, ax_target, ax_output)
 
    model.eval()
    input_data, target_data, output_data = [], [], []
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Normalizza e prepara i dati per la visualizzazione
            input_data.extend(inputs.squeeze(1).cpu().numpy())
            target_data.extend(targets.squeeze(1).cpu().numpy())
            output_data.extend(outputs.squeeze(1).cpu().numpy())
 
    # Configurazione iniziale della visualizzazione
    num_frames = len(input_data)
    current_frame_idx = 0
    fig, axes = plt.subplots(1,3,figsize=(15, 5))
    ax_input, ax_target, ax_output = axes
    plot_grid(current_frame_idx, input_data, target_data, output_data, ax_input, ax_target, ax_output)
 
    # Collega la funzione di gestione degli eventi
    fig.canvas.mpl_connect('key_press_event', on_key)
    # Mostra la finestra di visualizzazione
    plt.tight_layout()
    plt.show()
    
    
