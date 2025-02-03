import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd

# Creazione della colormap personalizzata
colors = ['blue', 'white', 'red', 'yellow']  # Colori per i valori specificati
bounds = [-1.5, -0.3, 0.4, 0.51, 1.5]
cmap = mcolors.ListedColormap(colors)
norm = mcolors.BoundaryNorm(bounds, cmap.N)

# Funzione per caricare i dati dal CSV
def load_grids_from_csv(file_path):
    df = pd.read_csv(file_path)
    frames = df['Frame'].values
    grids = df.drop(columns=['Frame']).values.reshape(len(frames), 70, 70)
    return frames, grids

# Caricamento dei dati dalle tre griglie
frames, grid_input = load_grids_from_csv('dataset/grid_input.csv')
_, grid_output = load_grids_from_csv('dataset/grid_output.csv')
num_frames = len(frames)
current_frame_idx = 0

# Funzione per visualizzare le tre griglie
def plot_grid(frame_id, input_data, output_data, ax_input, ax_output):
    ax_input.clear()
    ax_output.clear()
    
    # Input
    ax_input.imshow(input_data[frame_id], cmap=cmap, norm=norm, extent=[0, 21, -10.5, 10.5])
    ax_input.set_title("Input")
    ax_input.set_xlabel('X (m)')
    ax_input.set_ylabel('Y (m)')
    ax_input.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    ax_input.set_xticks(np.arange(0, 21, 0.3))
    ax_input.set_yticks(np.arange(-10.5, 10.5, 0.3))
    
    # Output
    ax_output.imshow(output_data[frame_id], cmap=cmap, norm=norm, extent=[0, 21, -10.5, 10.5])
    ax_output.set_title("Output")
    ax_output.set_xlabel('X (m)')
    ax_output.set_ylabel('Y (m)')
    ax_output.grid(which='both', color='gray', linestyle='--', linewidth=0.5)
    ax_output.set_xticks(np.arange(0, 21, 0.3))
    ax_output.set_yticks(np.arange(-10.5, 10.5, 0.3))

    
    plt.draw()

# Creazione della figura e degli assi
fig, (ax_input, ax_output) = plt.subplots(1, 2, figsize=(18, 6))
plot_grid(current_frame_idx, grid_input, grid_output, ax_input, ax_output)

# Funzione per la gestione degli eventi della tastiera
def on_key(event):
    global current_frame_idx
    if event.key == 'right' and current_frame_idx < num_frames - 1:
        current_frame_idx += 1
        plot_grid(current_frame_idx, grid_input, grid_output, ax_input, ax_output)
    elif event.key == 'left' and current_frame_idx > 0:
        current_frame_idx -= 1
        plot_grid(current_frame_idx, grid_input, grid_output, ax_input, ax_output)

fig.canvas.mpl_connect('key_press_event', on_key)
plt.tight_layout()
plt.show()
