import numpy as np
import os

# Funzione per leggere il file .dat
def load_dat_file(file_path):
    return np.loadtxt(file_path)

# Funzione per mappare le coordinate sulla griglia 70x70
def map_to_grid(x, y, grid_size, grid_resolution): #grid resolution 30cm
    grid_x = int(x / grid_resolution)  # Mappa X (orizzontale)
    grid_y = int((-y + 10.5) / grid_resolution)  # Mappa Y (verticale)
    
    # Verifica che le coordinate siano all'interno dei limiti della griglia
    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
        return grid_x, grid_y
    else:
        return None  # Fuori dalla griglia

# Funzione per aggiornare la griglia con i dati dei coni
def update_grid_with_cones(grid, cones_data, cone_type):
    for cone in cones_data:
        x, y = cone
        grid_coords = map_to_grid(x, y, grid_size, grid_resolution)
        
        if grid_coords is not None:
            grid_x, grid_y = grid_coords
            if cone_type == 'yellow':  # I coni yellow sono 1
                grid[grid_y, grid_x] = 1
            elif cone_type == 'blue':  # I coni blu sono -1
                grid[grid_y, grid_x] = -1

# Parametri della griglia
grid_size = 70  # La griglia è 70x70
grid_resolution = 0.3  # Ogni cella è 30 cm (0.3 metri)
horizon = 21  # L'orizzonte è di 20 metri

# Inizializza la griglia con zeri
grid = np.zeros((grid_size, grid_size))

# Percorso del dataset
base_path = 'dataset/final_dataset'

# Leggi i dati dai file
blue_cones = load_dat_file(os.path.join(base_path, 'blue_cones.dat'))
yellow_cones = load_dat_file(os.path.join(base_path, 'yellow_cones.dat'))

# Estrai i dati per il primo frame
frames = np.unique(blue_cones[:, 0])  # Supponiamo che i frame siano gli stessi per entrambi i file

# Creazione di un file CSV per salvare i dati delle griglie
output_file = 'dataset/grid_input.csv'
with open(output_file, 'w') as f:
    f.write("Frame," + ",".join([str(i) for i in range(grid_size * grid_size)]) + "\n")
    
    # Ciclo per ogni frame
    for frame_id in frames:
        # Estrai i dati per il frame corrente
        blue_cones_frame = blue_cones[blue_cones[:, 0] == frame_id, 1:3]
        yellow_cones_frame = yellow_cones[yellow_cones[:, 0] == frame_id, 1:3]
        
        # Resetta la griglia a zeri prima di ogni nuovo frame
        grid.fill(0)

        # Aggiorna la griglia con i coni blu (imposta a 1)
        update_grid_with_cones(grid, blue_cones_frame, 'blue')

        # Aggiorna la griglia con i coni gialli (imposta a -1)
        update_grid_with_cones(grid, yellow_cones_frame, 'yellow')
        
        # Aplanare la griglia (trasformarla in un array 1D)
        flat_grid = grid.flatten()
        
        # Scrivi il frame ID seguito dai valori della griglia piatta nel file CSV
        f.write(f"{frame_id}," + ",".join([str(int(value)) for value in flat_grid]) + "\n")

print(f"I dati delle griglie sono stati salvati in: {output_file}")