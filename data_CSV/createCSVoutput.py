import numpy as np
import os
import cv2

# Funzione per leggere il file .dat
def load_dat_file(file_path):
    return np.loadtxt(file_path)

# Funzione per mappare le coordinate sulla griglia
def map_to_grid(x, y, grid_size, grid_resolution):
    grid_x = int(x / grid_resolution)  # Mappa X
    grid_y = int((-y + 10.5) / grid_resolution)  # Mappa Y
    # Verifica che le coordinate siano all'interno dei limiti della griglia
    if 0 <= grid_x < grid_size and 0 <= grid_y < grid_size:
        return grid_x, grid_y
    return None  # Fuori dalla griglia

# Funzione per disegnare linee sottili
def draw_curve(grid, curve_data, curve_value, thickness=1):
    points = []
    for cone in curve_data:
        x, y = cone
        grid_coords = map_to_grid(x, y, grid_size, grid_resolution)
        if grid_coords is not None:
            points.append(grid_coords)

    if points:
        points = np.array(points, dtype=np.int32)
        # Disegna la curva come linea sottile
        cv2.polylines(grid, [points], isClosed=False, color=curve_value, thickness=thickness)

# Parametri della griglia
grid_size = 70  # La griglia è 70x70
grid_resolution = 0.3  # Ogni cella è 30 cm

# Inizializza la griglia
grid = np.zeros((grid_size, grid_size), dtype=np.float32)

# Percorso del dataset
base_path = 'dataset/final_dataset'

# Caricamento dati
blue_curve = load_dat_file(os.path.join(base_path, 'blue_curve.dat'))
yellow_curve = load_dat_file(os.path.join(base_path, 'yellow_curve.dat'))
center_curve = load_dat_file(os.path.join(base_path, 'center_curve.dat'))

# Frame unici
frames = np.unique(blue_curve[:, 0])

# Creazione CSV
output_file = 'dataset/grid_output.csv'
with open(output_file, 'w') as f:
    f.write("Frame," + ",".join([str(i) for i in range(grid_size * grid_size)]) + "\n")
    for frame_id in frames:
        # Dati per frame
        blue_curve_frame = blue_curve[blue_curve[:, 0] == frame_id, 1:3]
        yellow_curve_frame = yellow_curve[yellow_curve[:, 0] == frame_id, 1:3]
        center_curve_frame = center_curve[center_curve[:, 0] == frame_id, 1:3]

        # Resetta la griglia
        grid = np.zeros_like(grid)

        # Disegna linee sottili
        draw_curve(grid, blue_curve_frame, -1, thickness=1)  # Blu
        draw_curve(grid, yellow_curve_frame, 1, thickness=1)  # Giallo
        draw_curve(grid, center_curve_frame, 0.5, thickness=1)  # Rossa (linea centrale)

        # Salva griglia
        flat_grid = grid.flatten()
        f.write(f"{frame_id}," + ",".join(map(str, flat_grid)) + "\n")

print(f"File salvato in: {output_file}")
