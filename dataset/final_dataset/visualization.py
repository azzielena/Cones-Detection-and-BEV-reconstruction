import os
import numpy as np
import matplotlib.pyplot as plt

# Cambia questo percorso in base alla directory dei file
base_path = r'dataset/final_dataset'


# Funzione per leggere un file .dat
def load_dat_file(file_path):
    return np.loadtxt(file_path)

# Lettura dei file
yellow_cones = load_dat_file(os.path.join(base_path, 'yellow_cones.dat'))
blue_cones = load_dat_file(os.path.join(base_path, 'blue_cones.dat'))
yellow_curve = load_dat_file(os.path.join(base_path, 'yellow_curve.dat'))
blue_curve = load_dat_file(os.path.join(base_path, 'blue_curve.dat'))
center_curve = load_dat_file(os.path.join(base_path, 'center_curve.dat'))

# Estrazione dei frame unici
frames = np.unique(center_curve[:, 0])
num_frames = len(frames)
current_frame_idx = 0  # Iniziamo dal primo frame

# Funzione per tracciare il frame corrente
def plot_frame(frame_idx, ax1, ax2):
    frame_id = frames[frame_idx]

    # Filtra i dati per il frame corrente
    yellow_cones_frame = yellow_cones[yellow_cones[:, 0] == frame_id, 1:3]
    blue_cones_frame = blue_cones[blue_cones[:, 0] == frame_id, 1:3]
    yellow_curve_frame = yellow_curve[yellow_curve[:, 0] == frame_id, 1:3]
    blue_curve_frame = blue_curve[blue_curve[:, 0] == frame_id, 1:3]
    center_curve_frame = center_curve[center_curve[:, 0] == frame_id, 1:3]

    # Pulisci gli assi
    ax1.clear()
    ax2.clear()

    # Plot sinistra: coni
    ax1.scatter(yellow_cones_frame[:, 0], yellow_cones_frame[:, 1], c='yellow', label='Yellow Cones', s=50)
    ax1.scatter(blue_cones_frame[:, 0], blue_cones_frame[:, 1], c='blue', label='Blue Cones', s=50)
    ax1.set_title(f'Cones (Blue & Yellow) - Frame {int(frame_id)}')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.grid(True)
    ax1.axis('equal')
    ax1.legend()

    # Plot destra: curve
    ax2.plot(center_curve_frame[:, 0], center_curve_frame[:, 1], '-k', label='Center Curve', linewidth=1.5)
    ax2.plot(yellow_curve_frame[:, 0], yellow_curve_frame[:, 1], 'y.', label='Yellow Curve', markersize=10)
    ax2.plot(blue_curve_frame[:, 0], blue_curve_frame[:, 1], 'b.', label='Blue Curve', markersize=10)
    ax2.set_title(f'Curves (Center, Yellow, Blue) - Frame {int(frame_id)}')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()

    plt.draw()

# Funzione per gestire gli eventi della tastiera
def on_key(event):
    global current_frame_idx
    if event.key == 'right' and current_frame_idx < num_frames - 1:
        current_frame_idx += 1
        plot_frame(current_frame_idx, ax1, ax2)
    elif event.key == 'left' and current_frame_idx > 0:
        current_frame_idx -= 1
        plot_frame(current_frame_idx, ax1, ax2)

# Crea la finestra di visualizzazione
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.canvas.mpl_connect('key_press_event', on_key)
plot_frame(current_frame_idx, ax1, ax2)

# Mostra la figura
plt.tight_layout()
plt.show()
