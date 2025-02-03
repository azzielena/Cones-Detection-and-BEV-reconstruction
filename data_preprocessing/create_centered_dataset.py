import os
import numpy as np
import matplotlib.pyplot as plt

# Percorso base dei file
base_path = r'supporting_dataset\init_dataset'
# Cambia questo percorso in base alla directory dei file

# Funzione per leggere un file .dat
def load_dat_file(file_path):
    return np.loadtxt(file_path)

# Lettura dei file
yellow_cones = load_dat_file(os.path.join(base_path, 'yellow_cones_bev_init.dat'))
blue_cones = load_dat_file(os.path.join(base_path, 'blue_cones_bev_init.dat'))
yellow_curve = load_dat_file(os.path.join(base_path, 'yellow_curve_bev_init.dat'))
blue_curve = load_dat_file(os.path.join(base_path, 'blue_curve_bev_init.dat'))
center_curve = load_dat_file(os.path.join(base_path, 'center_curve_bev_init.dat'))

# Estrazione dei frame unici
frames = np.unique(center_curve[:, 0])
num_frames = len(frames)
current_frame_idx = 0  # Iniziamo dal primo frame



# Funzione per salvare i dati trasformati in un unico file
def save_all_transformed_data(output_file, data_list):
    # Converti la lista in un array NumPy
    all_data = np.vstack(data_list)
    # Salva l'array in un unico file
    np.savetxt(output_file, all_data, fmt='%d %.6f %.6f', comments='')



# Funzione per tracciare il frame corrente
def plot_frame(frame_idx):
    
    frame_id = frames[frame_idx]

    # Filtra i dati per il frame corrente
    yellow_cones_frame = yellow_cones[yellow_cones[:, 0] == frame_id, 1:3]
    blue_cones_frame = blue_cones[blue_cones[:, 0] == frame_id, 1:3]
    yellow_curve_frame = yellow_curve[yellow_curve[:, 0] == frame_id, 1:3]
    blue_curve_frame = blue_curve[blue_curve[:, 0] == frame_id, 1:3]
    center_curve_frame = center_curve[center_curve[:, 0] == frame_id, 1:3]   # (cx,cy)


    # Trova dove inizia la center line
    dx=-center_curve_frame[0][0]
    dy=-center_curve_frame[0][1]

    # Traslazione di tutti i punti
    center_curve_frame[:, 0] += dx
    center_curve_frame[:, 1] += dy

    yellow_cones_frame[:, 0] += dx
    yellow_cones_frame[:, 1] += dy

    blue_cones_frame[:, 0] += dx
    blue_cones_frame[:, 1] += dy

    yellow_curve_frame[:, 0] += dx
    yellow_curve_frame[:, 1] += dy

    blue_curve_frame[:, 0] += dx
    blue_curve_frame[:, 1] += dy    

    
    if(center_curve_frame[0][0]>center_curve_frame[1][0]):
        # stiamo andando da destra verso sinistra quindi dobbiamo flippare e cambiare colore

        # Flip
        center_curve_frame[:, 0] *=-1
        yellow_cones_frame[:, 0] *=-1
        blue_cones_frame[:, 0] *=-1
        yellow_curve_frame[:, 0] *=-1
        blue_curve_frame[:, 0] *=-1

        # cambiare colore
        ex_blu_cone=blue_cones_frame
        blue_cones_frame=yellow_cones_frame
        yellow_cones_frame=ex_blu_cone

        ex_blu_curve=blue_curve_frame
        blue_curve_frame=yellow_curve_frame
        yellow_curve_frame = ex_blu_curve


        # Aggiungi i dati trasformati alla lista
    for point in yellow_cones_frame:
        y_cones_transformed_data.append([frame_id, point[0], point[1]])
    for point in blue_cones_frame:
        b_cones_transformed_data.append([frame_id, point[0], point[1]])
    for point in yellow_curve_frame:
        y_curve_transformed_data.append([frame_id, point[0], point[1]])
    for point in blue_curve_frame:
        b_curve_transformed_data.append([frame_id, point[0], point[1]])
    for point in center_curve_frame:
        center_transformed_data.append([frame_id, point[0], point[1]])





# Lista per accumulare tutti i dati trasformati
y_cones_transformed_data=[]
b_cones_transformed_data=[]
y_curve_transformed_data=[]
b_curve_transformed_data=[]
center_transformed_data=[]

# Itera su tutti i frame e salva i dati
for frame_idx in range(num_frames):
    plot_frame(frame_idx)

# Salva tutti i dati trasformati in un unico file
y_cones_file = "supporting_dataset/centered_dataset/yellow_cones.dat"
save_all_transformed_data(y_cones_file, y_cones_transformed_data)

y_curve_file = "supporting_dataset/centered_dataset/yellow_curve.dat"
save_all_transformed_data(y_curve_file, y_curve_transformed_data)

b_cones_file = "supporting_dataset/centered_dataset/blue_cones.dat"
save_all_transformed_data(b_cones_file, b_cones_transformed_data)

b_curve_file = "supporting_dataset/centered_dataset/blue_curve.dat"
save_all_transformed_data(b_curve_file, b_curve_transformed_data)

center_file = "supporting_dataset/centered_dataset/center_curve.dat"
save_all_transformed_data(center_file, center_transformed_data)
