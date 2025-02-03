import os
import numpy as np
import matplotlib.pyplot as plt

# Percorso base dei file
base_path = r'New Dataset\rotated_dataset'

# Funzione per leggere un file .dat
def load_dat_file(file_path):
    return np.loadtxt(file_path)

# Lettura dei file
yellow_cones = load_dat_file(os.path.join(base_path, 'yellow_cones_roto.dat'))
blue_cones = load_dat_file(os.path.join(base_path, 'blue_cones_roto.dat'))
yellow_curve = load_dat_file(os.path.join(base_path, 'yellow_curve_roto.dat'))
blue_curve = load_dat_file(os.path.join(base_path, 'blue_curve_roto.dat'))
center_curve = load_dat_file(os.path.join(base_path, 'center_curve_roto.dat'))

# Estrazione dei frame unici
frames = np.unique(center_curve[:, 0])
num_frames = len(frames)
current_frame_idx = 0  # Iniziamo dal primo frame



# Funzione per salvare i dati trasformati in un unico file
def save_all_transformed_data(output_file, data_list, original):
    orig_data = np.array(original)
    trasf = np.array(data_list)
    
    # Concatenazione dei dati: prima i dati originali, poi quelli trasformati
    all_data = np.vstack((orig_data, trasf))
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

    # Flip rispetto all'asse x
    center_curve_frame_flipy = center_curve_frame.copy()
    yellow_cones_frame_flipy = yellow_cones_frame.copy()
    blue_cones_frame_flipy = blue_cones_frame.copy()
    yellow_curve_frame_flipy = yellow_curve_frame.copy()
    blue_curve_frame_flipy = blue_curve_frame.copy()
    
    center_curve_frame_flipy[:, 1] *=-1
    yellow_cones_frame_flipy[:, 1] *=-1
    blue_cones_frame_flipy[:, 1] *=-1
    yellow_curve_frame_flipy[:, 1] *=-1
    blue_curve_frame_flipy[:,1] *=-1
    

    # cambiare colore
    blue_cones_frame_flipy, yellow_cones_frame_flipy=yellow_cones_frame_flipy,blue_cones_frame_flipy
    blue_curve_frame_flipy,yellow_curve_frame_flipy=yellow_curve_frame_flipy,blue_curve_frame_flipy


        # Aggiungi i dati trasformati alla lista
    for point in yellow_cones_frame_flipy:
        y_cones_transformed_data.append([frame_id+1700, point[0], point[1]])
    for point in blue_cones_frame_flipy:
        b_cones_transformed_data.append([frame_id+1700, point[0], point[1]])
    for point in yellow_curve_frame_flipy:
        y_curve_transformed_data.append([frame_id+1700, point[0], point[1]])
    for point in blue_curve_frame_flipy:
        b_curve_transformed_data.append([frame_id+1700, point[0], point[1]])
    for point in center_curve_frame_flipy:
        center_transformed_data.append([frame_id+1700, point[0], point[1]])





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
y_cones_file = "New Dataset/final_dataset/yellow_cones.dat"
save_all_transformed_data(y_cones_file, y_cones_transformed_data, yellow_cones)

y_curve_file = "New Dataset/final_dataset/yellow_curve.dat"
save_all_transformed_data(y_curve_file, y_curve_transformed_data,yellow_curve)

b_cones_file = "New Dataset/final_dataset/blue_cones.dat"
save_all_transformed_data(b_cones_file, b_cones_transformed_data,blue_cones)

b_curve_file = "New Dataset/final_dataset/blue_curve.dat"
save_all_transformed_data(b_curve_file, b_curve_transformed_data,blue_curve)

center_file = "New Dataset/final_dataset/center_curve.dat"
save_all_transformed_data(center_file, center_transformed_data,center_curve)
