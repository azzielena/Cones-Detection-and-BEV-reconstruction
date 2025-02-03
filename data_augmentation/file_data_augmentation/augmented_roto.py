import os
import numpy as np

# Percorso base dei file
base_path = r'Dataset_Nuovi\dataset_centrato'
output_path = r'Dataset_Nuovi\dataset_ruotato_ELE'

# Funzione per leggere un file .dat
def load_dat_file(file_path):
    return np.loadtxt(file_path)

# Lettura dei file
yellow_cones = load_dat_file(os.path.join(base_path, 'yellow_cones.dat'))
blue_cones = load_dat_file(os.path.join(base_path, 'blue_cones.dat'))
yellow_curve = load_dat_file(os.path.join(base_path, 'yellow_curve.dat'))
blue_curve = load_dat_file(os.path.join(base_path, 'blue_curve.dat'))
center_curve = load_dat_file(os.path.join(base_path, 'center_curve.dat'))

# Funzione per rototraslazione
def roto_translation(data, theta, dx, dy):
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    rotated = data[:, 1:3] @ rotation_matrix.T
    translated = rotated + np.array([dx, dy])
    return np.column_stack((data[:, 0], translated))

'''
# Funzione per flip lungo l'asse y
def flip_y(data):
    flipped = data.copy()
    flipped[:, 2] *= -1  # Flip sull'asse y
    return flipped
'''

# Funzione per generare nuovi ID frame
def augment_ids(data, start_id):
    frame_ids = np.unique(data[:, 0])
    new_ids = {frame_id: start_id + i for i, frame_id in enumerate(frame_ids)}
    augmented = data.copy()
    augmented[:, 0] = np.vectorize(new_ids.get)(augmented[:, 0])
    return augmented

# Data augmentation globale
def augment_dataset(data, start_id):
    dx, dy = -center_curve[0, 1], -center_curve[0, 2]  # Calcolo traslazione globale
    transformed_45 = roto_translation(data, np.pi /4 , dx, dy)
    transformed_30 = roto_translation(data, np.pi / 6, dx, dy)
    transformed_15= roto_translation(data, np.pi /12 , dx, dy)
    transformed_22= roto_translation(data, np.pi /8 , dx, dy)

    # Assegna nuovi ID
    start_id_30= start_id 
    start_id_45 = start_id + 340
    start_id_15 = start_id + 340*2
    start_id_22 = start_id + 340*3

    augmented_30 = augment_ids(transformed_30, start_id_30)
    augmented_45 = augment_ids(transformed_45, start_id_45)
    augmented_15 = augment_ids(transformed_15, start_id_15)
    augmented_22 = augment_ids(transformed_22, start_id_22)
    
    return np.vstack((data ,augmented_30, augmented_45, augmented_15, augmented_22))

# Applicare augmentazione ai dataset
blue_cones_augmented = augment_dataset(blue_cones, 341)
yellow_cones_augmented = augment_dataset(yellow_cones, 341)
blue_curve_augmented = augment_dataset(blue_curve, 341)
yellow_curve_augmented = augment_dataset(yellow_curve, 341)
center_curve_augmented = augment_dataset(center_curve, 341)

# Salvare i file augmentati
def save_to_file(data, filename):
    output_file = os.path.join(output_path, filename)
    np.savetxt(output_file, data, fmt='%d %.6f %.6f')

os.makedirs(output_path, exist_ok=True)
save_to_file(blue_cones_augmented, "blue_cones_roto.dat")
save_to_file(yellow_cones_augmented, "yellow_cones_roto.dat")
save_to_file(blue_curve_augmented, "blue_curve_roto.dat")
save_to_file(yellow_curve_augmented, "yellow_curve_roto.dat")
save_to_file(center_curve_augmented, "center_curve_roto.dat")
