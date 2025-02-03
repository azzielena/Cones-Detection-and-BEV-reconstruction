import numpy as np

def process_multiple_dat_files(input_files, output_file, frame_arrays):
    """
    Filtra più file .dat in base ai rispettivi array di frame e concatena i risultati
    in un unico file con N_FRAME ricalcolato per essere unico e ordinato.

    Args:
        input_files (list[str]): Lista dei percorsi dei file .dat di input.
        output_file (str): Percorso del file di output finale.
        frame_arrays (list[list[int]]): Lista di array di frame, uno per ciascun file.
    """
    all_filtered_data = []
    current_frame_index = 1  # Iniziamo con N_FRAME = 1 per il file finale
    frame_mapping = {}  # Per mappare i vecchi frame ai nuovi indici

    for file_idx, (input_file, frame_array) in enumerate(zip(input_files, frame_arrays)):
        # Carica il file .dat
        try:
            data = np.loadtxt(input_file)
            if data.ndim == 1:  # Se il file ha una sola riga
                data = data.reshape(1, -1)
        except Exception as e:
            print(f"Errore durante il caricamento del file {input_file}: {e}")
            continue

        # Filtra le righe basandosi sui frame nell'array
        filtered_data = np.array([row for row in data if int(row[0]) in frame_array])
        if filtered_data.size == 0:
            continue  # Salta se non ci sono righe corrispondenti

        # Crea una nuova mappatura dei frame, mantenendo la sequenza
        for old_frame in np.unique(filtered_data[:, 0]):
            frame_mapping[old_frame] = current_frame_index
            current_frame_index += 1

        # Applica la nuova mappatura ai dati filtrati
        reindexed_data = np.array([
            [frame_mapping[row[0]], row[1], row[2]] for row in filtered_data
        ])
        
        # Aggiungi i dati reindicizzati alla lista
        all_filtered_data.append(reindexed_data)

    # Scrive il file con righe vuote tra i blocchi
    with open(output_file, "w") as f:
        #f.write("N_FRAME X Y\n")  # Intestazione
        for i, block in enumerate(all_filtered_data):
            if i > 0:  # Aggiunge una riga vuota tra i blocchi
                f.write("\n")
            for row in block:
                f.write(f"{int(row[0])} {row[1]:.6f} {row[2]:.6f}\n")


# è necessario modificare tutti i percorsi per ogni file blue e yellow cones e blue, yellow e red curve .dat
input_files = ["data_preprocessing/sequenze/bev_proj/center_curve_bev.dat",
               "data_preprocessing/sequenze/bev_proj1/center_curve_bev.dat",
               "data_preprocessing/sequenze/bev_proj2/center_curve_bev.dat",
               "data_preprocessing/sequenze/bev_proj3/center_curve_bev.dat",
               "data_preprocessing/sequenze/bev_proj4/center_curve_bev.dat",
               "data_preprocessing/sequenze/bev_proj5/center_curve_bev.dat",
               "data_preprocessing/sequenze/bev_proj6/center_curve_bev.dat",
               "data_preprocessing/sequenze/bev_proj7/center_curve_bev.dat",
               "data_preprocessing/sequenze/bev_proj8/center_curve_bev.dat"
               ]  # Percorsi dei file .dat
frame_arrays = [
    [21, 40],  # bev
    [1],  # bev1
    [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 14, 15, 18, 19, 22, 25, 28, 30, 33, 35, 36, 37, 38, 39, 42, 44, 45, 46, 47, 48, 49, 50, 51, 52, 55, 58, 59, 61, 64, 66, 68, 71, 72, 74, 77, 78, 79, 80, 82, 85, 86, 88, 89, 91, 92, 93, 96, 98, 99, 100, 102, 103, 104, 106, 108, 109, 111, 113, 114, 117, 118, 119, 120, 122, 124, 125, 126],  # bev2
    [30, 32,33, 35,36,37, 40, 41, 45, 48, 50, 51, 60, 61,62,63,64,65,66, 67, 71, 73, 75, 76, 77, 83,84,85, 87, 89, 90, 92,96,97,98,100,101,105],  # bev3
    [1, 14, 15, 17, 21, 26, 31, 32, 35, 38, 40, 46, 51, 53, 54, 62, 68, 70, 74, 77, 81, 87, 89, 96, 104, 106, 108, 113, 117, 118, 120, 121, 125, 129, 131, 132, 135, 140, 141],  # bev4
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 22, 27, 28, 30, 32, 34, 35, 37, 41, 45, 47, 48, 49, 53, 59, 60, 61, 63, 66, 68, 69, 72, 73, 75, 79, 80, 84, 85, 87, 137, 138, 143],  # bev5
    [7, 17, 18, 19, 24, 26, 27, 30, 33, 34, 39, 40, 43, 44, 46, 47, 49, 52, 53, 56, 59, 60, 61, 63, 65, 71, 72, 73, 76, 77, 81, 82, 83, 84, 85, 86, 87, 88, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105], #bev6
    [77],  # bev7
    [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 24, 27, 29, 30, 31, 32, 33, 36, 37, 38, 42, 45, 47, 49, 50, 51, 52, 53, 55, 60, 61, 63, 64, 65, 67, 68, 71, 73, 75, 77, 78, 82, 83, 84, 85, 86, 87, 88, 89, 94, 97, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119] # bev 8
]
output_file = "supporting_dataset/init_dataset/AIUTO_curve_bev_init.dat"  # Percorso del file di output (da cambiare ogni volta)

process_multiple_dat_files(input_files, output_file, frame_arrays)
