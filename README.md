# 📌 Cones Detection and BEV Reconstruction

## 📖 Project Description
In our project we focused on the creation of a neural network with the aim of reconstructing the roadway starting from the position of the blue and yellow cones in world coordinates that delimit its boundaries.

### 📂 Initial Input
Files containing (x, y) coordinates of yellow and blue cones collected from specific frames.

### 🛣️ Goal
Reconstruction of roadway boundaries and the centerline.

## 🔄 Project Development Pipeline

### 1️⃣ Dataset Preprocessing
✔️ Feature selection to remove unclear or erroneous frames and apply transformations to standardize the data.

✔️ Data augmentation to enrich the dataset and enhance the model's robustness.

✔️ Conversion into a grid format to serve as input for the neural network.

### 2️⃣ Neural Network Model Development
🧠 Various neural network models were developed and tested to identify the most suitable architecture for solving the problem.

### 3️⃣ Performance Evaluation and Results Visualization
📈 Evaluating the model's effectiveness in reconstructing the road and accurately localizing the cones.

## 📂 Repository Structure
Here’s how to navigate through the repository:

- **`data_preprocessing`** ➝ Contains input datasets with cone coordinates.
  - `sequenze/` ➝ Contains input datasets with cone coordinates.
  - `create_initial_dataset.py` ➝ Extracts the most relevant inputs by removing invalid ones.
  - `create_centered_dataset.py` ➝ standardizes the inputs, aligning all roadways to start from position (0,0).
- **`data_augmentation/`** ➝ Includes files that apply rotation and flipping along the x-axis to increase the number of available inputs.
- **`supporting_dataset/`** ➝ Contains the intermediate datasets generated by the preprocessing and data augmentation phase
- **`data_CSV/`** ➝ 

## 🚀 How to Use the Project
Per eseguire la nostra rete principale UNET e risolvere il task di identificazione della carreggiata si possono eseguire i seguenti passi. 

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/azzielena/Cones-Detection-and-BEV-reconstruction.git
cd Cones-Detection-and-BEV-reconstruction
```

### 2️⃣ Install Dependencies
It is recommended to use a virtual environment:
```bash
pip install -r requirements.txt
```

### 3️⃣ Start Training
```bash
python model_UNet/training_model.py
```

### 4️⃣ Evaluate Results
```bash
python model_UNet/test_visual_model.py
```

## 🎨 Results Visualization
The reconstructed roadway results can be viewed in the `results/` folder.

## 🤝 Contributions
This project was developed in collaboration with @ariannaCella.

If you would like to contribute, feel free to open an Issue or submit a Pull Request! 🎉
