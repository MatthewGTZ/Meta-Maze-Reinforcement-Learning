import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import matplotlib.pyplot as plt

# Cargar las experiencias guardadas
file_path = '/home/gdanietz/Schreibtisch/Universidad/Programacion/Segundo semicuatrimestre/Aprendizaje por Refuerzo/TrabajoFInal/MazeTrabajo/modelos/Experiences-9-30-1000.pickle'
with open(file_path, 'rb') as file:
    experiences = pickle.load(file)

# Preparar los datos para la RNN
observations = []
actions = []

for sequence in experiences:
    for step in sequence:
        observations.append(step[0])  # Observación
        actions.append(step[1])       # Acción

# Convertir a tensores para su uso en PyTorch
observations_tensor = torch.tensor(observations, dtype=torch.float)
actions_tensor = torch.tensor(actions, dtype=torch.long)

print(observations_tensor)
print(actions_tensor)