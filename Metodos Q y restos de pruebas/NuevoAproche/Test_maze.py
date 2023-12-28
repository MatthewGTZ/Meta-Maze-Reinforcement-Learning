
import gym
import numpy as np
import random
from enum import Enum
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


import gym
import metagym.metamaze
from metagym.metamaze import MazeTaskSampler
import numpy as np
import random
from enum import Enum
import time

# Asegúrate de tener la definición de tu clase LSTMModel aquí
# class LSTMModel(nn.Module):
#     ...
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden):
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x[:, -1, :])  # Tomar solo la última salida de la secuencia
        return x, hidden

    def init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))

class Action(Enum):
    W = 0
    E = 1
    S = 2
    N = 3


def get_surrounding_value(surroundings,direction):
    if direction == Action.N.value:
        return surroundings[1][2]
    if direction == Action.S.value:
        return surroundings[1][0]
    if direction == Action.E.value:
        return surroundings[2][1]
    if direction == Action.W.value:
        return surroundings[0][1]
# Funciones auxiliares
def wall_at(surroundings, direction):
    # ...
    return get_surrounding_value(surroundings, direction) == -1.0

def get_state(state, action, surroundings):
    # ...
    return state

def get_lstm_action(model, observation, hidden):
    observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        output, hidden = model(observation_tensor, hidden)
    return torch.argmax(output, dim=1).item(), hidden
input_size = 1
hidden_size = 128
output_size = 4  # Suponiendo 4 acciones posibles
num_layers = 1  # Puedes ajustar el número de capas
# Cargar el modelo LSTM
model_path = '/home/gdanietz/Schreibtisch/Universidad/Programacion/Segundo semicuatrimestre/Aprendizaje por Refuerzo/TrabajoFInal/NuevoAproche/experiencias/pruena.pth'
model = LSTMModel(input_size, hidden_size, output_size, num_layers)  # Asegúrate de definir estos parámetros correctamente
model.load_state_dict(torch.load(model_path))
model.eval()

# Ejemplo de cómo ejecutar el entrenamiento
env = gym.make("meta-maze-2D-v0", enable_render=True, task_type="ESCAPE")
episodes = 100  # Define el número de episodios
hidden = model.init_hidden(1)

for episode in range(episodes):
    surroundings = env.reset()
    done = False
    state = 0

    while not done:
        action, hidden = get_lstm_action(model, surroundings, hidden)
        if wall_at(surroundings, action):
            continue
        observation, reward, done, info = env.step(action)
        new_state = get_state(state, action, surroundings)
        state = new_state
        surroundings = observation

        # Aquí puedes agregar lógica adicional si es necesario
