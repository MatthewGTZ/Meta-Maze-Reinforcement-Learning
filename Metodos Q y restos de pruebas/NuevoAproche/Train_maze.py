import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Cargar el archivo de experiencias
file_path = '/home/gdanietz/Schreibtisch/Universidad/Programacion/Segundo semicuatrimestre/Aprendizaje por Refuerzo/TrabajoFInal/NuevoAproche/experiencias/Experiences-9-30-10000-0.35-v3.pickle'
with open(file_path, 'rb') as file:
    data = pickle.load(file)

# Normalizar los identificadores de los estados y preparar los datos para PyTorch
ids = data[:,:,0].astype(np.float32)
actions = data[:,:,1]

# Normalización
ids = (ids - np.mean(ids)) / np.std(ids)

# Convertir a tensores de PyTorch
ids = torch.tensor(ids, dtype=torch.float32)
actions = torch.tensor(actions, dtype=torch.long)

# Crear un conjunto de datos y un cargador de datos
dataset = TensorDataset(ids, actions)
data_loader = DataLoader(dataset, batch_size=1, shuffle=True)  # batch_size=1 para secuencias

# Definición del Modelo LSTM para Secuencias
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

# Funciones de Entrenamiento y Evaluación
def train(model, optimizer, criterion, epochs, data_loader):
    losses = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for states, actions_seq in data_loader:
            hidden = model.init_hidden(1)  # Inicializar para cada secuencia
            optimizer.zero_grad()
            for i in range(states.size(1)):  # Iterar a través de cada paso en la secuencia
                state = states[:, i].unsqueeze(-1).unsqueeze(0)  # Asegurar dimensiones correctas
                action = actions_seq[:, i]  # La acción correspondiente a este paso
                output, hidden = model(state, hidden)
                loss = criterion(output, action)  # 'action' ya debería ser 1D aquí
                total_loss += loss.item()
                loss.backward(retain_graph=True)
                print(f"Época {epoch+1}, Paso {i+1}, Pérdida: {loss.item():.4f}")  # Imprimir pérdida de cada paso
            optimizer.step()
        avg_loss = total_loss / len(data_loader)
        losses.append(avg_loss)
        print(f"Época {epoch+1}/{epochs}, Pérdida Promedio: {avg_loss:.4f}")  # Imprimir pérdida promedio de la época
    return losses

# Inicializar y entrenar el modelo
input_size = 1
hidden_size = 128
output_size = 4  # Suponiendo 4 acciones posibles
num_layers = 1  # Puedes ajustar el número de capas

model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

losses = train(model, optimizer, criterion, epochs=5, data_loader=data_loader)

# Guardar el modelo entrenado
model_path = '/home/gdanietz/Schreibtisch/Universidad/Programacion/Segundo semicuatrimestre/Aprendizaje por Refuerzo/TrabajoFInal/NuevoAproche/experiencias/pruena.pth'  # Cambia esto a la ruta deseada
torch.save(model.state_dict(), model_path)
# Graficar la pérdida durante el entrenamiento
plt.plot(losses)
plt.title('Progreso del Entrenamiento')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.show()
