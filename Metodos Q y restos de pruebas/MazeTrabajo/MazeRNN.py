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

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        x, hidden = self.rnn(x, hidden)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.fc(x[:, -1, :])
        return x

input_size = 1
hidden_size = 128
output_size = 4

model = RNNModel(input_size, hidden_size, output_size)

# Entrenar la RNN
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.03)

num_epochs = 64
batch_size = 32
total_batches = len(observations_tensor) // batch_size
loss_values = []

for epoch in range(num_epochs):
    total_loss = 0
    for i in range(total_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        batch_observations = observations_tensor[start_idx:end_idx].unsqueeze(-1)
        batch_actions = actions_tensor[start_idx:end_idx]

        optimizer.zero_grad()
        output = model(batch_observations)
        loss = criterion(output, batch_actions)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / total_batches
    loss_values.append(average_loss)

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {average_loss}')

# Graficar la pérdida
plt.plot(loss_values)
plt.title('Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Guardar el modelo entrenado
model_save_path = '/home/gdanietz/Schreibtisch/Universidad/Programacion/Segundo semicuatrimestre/Aprendizaje por Refuerzo/TrabajoFInal/MazeTrabajo/modelos/rnn_model.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Modelo entrenado guardado en '{model_save_path}'")
