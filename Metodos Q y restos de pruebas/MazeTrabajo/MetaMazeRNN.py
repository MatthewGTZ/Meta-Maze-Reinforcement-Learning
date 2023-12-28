import gym
import metagym.metamaze
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from metagym.metamaze import MazeTaskSampler

# Definición de la RNN que toma una secuencia de observaciones y predice la siguiente acción
class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden=None):
        x, hidden = self.rnn(x, hidden)
        x = self.fc(x[:, -1, :])  # Usamos la última salida para la predicción de la acción
        return torch.softmax(x, dim=-1), hidden

# Inicialización de la RNN
input_size = 9  # 3x3 observation matrix flattened
hidden_size = 128  # Puede ajustarse según la complejidad del entorno
output_size = 4  # 4 posibles acciones: N, S, W, E

model = RNNModel(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Función para convertir las observaciones y acciones en tensores
def prepare_sequence(observations, actions):
    obs_seq = np.array([obs.flatten() for obs in observations])
    act_seq = np.array(actions)
    return torch.tensor(obs_seq, dtype=torch.float), torch.tensor(act_seq, dtype=torch.long)
# ... [Código anterior y definiciones] ...

# Modificamos gather_experiences para incluir set_task y aplanar las observaciones
def gather_experiences_rnn(env, model, episodes, n_mazes):
    model.train()
    for episode in range(episodes):
        total_reward = 0
        for _ in range(n_mazes):
            # Establecer una nueva tarea
            task = MazeTaskSampler(...) # Define los parámetros de tu tarea aquí
            env.set_task(task.sample()) # Aplica la tarea al entorno
            
            observations = []
            actions = []
            rewards = []
            done = False
            state = env.reset()  # Ahora está bien llamar a reset después de set_task
            hidden = None

            while not done:
                env.render()  # Renderiza el entorno para visualización
                current_obs_flat = state.flatten()  # Aplana la observación
                current_obs = torch.tensor(current_obs_flat, dtype=torch.float).unsqueeze(0)
                action_probs, hidden = model(current_obs, hidden)
                action = torch.multinomial(action_probs, 1).item()
                observations.append(current_obs_flat)  # Guarda la observación aplanada
                actions.append(action)

                next_state, reward, done, _ = env.step(action)
                rewards.append(reward)
                state = next_state
                total_reward += reward

            # Entrenamiento de la RNN con las secuencias recopiladas
            obs_tensor, act_tensor = prepare_sequence(observations, actions)
            optimizer.zero_grad()
            action_scores, _ = model(obs_tensor.unsqueeze(0))
            loss = criterion(action_scores, act_tensor)
            loss.backward()
            optimizer.step()

        # Imprime estadísticas del episodio
        print(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

# Configuración del entorno y entrenamiento
env = gym.make("meta-maze-2D-v0", enable_render=True, task_type="ESCAPE")
gather_experiences_rnn(env, model, episodes=100, n_mazes=10)

# Cerrar el entorno al final
env.close()

# Guardar el modelo entrenado
torch.save(model.state_dict(), 'rnn_model.pth')
