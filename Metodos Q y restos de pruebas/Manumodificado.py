import gym
import metagym.metamaze
from metagym.metamaze import MazeTaskSampler
import numpy as np
from enum import Enum
import time

class Action(Enum):
    W = 0
    E = 1
    S = 2
    N = 3

def change_state(state, action, surroundings):
    new_state = state

    if action == Action.N.value:
        if surroundings[1][2] != -1:
            if state < grid_size:
                new_state = state + (grid_size * (grid_size - 1))
            else:
                new_state = state - grid_size

    elif action == Action.S.value:
        if surroundings[1][0] != -1:
            if state >= grid_size * (grid_size - 1):
                new_state = state - (grid_size * (grid_size - 1))
            else:
                new_state = state + grid_size

    elif action == Action.W.value:
        if surroundings[0][1] != -1:
            if state == 0:
                new_state = (grid_size * grid_size) - 1
            else:
                new_state = state - 1

    elif action == Action.E.value:
        if surroundings[1][1] != -1:
            if state == (grid_size * grid_size) - 1:
                new_state = 0
            else:
                new_state = state + 1

    return new_state

def get_action(state, opciones, probabilidades):
    if np.random.choice(opciones, p=probabilidades) == 0:
        action = env.action_space.sample()
    else:
        if np.max(q_matriz[state]) > 0:
            action = np.argmax(q_matriz[state])
        else:
            action = env.action_space.sample()
    
    return action

# Configuración del entorno y parámetros
env = gym.make("meta-maze-2D-v0", enable_render=True, task_type="ESCAPE")
grid_size = 9
task = MazeTaskSampler(n=grid_size, allow_loops=True, crowd_ratio=0.35)
env.set_task(task)

q_matriz = np.zeros((grid_size * grid_size, 4))

# Parámetros del algoritmo Q-learning
episodes = 2000
alpha = 0.5         # Tasa de aprendizaje
factor_desc = 0.99   # Factor de descuento
epsilon = 1.0       # Para la exploración y explotación
epsilon_decay = 0.001

for i in range(episodes):
    print("Episodio:", i + 1)
    surroundings = env.reset()
    done = False
    state = 0

    while not done:
        opciones = [0, 1] # 0 para acción aleatoria, 1 para la mejor acción basada en Q
        probabilidades = [epsilon, 1 - epsilon]

        action = get_action(state, opciones, probabilidades)

        observation, reward, done, info = env.step(action)
        new_state = change_state(state, action, surroundings)

        # Actualización de la fórmula Q-learning incluyendo alpha
        q_matriz[state, action] = (1 - alpha) * q_matriz[state, action] + \
                                  alpha * (reward + factor_desc * np.max(q_matriz[new_state]))

        state = new_state
        surroundings = observation

        epsilon = max(epsilon - epsilon_decay, 0)

# Evaluación
test_episodes = 3
nb_success = 0
for i in range(test_episodes):
    surroundings = env.reset()
    done = False
    state = 0
    
    while not done:
        action = np.argmax(q_matriz[state])
        observation, reward, done, info = env.step(action)
        env.render()
        new_state = change_state(state, action, surroundings)
        state = new_state
        surroundings = observation
        if reward > 0:
            nb_success += 1

        time.sleep(0.5)
        
print(f"Success rate = {nb_success / test_episodes * 100}%")

env.close()
