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

def change_state(state, action, surroundings, grid_size):
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

def get_action(state, q_matrix, epsilon, env):
    if np.random.rand() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(q_matrix[state])
    return action

def update_q_matrix(state, action, reward, new_state, q_matrix, alpha, gamma):
    q_matrix[state, action] = (1 - alpha) * q_matrix[state, action] + \
                              alpha * (reward + gamma * np.max(q_matrix[new_state]))
    return q_matrix

def train_agent(env, episodes, alpha, gamma, epsilon, epsilon_decay, q_matrix, grid_size):
    for i in range(episodes):
        print("Episodio:", i + 1)
        surroundings = env.reset()
        done = False
        state = 0

        while not done:
            action = get_action(state, q_matrix, epsilon, env)
            observation, reward, done, info = env.step(action)
            new_state = change_state(state, action, surroundings, grid_size)
            q_matrix = update_q_matrix(state, action, reward, new_state, q_matrix, alpha, gamma)

            state = new_state
            surroundings = observation
            epsilon = max(epsilon - epsilon_decay, 0)
    return q_matrix

def test_agent(env, q_matrix, test_episodes, grid_size):
    nb_success = 0
    for i in range(test_episodes):
        surroundings = env.reset()
        done = False
        state = 0
        
        while not done:
            action = np.argmax(q_matrix[state])
            observation, reward, done, info = env.step(action)
            env.render()
            new_state = change_state(state, action, surroundings, grid_size)
            state = new_state
            surroundings = observation
            if reward > 0:
                nb_success += 1

            time.sleep(0.5)
    success_rate = nb_success / test_episodes * 100
    return success_rate

# Configuración del entorno y parámetros
env = gym.make("meta-maze-2D-v0", enable_render=True, task_type="ESCAPE")
grid_size = 9
task = MazeTaskSampler(n=grid_size, allow_loops=True, crowd_ratio=0.35)
env.set_task(task)

q_matrix = np.zeros((grid_size * grid_size, 4))

# Parámetros del algoritmo Q-learning
episodes = 2000
alpha = 0.5         # Tasa de aprendizaje
gamma = 0.99         # Factor de descuento
epsilon = 1.0       # Para la exploración y explotación
epsilon_decay = 0.001

# Entrenamiento del agente
q_matrix = train_agent(env, episodes, alpha, gamma, epsilon, epsilon_decay, q_matrix, grid_size)

# Evaluación del agente
test_episodes = 3
success_rate = test_agent(env, q_matrix, test_episodes, grid_size)
print(f"Success rate = {success_rate}%")

env.close()
