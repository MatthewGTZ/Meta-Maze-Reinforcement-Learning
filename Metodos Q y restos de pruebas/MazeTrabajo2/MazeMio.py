import gym
import numpy as np
from metagym.metamaze import MazeTaskSampler
import time
from enum import Enum
# Asumimos que la observación es una matriz 3x3 donde cada celda puede ser -1, 0, o 1
# Por lo tanto, hay 3^9 posibles estados
num_possible_states = 3 ** 9

class Action(Enum):
    W = 0
    E = 1
    S = 2
    N = 3

def matrix_to_index(matrix):
    # Convertimos la matriz en un número entero único para usar como índice en la matriz Q
    flat_matrix = matrix.flatten()
    mapped_matrix = np.where(flat_matrix == -1, 2, flat_matrix)
    state_str = ''.join(str(int(e)) for e in mapped_matrix)
    return int(state_str, base=3)

if __name__ == '__main__':
    env = gym.make("meta-maze-2D-v0", enable_render=True, task_type="ESCAPE") 
    n = 9  # Este es el tamaño del laberinto
    task = MazeTaskSampler(n=n, allow_loops=True, crowd_ratio=0.35)
    env.set_task(task)

    q_matrix = np.zeros((num_possible_states, 4))

    episodes = 1000
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1
    epsilon_decay = 0.0001
    rng = np.random.default_rng()

    

    for episode in range(episodes):
        done = False
        observation = env.reset()  # Obtenemos la observación inicial
        state = matrix_to_index(observation)  # Convertimos la observación en un índice de estado
        i=0
        print("episodio: ",episode)
        while not done:
            i+=1
            if rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q_matrix[state])
                print("Explotacion: ",i)

            new_observation, reward, done, info = env.step(action)
            print(new_observation,reward,done, " Iteracion: ", i)
            new_state = matrix_to_index(new_observation)
            print(new_state)
            # Actualizamos la matriz Q
            q_matrix[state, action] = q_matrix[state, action] + \
                                      learning_rate * (reward + discount_factor * np.max(q_matrix[new_state]) - q_matrix[state, action])

            state = new_state  # Actualizamos el estado
            epsilon = max(epsilon - epsilon_decay, 0)  # Decrementamos epsilon

    # Evaluación después del entrenamiento
    test_episodes = 30
    nb_success = 0
    for i in range(test_episodes):
        observation = env.reset()
        state = matrix_to_index(observation)
        done = False

        while not done:
            action = np.argmax(q_matrix[state])
            observation, reward, done, info = env.step(action)
            env.render()
            state = matrix_to_index(observation)  # Actualizamos el estado basado en la nueva observación

            if reward > 0:
                nb_success += 1
            time.sleep(0.5)

    success_rate = (nb_success / test_episodes) * 100
    print(f"Success rate = {success_rate}%")

    env.close()