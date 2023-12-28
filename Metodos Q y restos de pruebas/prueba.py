import gym
import numpy as np
from metagym.metamaze import MazeTaskSampler
import time

def matrix_to_index(matrix):
    """
    Convert the 3x3 matrix observation into a unique state index.
    Map -1 to 2, 0 to 0, and 1 to 1, and then convert the result to a base-3 integer.
    """
    flat_matrix = matrix.flatten()
    # Mapeo: -1 -> 2, 0 -> 0, 1 -> 1
    mapped_matrix = np.where(flat_matrix == -1, 2, flat_matrix)
    state_str = ''.join(str(int(e)) for e in mapped_matrix)
    return int(state_str, base=3)
def evaluate_agent(env, q_table, test_episodes):
    nb_success = 0
    for i in range(test_episodes):
        observation = env.reset()
        state = matrix_to_index(observation)
        done = False

        while not done:
            action = np.argmax(q_table[state])
            observation, reward, done, info = env.step(action)
            env.render()  # Comenta esta línea si la representación visual no es necesaria
            new_state = matrix_to_index(observation)
            state = new_state

            if reward > 0:
                nb_success += 1

            time.sleep(0.5)  # Comenta esta línea si no quieres la pausa entre pasos

    success_rate = nb_success / test_episodes * 100
    return success_rate
if __name__ == '__main__':
    maze_env = gym.make("meta-maze-2D-v0", max_steps=1500, view_grid=1, task_type="ESCAPE")
    n = 9  # Assuming n is the width and height of the maze
    task = MazeTaskSampler(n=n, allow_loops=True, crowd_ratio=0.35)
    maze_env.set_task(task)

    # The state size will depend on the number of unique states we can have.
    # For a 3x3 matrix with each cell being -1 or 0, this is base-3 conversion.
    num_states = 3 ** (3 * 3)  
    q_table = np.zeros((num_states, 4))

    episodes = 1000
    learning_rate = 0.9
    discount_factor = 0.9
    epsilon = 1.0
    epsilon_decay_rate = 0.0001
    rng = np.random.default_rng()

    # ... [Resto del código anterior] ...

    for episode in range(episodes):
        done = False
        observation = maze_env.reset()
        state = matrix_to_index(observation)  # Codifica la observación inicial

        while not done:
            # Epsilon-greedy policy para la selección de acciones
            if rng.random() < epsilon:
                action = maze_env.action_space.sample()  # Explorar: acción aleatoria
            else:
                action = np.argmax(q_table[state])  # Explotar: mejor acción basada en q_table

            new_observation, reward, done, info = maze_env.step(action)
            new_state = matrix_to_index(new_observation)  # Codifica la nueva observación

            # Actualización de Q-table usando la regla de Q-learning
            q_table[state, action] = q_table[state, action] + learning_rate * (
                    reward + discount_factor * np.max(q_table[new_state]) - q_table[state, action])

            state = new_state
            epsilon = max(epsilon - epsilon_decay_rate, 0)  # Decaimiento de epsilon

        print(f"Episode: {episode}, State: {state}")
        # Evaluación del agente
    test_episodes = 3
    success_rate = evaluate_agent(maze_env, q_table, test_episodes)
    print(f"Success rate = {success_rate}%")
    maze_env.close()

