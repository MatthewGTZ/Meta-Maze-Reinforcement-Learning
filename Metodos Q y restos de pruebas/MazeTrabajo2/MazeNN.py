import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from enum import Enum
from metagym.metamaze import MazeTaskSampler
import random
# Tus definiciones de Action, get_surrounding_value, wall_at, get_state, etc.

class Action(Enum):
    W = 0
    E = 1
    S = 2
    N = 3

# ... otras funciones auxiliares que ya has definido ...

def preprocess_data(experiences, sequence_length):
    sequences = []
    for i in range(len(experiences) - sequence_length):
        sequence = experiences[i:i + sequence_length]
        observations = [obs for obs, act in sequence]
        actions = [act for obs, act in sequence]
        sequences.append((observations, actions))
    return sequences

class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, hidden = self.rnn(x)
        x = self.fc(x)
        return x

# --- Aux Functions ---

def get_surrounding_value(surroundings,direction):
    if direction == Action.N.value:
        return surroundings[1][2]
    if direction == Action.S.value:
        return surroundings[1][0]
    if direction == Action.E.value:
        return surroundings[2][1]
    if direction == Action.W.value:
        return surroundings[0][1]

def wall_at(surroundings, direction):
    # Direction = Action
    return get_surrounding_value(surroundings, direction) == -1.0

def get_state(state, action, surroundings):
    if wall_at(surroundings,direction=action):
        return state
        
    if(action == Action.N.value):
        if(state < grid_size):
            return state+(grid_size*(grid_size-1))
        else:
            return state-grid_size

    elif(action == Action.S.value):
        if(state >= grid_size*(grid_size-1)):
            return state-(grid_size*(grid_size-1))
        else:
            return state+grid_size

    elif(action == Action.W.value):
        if(state == 0):
            return (grid_size*grid_size)-1
        else:
            return state-1

    elif(action == Action.E.value):
        if(state == (grid_size*grid_size)-1):
            return 0
        else:
            return state+1

    return state

def get_action(env, Q, state, e, surroundings):
    # e: epsilon [0-1], high values -> more exploration
    is_wall = True

    if random.random() <=e:
        # Explore
        while is_wall:
            action = env.action_space.sample()
            is_wall = wall_at(surroundings, direction=action)
    else:
        # Exploit
        max_index = 1
        sorted_indexes = np.argsort(Q[state])
        while is_wall:
            action = sorted_indexes[-max_index]
            max_index +=1
            is_wall = wall_at(surroundings, direction=action)

    return action

def get_reward(reward, state, action, surroundings):
    if (reward < 0):
        reward = 0
    return reward

def get_policy(Q):
    """
    returns a python list of length "states" with the action with higher Q value
    """
    pi = []
    for values in Q:
        pi.append(np.argmax(values))
    return pi

def is_winning_policy(pi, env):
    """
    Returns true if the policy always wins for such task.
    Remember that this enviroment is deterministic.
    """
    surroundings = env.reset()
    done = False
    state = 0
    
    while not done:
        action = pi[state]
        observation, reward, done, info = env.step(action)
        new_state = get_state(state, action, surroundings)
        state = new_state
        surroundings = observation
        if(reward > 0):
            return True
    
    return False

def matrix_to_index(matrix):
    """
    Convertimos la matriz en un número entero único para usar como índice en la matriz Q
    Transforma las observaciones 3x3 en un índice entero
    """
    flat_matrix = matrix.flatten()
    mapped_matrix = np.where(flat_matrix == -1, 2, flat_matrix)
    state_str = ''.join(str(int(e)) for e in mapped_matrix)
    return int(state_str, base=3)


# --- Experience gathering methods ---

def gather_experiences(grid_size, seq_length, episodes, n_mazes):
    """
    Play "n_mazes" different mazes of size "grid_size".
    Returns experiences of winning policies
    """
    
    total_sequences = []
    complete_mazes = 0
    n_sequences = 0
    env = gym.make("meta-maze-2D-v0", enable_render=True, task_type="ESCAPE")
    for maze_count in range(n_mazes):
        exp_list = gather_one_exp(env, grid_size, episodes) # Exp_list is an array of (observation,action) of varying length
        if not(exp_list is None):
            sequences = trim_exp_to_window(exp_list, window_size=seq_length) # Obtains fixed sized experiences from the exp_list (using a sliding window)
            total_sequences.append(sequences)
            n_sequences +=len(sequences)
            complete_mazes +=1
        
        print("Progress: \t{}/{}".format(maze_count+1, n_mazes))
    
    print("\nMaze completition: {}%\t({}/{})".format((complete_mazes/n_mazes)*100,complete_mazes,n_mazes))
    print("Nº Sequences: {}".format(n_sequences))
    
    env.close()

    total_sequences = np.concatenate(total_sequences)

    return total_sequences
    
def gather_one_exp(env, grid_size, episodes, allow_loops=True, crowd_ratio=0.35, goal_reward=100):
    """
    Plays in a maze for "episodes". 
    Returns the experiences of the winning policy, or None if it is not a winning policy.
    """
    task = MazeTaskSampler(n=grid_size, allow_loops=allow_loops, crowd_ratio=crowd_ratio, goal_reward=goal_reward)
    env.set_task(task)

    Q = learn_Q(env, episodes, grid_size)
    pi = get_policy(Q)

    if is_winning_policy(pi, env):
        return get_exp_from_policy(pi, env)
    else:
        return None
    
def learn_Q(env, episodes, grid_size, factor_desc=0.9):
    """
    Uses Q-Learning for the given enviroment.
    Returns the Q table
    """
    epsilon = 1 # Cambiar esto no hace nada, se actualiza solo
    q_matriz = np.zeros((grid_size*grid_size, 4))

    for i in range(episodes):
        surroundings = env.reset()
        done = False
        state = 0

        while not done:
            action = get_action(env, q_matriz, state, epsilon, surroundings)
            observation, reward, done, info = env.step(action)
            new_state = get_state(state, action, surroundings)
            reward = get_reward(reward, state, action, surroundings)

            q_matriz[state,action] = reward+(factor_desc*np.max(q_matriz[new_state]))

            state = new_state
            surroundings = observation

        epsilon = 1 - i/(episodes)

    return q_matriz

def get_exp_from_policy(pi, env):
    """
    Uses the policy to win once, while gathering the observations and actions done.
    Returns an array containing pairs <observation_index, action_done>
    """
    exp_pairs = []

    surroundings = env.reset()
    done = False
    state = 0
    
    while not done:
        action = pi[state]

        exp_pairs.append((matrix_to_index(surroundings), action))

        observation, reward, done, info = env.step(action)
        new_state = get_state(state, action, surroundings)
        state = new_state
        surroundings = observation
    
    return exp_pairs

def trim_exp_to_window(exp_list, window_size):
    """
    Experiences obtained from policies' executions are of varying lengths
    so sets of smaller, fixed lengths, need to be extracted from exp_list.

    Returns an array containing several arrays, each of them containing sequences <observation, next_action>
    """

    array = np.array(exp_list)

    shape = (array.shape[0] - window_size + 1, window_size) + array.shape[1:]
    strides = (array.strides[0],) + array.strides
    sequences = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides, writeable=False)
    #print(exp_list)
    return sequences


# Script principal
if __name__ == "__main__":
    grid_size = 9
    episodes = 30
    n_mazes = 10
    sequence_length = 5

    result = gather_experiences(grid_size, sequence_length, episodes, n_mazes)
    processed_data = preprocess_data(result, sequence_length)

    input_size = 10  # Ajustar según tus datos
    hidden_size = 20
    output_size = 4  # Ajustar según tus datos
    rnn = SimpleRNN(input_size, hidden_size, output_size)

    optimizer = torch.optim.Adam(rnn.parameters(), lr=0.001)
    loss_function = nn.CrossEntropyLoss()

    num_epochs = 10  # Ajustar según sea necesario
    for epoch in range(num_epochs):
        for sequence in processed_data:
            optimizer.zero_grad()
            observations, actions = sequence
            observations_tensor = torch.FloatTensor(observations)
            actions_tensor = torch.LongTensor(actions)

            output = rnn(observations_tensor)
            loss = loss_function(output, actions_tensor)

            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch} Loss: {loss.item()}")
