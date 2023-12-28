# Resources: https://github.com/PaddlePaddle/MetaGym

import gym
import metagym.metamaze
from metagym.metamaze import MazeTaskSampler
import numpy as np
import random
from enum import Enum
import time
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from queue import Queue
import time
import matplotlib.pyplot as plt

# Desactivar todas las advertencias
import warnings
warnings.filterwarnings("ignore")


class Action(Enum):
    W = 0
    E = 1
    S = 2
    N = 3

"""
Watch out with Observation Rotation.
For visual observation:
      N
    X X 0
  W 0 0 0 E
    0 X X
      S
The observation array is:
           W
    [[ 0.  0. -1.]
S   [ -1.  0. -1.]  N
    [ -1.  0.  0.]]
           E

"""

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


def get_action(state, e, surroundings, actions_queue, surroundings_queue):
    # e: epsilon [0-1], high values -> more exploration
    is_wall = True

    if random.random() <=e:
        # Explore
        if actions_queue.qsize() == 5:
            actions = "".join(str(int(elem)) for elem in list(actions_queue.queue))
            surround = "".join(str(int(elem)) for elem in list(surroundings_queue.queue))
            input = scaler.transform(np.array([int(actions, base=4), int(surround)]).reshape(-1, 2))
            action = model.predict(input)[0]
            is_wall = wall_at(surroundings, direction=action)
            
        while is_wall:
            action = env.action_space.sample()
            is_wall = wall_at(surroundings, direction=action)   
    else:
        # Exploit
        max_index = 1
        sorted_indexes = np.argsort(q_matriz[state])
        while is_wall:
            action = sorted_indexes[-max_index]
            max_index +=1
            is_wall = wall_at(surroundings, direction=action)

    if actions_queue.qsize() == 5:
       actions_queue.get()
    actions_queue.put(action)

    return action

def get_reward(reward):
    if (reward < 0):
        reward = 0

    return reward


def create_model(file):
    with open(file, 'rb') as f:
        loaded_array = pickle.load(f)

    previous_experiences = []
    target_actions = []

    for sequence in loaded_array:
        experience_str = "".join(str(int(elem)) for elem in [sequence[0][1], sequence[1][1], sequence[2][1], sequence[3][1], sequence[4][1]])
        surroundings_str = "".join(str(int(elem)) for elem in [sequence[0][0], sequence[1][0], sequence[2][0], sequence[3][0], sequence[4][0]])
        previous_experiences.append([int(experience_str, base=4), int(surroundings_str)])
        target_actions.append(sequence[5][1])

    X = scaler.fit_transform(np.array(previous_experiences).reshape(-1, 2))
    Y = np.array(target_actions)
    X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.2, random_state=42, shuffle=True)
    RFModel = RandomForestClassifier(class_weight="balanced", n_estimators=10)
    RFModel.fit(X_train, y_train)
    y_pred = RFModel.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Model accuracy:", accuracy)

    return RFModel


env = gym.make("meta-maze-2D-v0", enable_render=True, task_type="ESCAPE") 
grid_size = 9
task = MazeTaskSampler(n=grid_size, allow_loops=True, crowd_ratio=0.35, goal_reward=100)
env.set_task(task)

q_matriz = np.zeros((grid_size*grid_size, 4))

scaler = MinMaxScaler()
file = "all_experiences_RF.pickle"
model = create_model(file)

episodes = 1000
factor_desc = 0.9

# Para la exlporación y explotación
epsilon = 1.0       
#epsilon_decay = 0.005

inicio = time.time()
rewards = []
for i in range(episodes):
    print("Episodio:",i+1)
    surroundings = env.reset()
    done = False
    state = 0
    actions_queue = Queue(maxsize=5)
    surroundings_queue = Queue(maxsize=5)

    while not done:
        action = get_action(state, epsilon, surroundings, actions_queue, surroundings_queue)
        if surroundings_queue.qsize() == 5:
            surroundings_queue.get()
        flat_matrix = surroundings.flatten()
        mapped_matrix = np.where(flat_matrix == -1, 2, flat_matrix)
        state_str = ''.join(str(int(e)) for e in mapped_matrix)
        surroundings_queue.put(int(state_str, base=3))

        observation, reward, done, info = env.step(action)
        #env.render()

        new_state = get_state(state, action, surroundings)
        reward = get_reward(reward)

        
        q_matriz[state,action] = reward+(factor_desc*np.max(q_matriz[new_state]))
        state = new_state
        surroundings = observation

    rewards.append(reward)
    
    #epsilon = max(epsilon - epsilon_decay, 0)
    epsilon = 1 - i/(episodes)

fin = time.time()

# Evaluation
test_episodes = 1
nb_success = 0
for i in range(test_episodes):
    surroundings = env.reset()
    done = False
    state = 0
    
    while not done:
        action = np.argmax(q_matriz[state])
        if wall_at(surroundings, direction=action):
            break
        observation, reward, done, info = env.step(action)
        env.render()
        new_state = get_state(state, action, surroundings)
        state = new_state
        surroundings = observation
        if(reward > 0):
            nb_success += 1

        time.sleep(0.5)
        

plt.plot(range(episodes), rewards, linestyle='-', color='b', label='Rewards')
plt.title('Rewards por Episodio')
plt.xlabel('Número de Episodio')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.show()

print("Success?", end=" ")
if nb_success == test_episodes:
    print ("YES")
else: 
    print ("NO")

print("Training time:", round(fin-inicio, 3), "s")


env.close()