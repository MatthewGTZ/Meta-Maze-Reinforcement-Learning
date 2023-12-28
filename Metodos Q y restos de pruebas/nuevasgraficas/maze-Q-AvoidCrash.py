# Resources: https://github.com/PaddlePaddle/MetaGym

import gym
import metagym.metamaze
from metagym.metamaze import MazeTaskSampler
import numpy as np
import random
from enum import Enum
import time
import matplotlib.pyplot as plt

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


def get_action(state, e, surroundings):
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
        sorted_indexes = np.argsort(q_matriz[state])
        while is_wall:
            action = sorted_indexes[-max_index]
            max_index +=1
            is_wall = wall_at(surroundings, direction=action)

    return action

def get_reward(reward):
    if (reward < 0):
        reward = 0

    return reward

env = gym.make("meta-maze-2D-v0", enable_render=True, task_type="ESCAPE") 
grid_size = 21
task = MazeTaskSampler(n=grid_size, allow_loops=True, crowd_ratio=0.35, goal_reward=100)
env.set_task(task)

q_matriz = np.zeros((grid_size*grid_size, 4))

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

    while not done:
        action = get_action(state, epsilon, surroundings)
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
    #print(epsilon)

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