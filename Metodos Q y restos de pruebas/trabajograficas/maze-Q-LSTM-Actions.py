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

import torch
from torch import nn
from torch import optim

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

################################################### LSTM Classes #################################################################################
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_dim, n_layers,sigma, dropout):

        # input size -> Dimension of the input signal
        # outpusize -> Dimension of the output signal
        # hidden_dim -> Dimension of the rnn state
        # n_layers -> If >1, we are using a stacked RNN

        super().__init__()

        self.hidden_dim = hidden_dim

        self.input_size = input_size

        self.output_size = output_size

        self.sigma = torch.Tensor(np.array(sigma))

        # define an RNN with specified parameters
        # batch_first=True means that the first dimension of the input will be the batch_size
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_dim, num_layers=n_layers, dropout=dropout, batch_first=True)

        self.relu = nn.ReLU()

        # One linear layer to estimate mean
        self.linear1 = nn.Linear(hidden_dim, 16) # YOUR CODE HERE
        self.linear2 =  nn.Linear(16, 4) # YOUR CODE HERE

        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x, h0=None):

        '''
        About the shape of the different tensors ...:

        - Input signal x has shape (batch_size, seq_length, input_size)
        - The initialization of the RNN hidden state h0 has shape (n_layers, batch_size, hidden_dim).
          If None value is used, internally it is initialized to zeros.
        - The RNN output (batch_size, seq_length, hidden_size). This output is the RNN state along time

        '''
        seq_length = x.size(1) # T

        # get RNN outputs
        # r_out is the sequence of states
        # hidden is just the last state (we will use it for forecasting)
        r_out, hidden = self.lstm(x, h0)

        # shape r_out to be (seq_length, hidden_dim) #UNDERSTANDING THIS POINT IS IMPORTANT!!
        r_out = r_out.reshape(-1, self.hidden_dim)

        r_out = self.linear1(r_out)
        r_out = self.relu(r_out)
        sample = self.linear2(r_out)

        return hidden, sample


class RNN_extended(LSTM):

    #Your code here

    def __init__(self, num_data_train, num_iter, sequence_length,
                 input_size, output_size, hidden_dim, n_layers, sigma, dropout, lr=0.001):

        super().__init__(input_size, output_size, hidden_dim, n_layers, sigma, dropout)

        self.hidden_dim = hidden_dim

        self.sequence_length = sequence_length

        self.num_layers = n_layers

        self.lr = lr #Learning Rate

        self.num_train = num_data_train #Number of training signals

        self.optim = optim.Adam(self.parameters(), self.lr)

        self.num_iter = num_iter

        self.criterion = torch.nn.CrossEntropyLoss()#YOUR CODE HERE

        # A list to store the loss evolution along training

        self.loss_during_training = []
        self.valid_loss_during_training = []


    def trainloop(self, x_train, y_train, x_val, y_val):

        # SGD Loop

        for e in range(int(self.num_iter)):

            self.optim.zero_grad()

            x = torch.Tensor(x_train).view([-1,1])

            y = torch.Tensor(y_train).view([-1]).to(torch.long)

            hid,sample = self.forward(x)

            loss = self.criterion(sample,y) #YOUR CODE HERE

            loss.backward()

            # This code helps to avoid vanishing exploiting gradients in RNNs
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(self.parameters(), 2.0)

            self.optim.step()

            self.loss_during_training.append(loss.item())

            with torch.no_grad():

                  running_loss = 0.

                  for i, x in enumerate(x_val):

                      x = torch.Tensor(x).view([-1,1])

                      y = torch.Tensor(y_val[i]).view([-1]).to(torch.long)

                      hid,sample = self.forward(x)

                      loss = self.criterion(sample,y)

                      running_loss += loss.item()

                  self.valid_loss_during_training.append(running_loss/len(x_val))

            if(e % 50 == 0):
              print('\nTrain Epoch: {} -> Training Loss: {:.6f}'.format(e,self.loss_during_training[-1]))
              print('Train Epoch: {} -> Validation Loss: {:.6f}'.format(e,self.valid_loss_during_training[-1]))



    def eval_performance(self, x_test, y_test):
        loss = 0
        accuracy = 0

        # Turn off gradients for validation, saves memory and computations
        with torch.no_grad():

            for i,x in enumerate(x_test):
                x = torch.Tensor(x).view([-1,1])
                y = torch.Tensor(y_test[i]).view([-1]).to(torch.long)

                hid, sample= self.forward(x)
                equals = (torch.argmax(sample, dim=1) == y)
                accuracy += torch.mean(equals.type(torch.FloatTensor))

            return accuracy/len(x_test)

##############################################################################################################################


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


def get_action(state, e, surroundings, actions_queue):
    # e: epsilon [0-1], high values -> more exploration
    is_wall = True

    if random.random() <=e:
        # Explore
        if actions_queue.qsize() == 5:
            actions = [elem for elem in list(actions_queue.queue)]
            input = torch.Tensor(actions).view([-1,1])
            _, sample = model.forward(input)
            action = int(torch.argmax(sample, dim=1)[4])
            is_wall = wall_at(surroundings, direction=action)
            if is_wall:
                while is_wall:
                    action = env.action_space.sample()
                    is_wall = wall_at(surroundings, direction=action)   
        else:
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


def load_model(file):
    with open(file, 'rb') as f:
        lstm_model = pickle.load(f)

    return lstm_model


env = gym.make("meta-maze-2D-v0", enable_render=True, task_type="ESCAPE") 
grid_size = 21
task = MazeTaskSampler(n=grid_size, allow_loops=True, crowd_ratio=0.35, goal_reward=100)
env.set_task(task)

q_matriz = np.zeros((grid_size*grid_size, 4))

scaler = MinMaxScaler()
file = "lstm_model_actions.pkl"
model = load_model(file)

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

    while not done:
        action = get_action(state, epsilon, surroundings, actions_queue)
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