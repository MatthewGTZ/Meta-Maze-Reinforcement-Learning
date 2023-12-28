import gym
import sys
import metagym.metamaze
from metagym.metamaze import MazeTaskSampler
import numpy as np
from enum import Enum

class Action(Enum):
    W = 0
    E = 1
    S = 2
    N = 3



if __name__=='__main__':
    maze_env = gym.make("meta-maze-2D-v0", max_steps=1500, view_grid=1, task_type="ESCAPE")


    n = 15
    #Esto lo ejecutas una sola vez 
    task = MazeTaskSampler(n=n, allow_loops=True, crowd_ratio=0.35)
    # Con pickle u otro serialiyador, te lo guardas el task para otra vez
    # task.save

    # Cuando quieras usar tu mapa, lo cargas
    # task.load

    # Pickle sirve para cualquier cosa, asi que puedes serializar Q si quieres tambien!
    q = np.zeros((n*n,4))
    #print(q)
    episodes = 1000
    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount rate. Near 0: more weight/reward placed on immediate state. Near 1: more on future state.
    epsilon = 1         # 1 = 100% random actions
    epsilon_decay_rate = 0.0001        # epsilon decay rate. 1/0.0001 = 10,000
    rng = np.random.default_rng()   # random number generator

    rewards_per_episode = np.zeros(episodes)
    for i in range(episodes):

        #Start the task
        done = False
        maze_env.set_task(task)
        state = maze_env.reset()
        #print(state)

        #maze_env.set_task(task)
        while True:
            maze_env.reset()
            done=False
            sum_reward = 0
            while not done:
                maze_env.render()

                if rng.random()<epsilon:
                    action = maze_env.action_space.sample()
                     
                else:
                    action = np.argmax(q[state,:])
                new_state, reward, done, info = maze_env.step(action)
                print(new_state,reward,done,info)
                
                q[state,action] = q[state,action] + learning_rate_a * (reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action])
                
                state = new_state

                epsilon = max(epsilon - epsilon_decay_rate, 0)
                if(epsilon==0):
                    learning_rate_a = 0.0001
                if reward == 1:
                    rewards_per_episode[i] = 1


                 
                



                #print(action)
                #sum_reward += reward
                
            #if(not maze_env.key_done):
                #print("Episode is over! You got %.1f score."%sum_reward)
                #maze_env.save_trajectory("trajectory_%dx%d.png"%(n, n))
                #if(sum_reward > 0.0):
                #   n += 2 # gradually increase the difficulty
                #  print("Increase the difficulty, n = %d"%n)
                #task = MazeTaskSampler(n=n, allow_loops=True, crowd_ratio=0.35)
                #maze_env.set_task(task)
            #else:
                #break