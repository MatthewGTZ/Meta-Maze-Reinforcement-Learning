import gym
import metagym.metamaze
from metagym.metamaze import MazeTaskSampler
import numpy as np
def run(episodes,render = True):
    maze_env = gym.make("meta-maze-2D-v0", enable_render=True if render else None, task_type="ESCAPE") # Running a 2D Maze with ESCAPE task

    '''
    #Sample a task by specifying the configurations
    task = MazeTaskSampler(
        n            = 15,  # Number of cells = n*n
        allow_loops  = False,  # Whether loops are allowed
        crowd_ratio  = 0.40,   # Specifying how crowded is the wall in the region, only valid when loops are allowed. E.g. crowd_ratio=0 means no wall in the maze (except the boundary)
        cell_size    = 2.0, # specifying the size of each cell, only valid for 3D mazes
        wall_height  = 3.2, # specifying the height of the wall, only valid for 3D mazes
        agent_height = 1.6, # specifying the height of the agent, only valid for 3D mazes
        view_grid    = 1, # specifiying the observation region for the agent, only valid for 2D mazes
        step_reward  = -0.01, # specifying punishment in each step in ESCAPE mode, also the reduction of life in each step in SURVIVAL mode
        goal_reward  = 1.0, # specifying reward of reaching the goal, only valid in ESCAPE mode
        initial_life = 1.0, # specifying the initial life of the agent, only valid in SURVIVAL mode
        max_life     = 2.0, # specifying the maximum life of the agent, acquiring food beyond max_life will not lead to growth in life. Only valid in SURVIVAL mode
        food_density = 0.01,# specifying the density of food spot in the maze, only valid in SURVIVAL mode
        food_interval= 100, # specifying the food refreshing periodicity, only valid in SURVIVAL mode
        )
    '''



    n = 15

    task = MazeTaskSampler(n=n, allow_loops=True, crowd_ratio=0.35)

    #Set the task configuration to the meta environment

    #CREO TABLA PARA LOS ESTADOS Y LAS ACCCIONES 
    #q = np.zeros((maze_env.observation_space,maze_env.action_space))

    q = np.zeros((n*n,4))
    



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

        while not done:
            maze_env.render()
            if rng.random()<epsilon:
                action = maze_env.action_space.sample() 
            else:
                #action = np.argmax(q[state,:])
                print("maybe")

            new_state,reward,done,info = maze_env.step(action)
            
            q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
            )
            state = new_state
        
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001

        if reward == 1:
            rewards_per_episode[i] = 1

        #while True:
            
            #observation, reward, done, info = maze_env.step(action)
                
            
        maze_env.close()

    """
            if done:
                maze_env.reset()

                """
        
if __name__ == '__main__':
    # run(15000)

    run(1000, render=False)