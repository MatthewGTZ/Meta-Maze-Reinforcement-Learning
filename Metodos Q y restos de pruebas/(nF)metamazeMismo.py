import gym
import pickle
import metagym.metamaze
from metagym.metamaze import MazeTaskSampler

def test_2d_maze(max_iteration, task_type):
    print("Testing 2D Maze with task type:", task_type)
    maze_env = gym.make("meta-maze-2D-v0", max_steps=200, enable_render=True, view_grid=1, task_type=task_type)

    task_filename = 'maze_task.pkl'
    try:
        # Intenta cargar el mapa desde un archivo
        with open(task_filename, 'rb') as f:
            task = pickle.load(f)
    except FileNotFoundError:
        # Si el archivo no existe, crea un nuevo mapa y lo guarda
        n = 9
        task = MazeTaskSampler(n=n, step_reward=-0.01, goal_reward=1.0)
        with open(task_filename, 'wb') as f:
            pickle.dump(task, f)

    maze_env.set_task(task)
    iteration = 0

    while iteration < max_iteration:
        iteration += 1
        maze_env.reset()

        done = False
        sum_reward = 0
        while not done:
            maze_env.render()
            state, reward, done, _ = maze_env.step(maze_env.action_space.sample())
            sum_reward += reward

        # La dificultad ya no aumentará aquí, ya que quieres mantener el mismo mapa
        print("Get score %f" % sum_reward)

if __name__ == "__main__":
    test_2d_maze(100, "ESCAPE")