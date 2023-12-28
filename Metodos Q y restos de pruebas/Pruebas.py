import gymnasium as gym

#Creacion del entorno lunar launcher para la version 2 y render human
env = gym.make("CarRacing-v2", domain_randomize=True, render_mode="human")

#Reset para obtener el estado inicial
observation, info = env.reset()
print(observation)

for _ in range(1000):
#Escoge una accion al azar. Aqui deberia estar la funcion del agente que escoge la accion a ejecutar
    action = env.action_space.sample()

    #Step para realizar los cambios en el entorno
    observation, reward, terminated, truncated, info = env.step(action)
    #Reset del entorno si se acaba el episodio
    if terminated or truncated:
        observation, info = env.reset()


#Cerrar la instancia de pygame
env.close()