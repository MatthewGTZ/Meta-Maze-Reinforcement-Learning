
from torch import nn
from torch import optim
import numpy as np
from enum import Enum
import time
import pickle
import matplotlib.pyplot as plt

with open("/home/gdanietz/Schreibtisch/Universidad/Programacion/Segundo semicuatrimestre/Aprendizaje por Refuerzo/TrabajoFInal/MazeTrabajo/modelos/Experiences-9-30-1000.pickle", 'rb') as f:
    loaded_array = pickle.load(f)

Y = loaded_array[:,1:]
X = loaded_array[:,:-1]

print(Y)


X = [[int(sublist[1]) for sublist in row] for row in X]
Y = [[int(sublist[1]) for sublist in row] for row in Y]

#print(X)
