import math
import sys

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
import pde

rho = 1700  # density
L = 1429  # Latent heat of fusion
MP = 1373
Lmax = 50
Tmax = 60

def effectiveSpecificHeat(T, Ql, c):
    if T < 1373 or T > 1653:
        return c
    else:
        return c + Ql


def calcHeatCap(temp):
    return 670 + 1e3 * ((temp - 250) / 530.6) - 1e3 * ((temp - 250) / 498.7) ** 2


def calcThermalConductivity(temp):
    c1 = 1.281 * 1e-2
    c2 = 4.431 * 1e-10
    return c1 + c2 * temp ** 3


def calcSpecificHeat(temp):
    return -1848.5 + 1047.41 * np.log(temp)


def alpha(k, c, rho):
    return k / (c * rho)


def func(x, t):
    return t + 0.5 * x ** 2


def T(x, t):  # temperature at time t and distance x
    if np.any(t) or (x == Lmax) or (np.any(x)):
        return func(x, t)
    else:
        return T(0, t) + (T(x, 0) - T(0, t)) * \
                        math.erf(x / (2 * math.sqrt(alpha(
                            calcThermalConductivity(T(x, t)), calcSpecificHeat(
                                T(x, t)), rho) * t)))


# plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

temp = np.linspace(200, 1800, 100)
karray = calcThermalConductivity(temp)
Carray = calcHeatCap(temp)
carray = calcSpecificHeat(temp)
funcAnswer = func(5, 3)
funcAnswer2 = func(1, 7)

x = np.arange(-20, Lmax, 0.5)
t = np.arange(0, Tmax, 0.5)
# temperature = T(x, t)
x, t = np.meshgrid(x, t)
f = func(x, t)
temperature = T(x, t)

fig = plt.figure()
ax = plt.axes(projection='3d')
print(shape(x))
print(shape(t))
print(shape(f))
ax.plot_surface(x, t, temperature)
# ax.scatter(5, 3, funcAnswer, edgecolors='black')
# ax.scatter(1, 7, funcAnswer2, edgecolors='black')
plt.show()

# plt.plot(temp, Carray, color='red', label='Heat Capacity')
#
# plt.plot(temp, carray, color='green', label='Specific Heat')
# plt.xlabel('Temperature')
# plt.legend()
# plt.show()
#
# plt.plot(temp, karray, color='blue', label='Thermal Conductivity')
# plt.legend()
# plt.show()
# a = alpha(karray, carray, rho)
# plt.plot(temp, a, label='Diffusivity')
# plt.xlabel('Temperature')
# plt.legend()
# plt.show()
# k = calcThermalConductivity(L)
# c = calcSpecificHeat(L)

# grid = pde.UnitGrid([64, 64])  # generate grid
# state = pde.ScalarField.random_uniform(grid)  # generate initial condition
#
# eq = pde.DiffusionPDE(diffusivity=float(alpha(k, c, rho)))  # define the pde
# result = eq.solve(state, t_range=50)  # solve the pde
# result.plot()
# plt.show()
