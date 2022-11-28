import math
import random
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from sympy import *
import scipy
from scipy import special

import matplotlib.animation as animation
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

rho = 1700  # density
L = 1429  # Latent heat of fusion
MP = 1373


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
    return t + 0.5 * (x ** 2)
    # return t + 0.5 * (x ** 2)


def Tx(x, t, h, i, k, Lx):
    return (T(x + h, t, i, k, Lx) - T(x, t, i, k, Lx)) / h


def Txx(x, t, h, i, k, Lx):
    return (T(x + h, t, i, k, Lx) - 2 * T(x, t, i, k, Lx) + T(x - h, t, i, k, Lx)) / h ** 2


def Tt(x, t, h, i, k, Lx):
    return (T(x, t + h, i, k, Lx) - T(x, t, i, k, Lx)) / h


def heatEq(x, t, h, i, k, Lx):
    if Tt(x[i], t[k], h, i, k, Lx) == alpha(calcThermalConductivity(T(
            x, t, i, k, Lx)), calcSpecificHeat(T(x, t, i, k, Lx)), rho) * Txx \
                (x[i], t[k], h, i, k, Lx):
        return True
    else:
        return False


def T(x, t, i=0, j=0, k=0, L=1, max=5):  # temperature at time t and distance x
    # print(f'x[{i}] = {x[i]}')
    # print(f't[{k}] = {t[k]}')
    if i >= max or k >= max:
        return np.array(func(x, t))
    # if not t[k].all():
    #     print(f'k = {k}')
    #     k += 1
    #     return np.array(func(x, t))
    # elif x[i][j] == L or not x[i][j]:
    #     print(f'i = {i}')
    #     print(f'j = {j}')
    #     i += 1
    #     j += 1
    #     return np.array(func(x, t))
    else:
        i += 1
        k += 1
        j += 1
        return T(np.zeros(1), t, i, j, k) + (T(x, np.zeros(1), i, j, k) - T(
            np.zeros(1), t, i, j, k)) * special.erf(x / (2 * np.sqrt(alpha(
            calcThermalConductivity(MP), calcSpecificHeat(
                MP), rho) * 256)))


# def animate(i):
#     ax.view_init(elev=0., azim=i)
#     return fig,

def rotate(angle):
    ax.view_init(azim=angle)


# plt.rcParams["figure.figsize"] = [7.50, 3.50]
plt.rcParams["figure.autolayout"] = True

temperature = np.linspace(200, 1800, 100)
karray = calcThermalConductivity(temperature)
Carray = calcHeatCap(temperature)
carray = calcSpecificHeat(temperature)
# funcAnswer = func(np.array(5), 3)
# funcAnswer2 = func(1, 7)
# Lx = float(input("Please provide the length of the cylinder (x):\t"))
# Ly = float(input("Please provide the height of the cylinder (y):\t"))
# Tmax = float(input("Please provide the desired time limit:\t"))
x = np.linspace(0, 1, 5)
t = np.linspace(0, 256, 5)
# y = np.linspace(0, 1, 5)
# y = np.linspace(Ly, 0, 5)
# t = np.array([0, .3, .6, 1, 1.5])
# x = np.arange(0, Lmax, 0.5)
# y = np.arange(0, Lmax, 0.5)
# t = np.arange(0, Tmax, 0.5)
x, t = np.meshgrid(x, t)
temp = T(x, t)
# temp = func(x, t)
# x, t = np.meshgrid(x, t)

print(f'x = {x}')
print(f't = {t}')
# t = np.reshape(t, x.shape)
# temperature = T(x, t)
# t, temperature = np.meshgrid(t, temperature)
# myarray = np.empty(1, dtype=np.int)
# myarray.fill(5)

# ansone = T(np.repeat(5, 1), np.repeat(3, 1))
# anstwo = T(np.repeat(1, 1), np.repeat(7, 1))
# R, P = x*np.cos(y), x*np.sin(y)
# R = np.sqrt(x**2+y**2)
# P = np.arctan2(y, x)
# ==================
# hxy = np.hypot(x, y)
# r = np.hypot(hxy, temperature)
# el = np.arctan2(temperature, hxy)
# az = np.arctan2(y, x)
# xy = np.sqrt(x ** 2 + y ** 2 + temperature ** 2)  # sqrt(x² + y²)
#
# x_2 = x ** 2
# y_2 = y ** 2
# z_2 = temperature ** 2
#
# r = np.sqrt(x_2 + y_2 + z_2)  # r = sqrt(x² + y² + z²)
#
# theta = np.arctan2(y, x)
#
# phi = np.arctan2(xy, temperature)

fig = plt.figure()
ax = plt.axes(projection='3d')
# print(shape(R))
# print(shape(P))
# print(theta)
# print(phi)
# print(r)
# print(shape(temperature))

color_dimension = t  # change to desired fourth dimension
minn, maxx = color_dimension.min(), color_dimension.max()
norm = colors.Normalize(minn, maxx)
m = plt.cm.ScalarMappable(norm=norm, cmap='afmhot_r')
m.set_array([])
fcolors = m.set_array(color_dimension)

afmhot_r = cm.get_cmap('afmhot_r', 12)
newcolors = afmhot_r(t)
newcmp = ListedColormap(newcolors, name='afmhot_r')
print("temperatures are:")
print(temp)
# ax.plot_surface(r, theta, phi, cmap="afmhot_r", alpha=.7)
ax.plot_surface(x, t, temp, cmap="afmhot_r", alpha=.7)

# ax = plt.gca()
# ax.hold(True)
# anim = animation.FuncAnimation(fig, animate,
#                                frames=360, interval=20)

# ax.scatter(5, 3, ansone, color='red')
# ax.scatter(1, 7, anstwo, color='red')
rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)

# ax.set_xlabel('r')
# ax.set_ylabel('theta')
# ax.set_zlabel('phi')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('temperature')
fig.canvas.draw()
fig.draw_without_rendering()

plt.show()

rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')
# anim.save('basic_animation.mp4', fps=30, writer='ffmpeg')
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
