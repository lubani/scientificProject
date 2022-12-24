import math
from math import inf, floor

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits import mplot3d
from numpy import ma, NaN

L = 1.0  # length of the bar (m)
LH = 1429  # latent heat of fusion
T = 253  # initial temperature of the bar (K)
T_m = 1373  # melting temperature (K)
rho = 1.7  # density (kg/m^3)

# Finite differences parameters
dx = 0.01  # spatial step size (m)
dt = 0.001  # time step size (s)

# Number of time steps
n_steps = 1000
timer = 10


def calcHeatCap(temp):
    c = 670 + 1000 * ((temp - 250) / 530.6) - 1000 * ((temp - 250) / 498.7) ** 2
    # print(f'c = {c}')
    # print(f'temp = {temp}')
    return c


def calcThermalConductivity(temp):
    return 0.0128 + 5.1e-8 * temp ** 2 - 2.4e-4 * temp + 0.15


def calcSpecificHeat(temp):
    c = -1848.5 + 1047.41 * ma.log(temp)
    return c


def alpha(k, c, rho):
    if np.isnan(k) or np.isnan(c):
        return 1
    return k / (c * rho)


def effectiveSpecificHeat(T, Ql, c):
    if np.logical_or(T < 1373, T > 1653).any():
        return c
    else:
        return c + Ql


def calcTemperature(x, t):
    if t == 0:
        return T
    elif x == 0:
        return T_m
    else:
        return T_m - (T_m - T) * math.erf(x / (2 * math.sqrt(
            alpha(calcThermalConductivity(T),
                  calcSpecificHeat(T), rho) * t)))


def calcEnthalpy(x, t):
    temper = calcTemperature(x, t)
    if temper < T_m:
        return rho * calcSpecificHeat(temper) * (temper - T_m)
    elif np.isclose(temper, T_m).any():
        return LH
    else:
        return rho * calcSpecificHeat(temper) * (temper - T_m) + rho * LH


def calcHeatFlux(x, t, E_interface, T_m, k):
    # Calculate the temperature at the interface using the calcTemperature function
    T_interface = calcTemperature(x, t)

    # Compute the heat flux at the interface
    heat_flux_interface = -k * (T_interface - T_m) / dx

    return heat_flux_interface


def boundary_condition(x, t):
    E = t + (x ** 2) / 2
    if E < 0:
        # Set enthalpy to 0 and temperature to T_m
        E = 0
        T = T_m
    else:
        # Calculate temperature using calcTemperature function
        T = calcTemperature(E, rho)
    return T


# The enthalpy method is a way of solving the Stefan problem, which is a mathematical model for the process of
# solid-liquid phase change (melting or freezing). In the enthalpy method, the enthalpy (a measure of the total
# thermal energy of a substance) is used as a variable to represent the temperature distribution in the system. The
# enthalpy function is a nonlinear version of the heat equation, and it is solved numerically using a finite
# difference scheme.

x_arr = np.linspace(0, L, n_steps)
t_arr = np.linspace(0, timer, n_steps)
print(f'x = {x_arr}, t = {t_arr}')
# x, t = np.meshgrid(x_arr, t_arr)
E_arr = np.empty((n_steps, n_steps))
T_arr = np.empty_like(E_arr)
for j, t_val in enumerate(t_arr):
    for i, x_val in enumerate(x_arr):
        E_arr[i, j] = calcEnthalpy(x_val, t_val)
        T_arr[i, j] = calcTemperature(x_val, t_val)

# Plot the temperature vs. enthalpy values
plt.plot(E_arr, T_arr)
plt.xlabel('Enthalpy (J/kg)')
plt.ylabel('Temperature (K)')
plt.show()
plt.savefig('enthalpy.png')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the surface with x = x, y = t, and z = T_arr
ax.plot_surface(x_arr, t_arr, T_arr, cmap='coolwarm')
plt.xlabel('x (m)')
plt.ylabel('t (s)')

# Function to update the plot at each frame of the animation
def update(num):
    ax.view_init(elev=10., azim=num)


# Create the animation
anim = animation.FuncAnimation(fig, update, frames=np.arange(0, 360, 2),
                               interval=100)

# Show the plot

plt.show()
# Save the animation to a GIF file
anim.save('animation.gif', writer='Pillow', fps=30)
