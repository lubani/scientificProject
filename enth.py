from math import inf

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
# Import the mplot3d library from matplotlib
from mpl_toolkits import mplot3d
from numpy import ma


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
    return -1848.5 + 1047.41 * ma.log(temp)


def alpha(k, c, rho):
    return k / (c * rho)


# Set the time for which you want to plot the temperature distribution
time = 0.5  # time in seconds

# Stefan problem parameters
L = 1.0  # length of the bar (m)
LH = 1429  # latent heat of fusion
T = 253  # initial temperature of the bar (K)
T_m = 1429  # melting temperature (K)

rho = 1700.0  # density (kg/m^3)

c = calcSpecificHeat(T)  # specific heat capacity (J/kgK)
k = calcThermalConductivity(T)  # thermal conductivity (W/mK)
# Finite differences parameters
dx = 0.01  # spatial step size (m)
dt = 0.001  # time step size (s)

# Number of time steps
n_steps = 10000
# Initialize the temperature, solid-liquid interface, and heat flux arrays
# We assume that the temperature is uniform throughout the PCM at start.
T_arr = np.zeros((n_steps, int(L / dx)))
T_arr[0, :] = T
T_arr[0, 0] = T_m
s_arr = np.zeros(n_steps)
s_arr[:] = L

heat_flux_arr = np.zeros((n_steps, int(L / dx)))

# Set the initial solid-liquid interface and the width of the mushy zone
mushy_zone_width = 0.01  # width of the mushy zone in meters

for i in range(1, n_steps):
    if s_arr[i - 1] == inf:
        s = len(T_arr[i - 1]) - 1
    else:
        s = int(s_arr[i - 1] / dx)
    # # Check if the solid-liquid interface is within the valid range for the T_arr array
    # if s < 0:
    #     s = 0
    # elif s >= len(T_arr[i - 1]):
    #     s = len(T_arr[i - 1]) - 1

    # Update the heat flux at each point in the bar
    heat_flux_arr[i, 1:-1] = -k * (T_arr[i, 2:] - T_arr[i, :-2]) / (2 * dx)
    #  update the heat flux at the left end
    heat_flux_arr[i, 0] = k * (T_arr[i, 0] - T) / dx
    # update the heat flux at the right end
    heat_flux_arr[i, -1] = k * (T_m - T_arr[i, -1]) / dx
    print(f'heat flux array = {heat_flux_arr}')
    # Calculate the enthalpy at the solid-liquid interface
    if s < len(T_arr[i - 1]):
        H = heat_flux_arr[i - 1, s] / (T_m - T_arr[i - 1, s])
        print("T_arr[i - 1, s] = ", T_arr[i - 1, s])
    else:
        H = heat_flux_arr[i - 1, -1] / (T_m - T_arr[i - 1, -1])
        print("T_arr[i - 1, s] = ", T_arr[i - 1, -1])
    print(f'H = {H}')
    # Update the temperature at each point in the bar
    T_arr[i, 1:-1] = T_arr[i - 1, 1:-1] + dt * k * (
            T_arr[i - 1, 2:] - 2 * T_arr[i - 1, 1:-1] + T_arr[i - 1, :-2]) / dx ** 2
    # Set the boundary conditions
    T_arr[:, 0] = T_m  # left end of the bar
    T_arr[:, -1] = T  # right end of the bar
    # Update the temperature in the mushy zone
    width = int(mushy_zone_width / dx)
    if s - width < 0:
        s = width
        T_arr[i, 0:s + width] = np.linspace(T_arr[i, 0], T_arr[i, s + width], s + width)
    elif s + width >= len(T_arr[i - 1]):
        s = len(T_arr[i - 1]) - width - 1
        T_arr[i, s - width:len(T_arr[i])] = np.linspace(T_arr[i, s - width], T_arr[i, -1], len(T_arr[i]) - s + width)
    else:
        T_arr[i, s - width:s + width] = np.linspace(T_arr[i, s - width], T_arr[i, s + width], 2 * width + 1)
    # Update the position of the solid-liquid interface
    s_arr[i] = s_arr[i - 1] + dt * width

ts = 1000
# Generate x and y coordinates for the points on the surface
x = np.linspace(0, L, int(L / dx))
t = np.linspace(0, ts, n_steps)
X, times = np.meshgrid(x, t)

# Generate a 3D surface plot using the temperature values from the T_arr array
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_surface(X, times, T_arr, cmap='coolwarm')

ax.set_xlabel('x (m)')
ax.set_ylabel('t (s)')
ax.set_zlabel('T (K)')
plt.show()


# Code to generate the initial plot, including creating the figure and axes object
def plot_function():
    # Create the initial 3D surface plot using the x, t, and T_arr arrays
    ax.plot_surface(X, times, T_arr, cmap='coolwarm')
    ax.set_xlabel('x (m)')
    ax.set_ylabel('t (s)')
    ax.set_zlabel('T (K)')

    # Return the updated plot
    return ax


def update_function(i):
    # Rotate the plot around the y-axis by a given angle
    ax.view_init(elev=10., azim=i * 4)

    # Return the updated plot
    return ax


plt.show()

# Calculate the temperature distribution at the given time
# x = np.linspace(0, L, int(L / dx))  # position array in meters
T_dist = T_arr[int(time / dt), :]  # temperature distribution array in Kelvins

# Plot the temperature distribution
plt.plot(x, T_dist)
plt.xlabel('x (m)')
plt.ylabel('T (K)')
plt.show()

print("s_arr = ", s_arr)
print("T_arr = ", T_arr)

# Code to generate the initial plot, including creating the figure and axes object
# Use FuncAnimation to create the animation, passing it the Animation object, figure, and update function
animation = animation.FuncAnimation(fig, update_function, init_func=plot_function, frames=100)
# Save the animation to a file
animation.save('stefan_problem1.gif', fps=20)
