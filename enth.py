import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# Set the time for which you want to plot the temperature distribution
time = 0.5  # time in seconds

# Stefan problem parameters
L = 1.0  # length of the bar (m)
T = 1600.0  # temperature of the bar (K)
T_m = 1400.0  # melting temperature (K)
k = 0.03  # thermal conductivity (W/mK)
rho = 1600.0  # density (kg/m^3)
c = 800.0  # specific heat capacity (J/kgK)
posx = 0.5
posy = 0.7
# Finite differences parameters
dx = 0.1  # spatial step size (m)
dt = 0.01  # time step size (s)

# Number of time steps
n_steps = 1000

# Initialize the temperature and solid-liquid interface arrays
T_arr = T * np.ones((n_steps, int(L / dx)))
s_arr = np.zeros(n_steps)

# Set the initial solid-liquid interface
s_arr[0] = 0.0

# Solve the Stefan problem using the enthalpy method and the finite differences method
for i in range(1, n_steps):
    # Calculate the enthalpy at the solid-liquid interface
    s = s_arr[i - 1] / dx
    if s >= L:
        s = L - dx
    H = rho * c * (T_arr[i - 1, int(s)] - T_m)

    # Update the temperature at each point in the bar
    T_arr[i, 1:-1] = T_arr[i - 1, 1:-1] + dt * k * (
            T_arr[i - 1, 2:] - 2 * T_arr[i - 1, 1:-1] + T_arr[i - 1, :-2]) / dx ** 2

    # Update the solid-liquid interface position
    s_arr[i] = s_arr[i - 1] + dt * H / (rho * c)

    # Update the temperature at the solid-liquid interface
    s = s_arr[i] / dx
    if s >= L:
        s = L - dx
    T_arr[i, int(s)] = T_m

# Plot the temperature distribution over time as a surface plot
x = np.linspace(0, L, int(L / dx))
t = np.linspace(0, n_steps * dt, n_steps)
X, T = np.meshgrid(x, t)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, T_arr, cmap='coolwarm')
ax.set_xlabel('x (m)')
ax.set_ylabel('t (s)')
ax.set_zlabel('T (K)')


# Code to generate the initial plot, including creating the figure and axes object
def plot_function():
    # Create the initial 3D surface plot using the x, t, and T_arr arrays
    ax.plot_surface(X, T, T_arr, cmap='coolwarm')
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


# Code to generate the initial plot, including creating the figure and axes object
# Use FuncAnimation to create the animation, passing it the Animation object, figure, and update function
animation = animation.FuncAnimation(fig, update_function, init_func=plot_function, frames=100)
plt.show()

# Save the animation to a file
animation.save('stefan_problem1.gif', fps=30)

# Calculate the temperature distribution at the given time
x = np.linspace(0, L, int(L / dx))  # position array in meters
T_dist = T_arr[int(time / dt), :]  # temperature distribution array in Kelvins

# Plot the temperature distribution
plt.plot(x, T_dist)
plt.xlabel('x (m)')
plt.ylabel('T (K)')
plt.show()
