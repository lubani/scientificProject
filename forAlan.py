import numpy as np
from matplotlib import pyplot as plt
import matplotlib.animation as animation


def func(x, t):
    return t + 0.5 * (x ** 2)
    # return t + 0.5 * (x ** 2)


def rotate(angle):
    ax.view_init(azim=angle)


plt.rcParams["figure.autolayout"] = True

x = np.linspace(0, 1, 5)

t = np.array([0, .3, .6, 1, 1.5])

temp = func(x, t)

x, temp = np.meshgrid(x, temp)
print(f'x = {x}')
print(f't = {t}')

fig = plt.figure()
ax = plt.axes(projection='3d')

print("temperatures are:")
print(temp)
ax.plot_surface(x, t, temp, cmap="afmhot_r", alpha=.7)

rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 2), interval=100)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('temperature')
fig.canvas.draw()
fig.draw_without_rendering()

plt.show()

rot_animation.save('rotation.gif', dpi=80, writer='imagemagick')
