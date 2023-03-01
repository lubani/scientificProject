import math
import tkinter as tk
import numpy as np
from PIL import ImageGrab
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

np.set_printoptions(precision=8, suppress=True)


class HeatConduction:
    def __init__(self):
        super().__init__()
        # Initialize the inherited classes
        super(HeatConduction, self).__init__()

        # Other variables and constants
        self.LH = 1  # latent heat of fusion
        self.T = 0  # initial temperature of the bar (K)
        self.T_m = 1  # melting temperature (K)
        self.rho = 1.7
        # Finite differences parameters
        self.dx = 0.2  # spatial step size (m)
        self.dt = 0.4 * self.dx ** 2

        # Number of time steps
        self.n_steps = 100
        self.timer = 10

    def alpha(self, k, c, rho):
        if c == 0:
            return 1
        return k / (c * rho)

    def calcThermalConductivity(self, temp):
        return 0.001561 + 5.426 * 1e-11 * temp ** 3

    def calcSpecificHeat(self, temp):
        ca = -2.32 * 1e-2
        cb = 2.13 * 1e-3
        cc = 1.5 * 1e-5
        cd = -7.37 * 1e-8
        ce = 9.66 * 1e-11
        return ca + cb * temp + cc * temp ** 2 + cd * temp ** 3 + ce * temp ** 4

    def calcTemperature(self, x_arr, t_arr, T_arr):
        lmbda = self.dt / self.dx ** 2
        for j in range(len(t_arr) - 1):
            for i in range(1, len(x_arr) - 1):
                if x_arr[i] != 0 and t_arr[j] != 0:
                    T_arr[i, j + 1] = T_arr[i, j] * (1 - 2 * lmbda) + lmbda * \
                                      (T_arr[i + 1, j] + T_arr[i - 1, j])
        return T_arr

    def calcTemperature3(self, x_arr, t_arr):
        lmbda = self.dt / self.dx ** 2
        # T_vec = np.ones_like(x_arr) * 253
        T_vec = np.ones_like(x_arr)
        T_vec[0] = 0  # set the left boundary to 0
        T_vec[-1] = 1  # set the right boundary to 1
        with open("matrix.txt", "a+") as f:
            for j in range(len(t_arr)):
                T_new = T_vec.copy()
                for i in range(1, len(T_vec) - 1):
                    T_new[i] = T_vec[i] * (1 - 2 * lmbda) + lmbda * \
                               (T_vec[i + 1] + T_vec[i - 1])
                T_vec = T_new.copy()
                f.write(f'T[t={t_arr[j]:.1f}]:\n{T_vec}\n')
                print(f'T[t={t_arr[j]:.1f}]:\n{T_vec}')
        return T_vec

    def calcEnthalpy2(self, T):
        if T >= self.T_m:
            E = (T - self.T_m) * self.rho * self.calcSpecificHeat(T)
        else:
            E = self.calcSpecificHeat(T) * (self.T_m - T)
        return E

    def explicitSol(self, x_val, t_val):
        return math.erf(x_val / (2 * math.sqrt(t_val)))


class InputWindow(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Input Window")
        # Create input fields
        x_label = tk.Label(self, text="Length of the bar:")
        x_label.pack(side=tk.LEFT)
        self.x_entry = tk.Entry(self)
        self.x_entry.pack(side=tk.LEFT)

        t_label = tk.Label(self, text="Final time:")
        t_label.pack(side=tk.LEFT)
        self.t_entry = tk.Entry(self)
        self.t_entry.pack(side=tk.LEFT)
        # Create submit button
        submit_button = tk.Button(self, text="Submit", command=self.submit)
        submit_button.pack(side=tk.LEFT)

    def submit(self):
        self.x_input = self.x_entry.get()
        self.t_input = self.t_entry.get()
        x = float(self.x_input)
        T_f = float(self.t_input)

        # Close the window and return the input values
        self.quit()
        return x, T_f


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.line3 = None
        self.input_window = InputWindow()
        self.input_window.mainloop()

        # Get the input values from the window
        self.L, self.t_max = self.input_window.submit()
        self.input_window.destroy()
        self.HC = HeatConduction()
        self.m = 100
        self.LH = 1
        self.T = 0
        self.T_m = 1
        self.rho = 1.7
        self.M = 100

        self.dx = 1
        self.dt = 0.4 * self.dx ** 2
        self.root = tk.Tk()
        self.root.title("Heat Conduction")
        self.left = 10
        self.top = 10
        self.title = 'Regolith Properties'
        self.width = 1280
        self.height = 1000

        # Create the 2D line plot
        self.fig1, self.ax1 = plt.subplots(figsize=(4, 4), dpi=100)
        self.ax1.set_title("E(x) Line Plot")
        self.ax1.set_xlabel("E")
        self.ax1.set_ylabel("T")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.root)

        self.fig2 = plt.Figure(figsize=(4, 4), dpi=100)
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        self.ax2.set_title("3D Surface Plot")
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel("t")
        self.ax2.set_zlabel("T")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.root)
        self.fig3 = plt.Figure(figsize=(4, 4), dpi=100)

        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.set_title("T(x) Line Plot")
        self.ax3.set_xlabel("x")
        self.ax3.set_ylabel("T")
        self.canvas3 = FigureCanvasTkAgg(self.fig3, self.root)
        self.continue_init()

    def continue_init(self):
        self.t_arr = np.linspace(0, self.t_max, 10)
        self.x_arr = np.linspace(0, self.L, 10)

        self.E_arr = np.zeros((len(self.x_arr), len(self.t_arr)))
        self.T_arr = np.ones_like(self.E_arr) * self.T_m

        # self.k = self.HC.calcThermalConductivity(self.T_m)
        # self.c = self.HC.calcSpecificHeat(self.T_m)
        self.T_arr[:, 0] = self.T_m
        self.T_arr[0, :] = self.T
        self.calcAll()

    def calcAll(self):
        self.T_arr = self.HC.calcTemperature(self.x_arr, self.t_arr, self.T_arr)
        for i in range(len(self.x_arr)):
            for j in range(len(self.t_arr)):
                self.E_arr[i, j] = self.HC.calcEnthalpy2(self.T_arr[i, j])
        self.T_arr = np.where(np.isnan(self.T_arr), self.T, self.T_arr)
        self.update_line_plot(self.E_arr, self.T_arr)
        self.update_surface_plot(self.x_arr, self.t_arr, self.T_arr)
        self.x_arr2 = np.arange(0, self.L + self.dx, self.dx)
        self.t_arr2 = np.arange(0, self.t_max + self.dt, self.dt)
        with open("matrix.txt", "w+") as f:
            f.write(f'x_arr = {self.x_arr2}\n')
            print(f'x_arr = {self.x_arr2}')
            f.write(f't_arr = {self.t_arr2}\n')
            print(f't_arr = {self.t_arr2}')
        self.T_vec = self.HC.calcTemperature3(self.x_arr2, self.t_arr2)
        print(f'T_vec = {self.T_vec}')
        self.update_line_plot3(self.x_arr2, self.T_vec)
        T_vec_explicit = np.empty_like(self.T_vec)
        for i, val in enumerate(self.x_arr2):
            T_vec_explicit[i] = self.HC.explicitSol(val, self.t_max)
        with open("matrix.txt", "a+") as f:
            f.write(f'\nexplicit T_vec = {T_vec_explicit}\n')
        print(f'explicit T_vec = {T_vec_explicit}')
        label_text = f'x array = {self.x_arr2}\n' \
                     f't array = {self.t_arr2}\n' \
                     f'T array using FDM = {self.T_vec}\n' \
                     f'explicit T array = {T_vec_explicit}'
        labelx = tk.Label(self.root, text=label_text, font='sans')
        labelx.place(relx=0.5, rely=1.0, anchor="s")

        # take a screenshot of the whole window
        # screenshot = ImageGrab.grab(bbox=(self.root.winfo_rootx(), self.root.winfo_rooty(),
        #                             self.root.winfo_rootx() + self.root.winfo_width(),
        #                             self.root.winfo_rooty() + self.root.winfo_height()))
        #
        # # save the screenshot to a file
        # screenshot.save("tk.png")
        self.run()

    def run(self):
        self.root.mainloop()
        # self.root.quit()
        # self.root.destroy()

    def update_line_plot(self, x, y):
        self.line1 = self.ax1.plot(x, y)
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.root)
        self.canvas1.get_tk_widget().pack(side=tk.LEFT)
        self.line1[0].set_data(x, y)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.canvas1.print_figure("TE.png")
        self.canvas1.draw()

    def update_line_plot3(self, x, y):

        self.line3 = self.ax3.plot(x, y)
        self.canvas3 = FigureCanvasTkAgg(self.fig3, self.root)
        self.canvas3.get_tk_widget().pack(side=tk.LEFT)
        # self.line3[0].set_data(x, y)
        self.ax3.relim()
        self.ax3.autoscale_view()
        # self.ax3.legend(labels=[f'{x_:.2f}' for x_ in self.x_arr2])
        self.canvas3.print_figure("Tx.png")
        self.canvas3.draw()

    def update_surface_plot(self, x, y, z):
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.root)
        self.canvas2.get_tk_widget().pack(side=tk.LEFT)
        self.surface = self.ax2.plot_surface(x, y, z, cmap='coolwarm')
        self.canvas2.print_figure("Txt.png")
        self.canvas2.draw()

    def save_plot(self, canvas):
        canvas.grab().save("interfaceAnim.png")


if __name__ == '__main__':
    app = App()
    # app.root.mainloop()
