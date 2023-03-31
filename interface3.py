import os
import tkinter as tk
import google
import logging
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import backend
from backend import *

np.set_printoptions(precision=8, suppress=True)


class chooseInput(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Input Window")
        self.label = tk.Label(self, text="Choose mode:")
        self.label.grid(row=0, column=0)
        self.v = tk.IntVar()
        self.v.set(0)
        self.choices = [("Input thermo-physical properties", 0),
                        ("Choose between materials", 1)]
        for string, val in self.choices:
            tk.Radiobutton(self, text=string, variable=self.v, value=val).grid(row=val + 1, column=0)

        # Create submit button
        submit_button = tk.Button(self, text="Submit", command=self.callInputMode)
        submit_button.grid(row=3, column=0, columnspan=2)

    def callInputMode(self):
        self.destroy()


class InputWindow0(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Input Window")
        # Create input fields
        x_label = tk.Label(self, text="Length of the bar:")
        x_label.grid(row=0, column=0)
        self.x_entry = tk.Entry(self)
        self.x_entry.grid(row=0, column=1)

        t_label = tk.Label(self, text="Final time:")
        t_label.grid(row=1, column=0)
        self.t_entry = tk.Entry(self)
        self.t_entry.grid(row=1, column=1)

        k_label = tk.Label(self, text="Conductivity:")
        k_label.grid(row=2, column=0)
        self.k_entry = tk.Entry(self)
        self.k_entry.grid(row=2, column=1)

        c_label = tk.Label(self, text="Specific Heat:")
        c_label.grid(row=3, column=0)
        self.c_entry = tk.Entry(self)
        self.c_entry.grid(row=3, column=1)

        rho_label = tk.Label(self, text="Density:")
        rho_label.grid(row=4, column=0)
        self.rho_entry = tk.Entry(self)
        self.rho_entry.grid(row=4, column=1)

        Tm_label = tk.Label(self, text="Melting Temperature:")
        Tm_label.grid(row=5, column=0)
        self.Tm_entry = tk.Entry(self)
        self.Tm_entry.grid(row=5, column=1)

        LH_label = tk.Label(self, text="Latent Heat:")
        LH_label.grid(row=6, column=0)
        self.LH_entry = tk.Entry(self)
        self.LH_entry.grid(row=6, column=1)
        # Create submit button
        submit_button = tk.Button(self, text="Submit", command=self.submit)
        submit_button.grid(row=7, column=1, columnspan=2)

    def submit(self):
        self.x_input = self.x_entry.get()
        self.t_input = self.t_entry.get()

        self.k_input = self.k_entry.get()
        self.c_input = self.c_entry.get()
        self.rho_input = self.rho_entry.get()

        self.Tm_input = self.Tm_entry.get()
        self.LH_input = self.LH_entry.get()
        x = float(self.x_input)
        t = float(self.t_input)
        k = float(self.k_input)
        c = float(self.c_input)
        rho = float(self.rho_input)
        Tm = float(self.Tm_input)
        LH = float(self.LH_input)

        # Close the window and return the input values
        self.quit()
        return x, t, k, c, rho, Tm, LH


class InputWindow1(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Input Window")
        # Create input fields
        x_label = tk.Label(self, text="Length of the bar:")
        x_label.grid(row=0, column=0)
        self.x_entry = tk.Entry(self)
        self.x_entry.grid(row=0, column=1)

        t_label = tk.Label(self, text="Final time:")
        t_label.grid(row=1, column=0)
        self.t_entry = tk.Entry(self)
        self.t_entry.grid(row=1, column=1)

        # Create material selection dropdown
        material_label = tk.Label(self, text="Material:")
        material_label.grid(row=2, column=0)
        self.material_var = tk.StringVar()
        self.material_var.set("Iron")  # Set default material to iron
        material_options = ["Iron", "Regolith"]
        self.material_dropdown = tk.OptionMenu(self, self.material_var, *material_options)
        self.material_dropdown.grid(row=2, column=1)

        # Create submit button
        submit_button = tk.Button(self, text="Submit", command=self.submit)
        submit_button.grid(row=3, column=1, columnspan=2)

    def submit(self):
        self.x_input = self.x_entry.get()
        self.t_input = self.t_entry.get()
        self.material_input = self.material_var.get()
        x = float(self.x_input)
        t = float(self.t_input)

        # Close the window and return the input values
        self.quit()
        return x, t, self.material_input


class App:
    def __init__(self):
        # super().__init__()
        self.choiceWindow = chooseInput()
        self.choiceWindow.wait_window()
        if self.choiceWindow.v.get() == 0:
            self.input_window = InputWindow0()
            self.input_window.mainloop()
            self.L, self.t_max, self.k, self.c, self.rho, \
            self.T_m, self.LH = self.input_window.submit()
            self.input_window.destroy()
            self.pcm = backend.customPCM(self.k, self.c, self.rho, self.T_m, self.LH)
        else:
            self.input_window = InputWindow1()
            # Get the input values from the window
            self.input_window.mainloop()
            self.L, self.t_max, self.material = self.input_window.submit()
            self.input_window.destroy()
            self.pcm = globals()[self.material]()

        # instead of self.pcm=Iron(),
        # add functionality to determine if string is "Iron" then return object of classs Iron(),
        # same for "Regolith" and class Regolith()

        self.root = tk.Tk()
        self.root.configure(bg='white')
        self.root.title("Heat Conduction")
        self.root.grid_anchor("center")

        self.root.focus_force()
        # Create the 2D line plot
        self.fig1, self.ax1 = plt.subplots(figsize=(4, 3), dpi=100)
        self.ax1.set_title("T(E) Line Plot")
        self.ax1.set_xlabel("E")
        self.ax1.set_ylabel("T")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.root)

        self.fig2 = plt.Figure(figsize=(4, 3), dpi=100)
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        self.ax2.set_title("3D Surface Plot")
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel("t")
        self.ax2.set_zlabel("T")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.root)
        self.fig3 = plt.Figure(figsize=(4, 3), dpi=100)

        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.set_title("T(x) Line Plot")
        self.ax3.set_xlabel("x")
        self.ax3.set_ylabel("T")
        self.canvas3 = FigureCanvasTkAgg(self.fig3, self.root)

        # Create the text widget and add the contents of the file
        self.text_output = tk.Text(self.root)

        # Pack the widgets using grid
        self.canvas1.get_tk_widget().grid(row=0, column=0)
        self.canvas2.get_tk_widget().grid(row=1, column=0)
        self.canvas3.get_tk_widget().grid(row=0, column=1)
        self.text_output.grid(row=1, column=1)
        self.continue_init()

    def continue_init(self):
        self.t_arr = np.linspace(0, self.t_max, 10)
        self.x_arr = np.linspace(0, self.L, 10)

        self.E_arr = np.zeros((len(self.x_arr), len(self.t_arr)))
        self.T_arr = np.ones_like(self.E_arr) * self.pcm.T_m

        # self.T_arr[:, 0] = self.pcm.T_m
        self.T_arr[0, :] = self.pcm.T
        self.calcAll()

    def calcAll(self):
        self.T_arr = self.pcm.calcTemperature(self.x_arr, self.t_arr, self.T_arr, self.pcm)
        for i in range(len(self.x_arr)):
            for j in range(len(self.t_arr)):
                self.E_arr[i, j] = self.pcm.calcEnthalpy2(self.T_arr[i, j], self.pcm)
        self.T_arr = np.where(np.isnan(self.T_arr), self.pcm.T_m, self.T_arr)
        self.x_arr2 = np.arange(0, self.L + self.pcm.dx, self.pcm.dx)
        self.t_arr2 = np.arange(0, self.t_max + self.pcm.dt, self.pcm.dt)
        with open("matrix.txt", "w+") as f:
            f.write(f'length intervals = {self.x_arr2}\n')
            print(f'length intervals = {self.x_arr2}')
            f.write(f'time intervals = {self.t_arr2}\n')
            print(f'time intervals = {self.t_arr2}')
        self.T_vec, self.T_arr2, self.t_arr3 = self.pcm.calcTemperature3(self.x_arr2, self.t_max, self.pcm)
        print(f'temperature vector at final time = {self.T_vec}')
        self.update_line_plot3(self.x_arr2, self.T_vec)
        T_vec_explicit = np.empty_like(self.T_vec)
        for i, val in enumerate(self.x_arr2):
            T_vec_explicit[i] = self.pcm.explicitSol(val, self.t_max, self.pcm)
        with open("matrix.txt", "a+") as f:
            f.write(f'\ntemperature by error function at the last recorded time = {T_vec_explicit}\n')
        print(f'temperature by error function at final time = {T_vec_explicit}')
        self.t_arr4 = np.linspace(0, self.t_max, len(self.x_arr2))
        self.update_line_plot(self.E_arr, self.T_arr)
        print(f'{self.x_arr2.shape}, {self.t_arr3.shape}, {self.T_arr2.shape}')
        self.update_surface_plot(self.x_arr2, self.t_arr3, self.T_arr2)
        self.update_line_plot3(self.x_arr2, self.T_vec)
        with open("matrix.txt", "r") as f:
            content = f.read()
            self.text_output.insert(tk.END, content)

        # Wait until the window is closed before continuing execution
        self.root.wait_window()
        self.run()

    def update_line_plot(self, x, y):
        self.line1 = self.ax1.plot(x, y)
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.canvas1.print_figure("TE.png")
        self.canvas1.draw()

    def update_line_plot3(self, x, y):
        self.line3 = self.ax3.plot(x, y)
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.canvas3.print_figure("Tx.png")
        self.canvas3.draw()

    def update_surface_plot(self, x, y, z):
        x, y = np.meshgrid(x, y)
        self.surface = self.ax2.plot_surface(x, y, np.transpose(z), cmap='coolwarm')
        self.canvas2.print_figure("Txt.png")
        self.canvas2.draw()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    if int(os.environ.get("PRODUCTION", 0)) == 1:
        logging_client = google.cloud.logging.Client()
        logging_client.setup_logging()
    app = App()
