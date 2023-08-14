import os
import tkinter as tk
import google
import logging
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk

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

        # Create the main window
        self.root = tk.Tk()
        self.root.wm_title("Embedding in Tk")
        # self.root.geometry("1200x800")
        self.root.configure(bg='white')
        self.root.title("Heat Conduction")
        self.root.grid_anchor("center")

        # Create a frame to hold the scale and the line plot
        self.frame = tk.Frame(self.root)
        self.frame.grid(row=0, column=0, columnspan=2, sticky='n')  # Span the frame across columns

        # Create the scale widget
        self.time_scale = tk.Scale(self.root, from_=0, to=self.t_max, resolution=self.pcm.dt,
                                   orient=tk.HORIZONTAL, command=self.update_time)
        # Position the scale at the top of the frame
        self.time_scale.grid(row=0, column=0, sticky='n')

        # Create a variable for the dropdown selection
        self.solution_type = tk.StringVar(self.root)
        self.solution_type.set("Implicit")  # default value

        # Create the dropdown menu
        solution_menu = tk.OptionMenu(self.frame, self.solution_type,
                                      "Implicit", "Numerical", "Analytical")
        solution_menu.grid(row=0, column=1, sticky='n')  # Position the dropdown menu at the top center
        # Link the update function to the variable
        self.solution_type.trace('w', self.update_solution_type)

        self.root.focus_force()

        # Create the 2D line plot
        self.fig1, self.ax1 = plt.subplots(figsize=(4, 3), dpi=100)
        self.ax1.set_title("T(E) Line Plot")
        self.ax1.set_xlabel("E")
        self.ax1.set_ylabel("T")
        self.canvas1 = FigureCanvasTkAgg(self.fig1, self.root)
        self.canvas1.draw()
        self.fig2 = plt.Figure(figsize=(4, 3), dpi=100)
        self.ax2 = self.fig2.add_subplot(111, projection='3d')
        self.ax2.set_title("3D Surface Plot")
        self.ax2.set_xlabel("x")
        self.ax2.set_ylabel("t")
        self.ax2.set_zlabel("T")
        self.canvas2 = FigureCanvasTkAgg(self.fig2, self.root)

        # self.fig3 = plt.Figure(figsize=(4, 3), dpi=100)
        # self.ax3 = self.fig3.add_subplot(111)
        # self.ax3.set_title("T(x) Line Plot")
        # self.ax3.set_xlabel("x")
        # self.ax3.set_ylabel("T")
        # self.canvas3 = FigureCanvasTkAgg(self.fig3, self.root)


        # Create the line plot
        self.fig3 = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.grid(True)
        self.ax3.set_xlabel("x [m]")
        self.ax3.set_ylabel("T [C]")
        self.ax3.set_title("T(x)")
        # Create the canvas for the line plot and position it at the top of the frame
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.root)  # A tk.DrawingArea.
        self.canvas3.draw()
        self.canvas3.get_tk_widget().grid(row=1, column=1, sticky='nsew')
        # Create lines for temperature and enthalpy
        self.line1, = self.ax1.plot([], [], 'r-', label='Line Plot 1')  # Notice the comma
        # self.line2, = self.ax3.plot([], [], 'b-', label='H(x)')  # Notice the comma
        self.line3, = self.ax3.plot([], [], 'g-', label='Line Plot 3')  # Notice the comma

        self.ax3.legend()

        # Configure the frame to give all extra space to the line plot
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        # Create the text widget and add the contents of the file
        self.text_output = tk.Text(self.root)
        self.v = tk.Scrollbar(self.root, orient='vertical')
        # Grid the widgets
        self.canvas1.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        self.canvas2.get_tk_widget().grid(row=2, column=0, sticky='nsew')
        self.text_output.grid(row=2, column=1, sticky='nsew')

        # Configure the main window to distribute extra space among the widgets
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(2, weight=1)

        self.calcAll()

    def calcAll(self):
        selected_solution = self.solution_type.get()
        self.x_arr2 = np.arange(0, self.L + self.pcm.dx, self.pcm.dx)
        self.t_arr2 = np.arange(self.pcm.dt, self.t_max, self.pcm.dt)
        # Compute both solutions (only once)

        self.T_arr_numerical, self.t_arr3 = self.pcm.calcTemperature3(self.x_arr2, self.t_max, self.pcm)
        self.T_arr_analytical = self.pcm.explicitSol(self.x_arr2, self.t_arr2, self.pcm)
        self.T_arr_implicit, self.t_arr4 = self.pcm.implicitSol(self.x_arr2, self.t_max, self.pcm)
        # Call update_plots to display the selected solution
        self.update_solution_type()

    def update_solution_type(self, *args):
        selected_solution = self.solution_type.get()
        if selected_solution == "Analytical":
            self.T_arr_to_display = self.T_arr_analytical
            self.t_arr_final = self.t_arr2
        elif selected_solution == "Numerical":
            self.T_arr_to_display = self.T_arr_numerical
            self.t_arr_final = self.t_arr3
        elif selected_solution == "Implicit":
            self.T_arr_to_display = self.T_arr_implicit
            self.t_arr_final = self.t_arr4

        # Update the 'to' parameter of the time scale widget
        self.time_scale.config(to=self.t_arr_final[-1])

        # Call update_plots to update the plots based on the selected solution type
        self.update_plots()

    def update_plots(self):

        # Get the current time index from the timescale widget
        t_idx = int(self.time_scale.get())
        t_idx = t_idx if t_idx < self.T_arr_to_display.shape[1] else self.T_arr_to_display.shape[1] - 1

        # Update line plot 3 using the selected time index
        self.update_line_plot3(self.x_arr2, self.T_arr_to_display[:, t_idx])

        self.H_arr = self.pcm.calcEnthalpy2(self.T_arr_to_display, self.pcm)
        print("H values from calcEnthalpy2:", self.H_arr)

        self.energy_sufficiency = self.pcm.calcEnergySufficiency(self.H_arr)
        self.update_line_plot(self.H_arr[:, t_idx], self.T_arr_to_display[:, t_idx])
        self.update_surface_plot(self.x_arr2, self.t_arr_final, self.T_arr_to_display)
        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()
        # Update the text widget with the contents of the file
        # with open("matrix.txt", "r") as f:
        #     content = f.read()
        #     self.text_output.insert(tk.END, content)

        # # Wait until the window is closed before continuing execution
        # self.v.config(command=self.text_output.yview)
        # self.text_output["yscrollcommand"] = self.v.set
        # self.v.grid(row=1, column=2, sticky='ns')
        # self.root.wait_window()
        # self.run()

    def update_line_plot(self, x, y):
        self.line1.set_data(x, y)
        self.ax1.relim()
        self.ax1.set_xlim([np.min(x), np.max(x)])
        self.ax1.set_ylim([np.min(y), np.max(y)])
        self.ax1.autoscale_view()
        self.canvas1.print_figure("TE.png")
        self.canvas1.draw()

    def update_line_plot3(self, x, y):
        self.line3.set_data(x, y)
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.canvas3.print_figure("Tx.png")
        self.canvas3.draw()

    def update_time(self, event):
        t_idx = int(self.time_scale.get())
        self.t_idx = t_idx if t_idx < self.T_arr_to_display.shape[1] else self.T_arr_to_display.shape[1] - 1

        self.line1.set_ydata(self.H_arr[:, self.t_idx])
        self.ax1.relim()
        self.ax1.autoscale_view()
        self.canvas1.draw()

        self.line3.set_ydata(self.T_arr_to_display[:, self.t_idx])
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.canvas3.draw()

    def update_surface_plot(self, x, y, z):
        self.ax2.clear()
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
    app.run()  # This line should be the last one in the main block

