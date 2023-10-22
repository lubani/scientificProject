import os
import tkinter as tk
import google
import logging
from matplotlib.ticker import ScalarFormatter
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from matplotlib.backends._backend_tk import NavigationToolbar2Tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from tkinter import ttk
import deepLearning
from deepLearning import *
from PIL import Image, ImageTk
import backend
from backend import *
from keras.callbacks import EarlyStopping

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
        self.choiceWindow = chooseInput()
        self.choiceWindow.wait_window()
        choice = self.choiceWindow.v.get()
        print(f"Debug: User choice for input window = {choice}")

        if choice == 0:
            self.input_window = InputWindow0()
            self.input_window.mainloop()
            self.L, self.t_max, self.k, self.c, self.rho, self.T_m, self.LH = self.input_window.submit()
            self.input_window.destroy()
            self.pcm = backend.customPCM(self.k, self.c, self.rho, self.T_m, self.LH)
        else:
            self.input_window = InputWindow1()
            self.input_window.mainloop()
            self.L, self.t_max, self.material = self.input_window.submit()
            self.input_window.destroy()
            self.pcm = globals()[self.material]()

        print(f"Debug: Selected material = {self.material if choice else 'Custom'}")

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
        self.time_scale.grid(row=0, column=0, sticky='nsew')

        # Create a variable for the dropdown selection
        self.solution_type = tk.StringVar(self.root)
        self.solution_type.set("Analytical")  # default value

        # Create the dropdown menu
        solution_menu = tk.OptionMenu(self.frame, self.solution_type,
                                      "Analytical", "Enthalpy Method", "Implicit", "Numerical", "PINN")
        solution_menu.grid(row=0, column=1, sticky='nsew')  # Position the dropdown menu at the top center
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
        self.fig_PINN = plt.Figure(figsize=(15, 5), dpi=100)
        self.moving_boundary_locations = []
        self.indices = []

        self.calcAll()

    def calcAll(self):
        self.x_arr2 = self.x_arr5 = np.arange(0, self.L, self.pcm.dx)
        self.t_arr2 = self.t_arr5 = np.arange(self.pcm.dt, self.t_max, self.pcm.dt)

        print(f"Debug: self.x_arr2 length = {len(self.x_arr2)}, values = {self.x_arr2}")
        print(f"Debug: self.t_arr2 length = {len(self.t_arr2)}, values = {self.t_arr2}")

        self.T_arr_numerical, self.t_arr3 = self.pcm.calcTemperature3(self.x_arr2, self.t_max, self.pcm)
        self.T_arr_analytical = self.pcm.explicitSol(self.x_arr2, self.t_arr2, self.pcm)
        self.T_arr_implicit, self.H_arr_final, self.t_arr4 = self.pcm.implicitSol(self.x_arr2, self.t_max, self.pcm)
        self.T_arr_enth, self.H_arr_enth = self.pcm.solve_stefan_problem_enthalpy(self.pcm, self.L, self.t_max)
        # self.H_arr_final = self.H_arr_enth
        print(f"Debug: self.x_arr2 length = {len(self.x_arr2)}, values = {self.x_arr2}")
        print(f"Debug: self.t_arr2 length = {len(self.t_arr2)}, values = {self.t_arr2}")

        T_m = self.pcm.T_m  # Melting temperature from your PCM class
        T = self.T_arr_implicit  # Temperature field from your enthalpy solver
        moving_boundary_indices = self.pcm.calculate_moving_boundary_indices(self.T_arr_implicit, self.pcm.T_m)
        print("Debug: Moving boundary indices = ", moving_boundary_indices)
        # Convert indices to actual spatial locations if needed
        dx = self.pcm.dx  # Your spatial grid spacing from your PCM class

        for i, val in enumerate(moving_boundary_indices):
            if val is not None:
                self.moving_boundary_locations.append(val * dx)
                self.indices.append(i)

        self.moving_boundary_locations = np.array(self.moving_boundary_locations)
        self.indices = np.array(self.indices)

        # Call update_plots to display the selected solution
        self.update_solution_type()

    def update_solution_type(self, *args):
        selected_solution = self.solution_type.get()

        print(f"Debug: Selected solution type = {selected_solution}")

        if selected_solution == "Analytical":
            self.T_arr_to_display = self.T_arr_analytical
            self.t_arr_final = self.t_arr2
        elif selected_solution == "Numerical":
            self.T_arr_to_display = self.T_arr_numerical
            self.t_arr_final = self.t_arr3
        elif selected_solution == "Implicit":
            self.T_arr_to_display = self.T_arr_implicit
            self.t_arr_final = self.t_arr4
        elif selected_solution == "Enthalpy Method":
            self.T_arr_to_display = self.T_arr_enth
            self.t_arr_final = self.t_arr5
        elif selected_solution == 'PINN':

            # Generate data based on the selected problem type
            x, y, x_boundary = self.pcm.generate_data(self.L, self.t_max)

            # Calculate boundary indices
            self.boundary_indices = self.pcm.calculate_boundary_indices(
                x, self.L, self.pcm.dt, T=self.T_arr_implicit,
                T_m=self.pcm.T_m, mode='initial')
            print("Debug: Calculated boundary_indices =", self.boundary_indices)

            # Here x and y are your initial data points
            initial_data = (x, y)

            # Create CustomPINNModel object
            model = CustomPINNModel(input_dim=2, output_dim=1, alpha=self.pcm.alpha2,
                                    T_m=self.pcm.T_m, T_a=self.pcm.T_a, boundary_indices=self.boundary_indices,
                                    initial_data=initial_data, y=y, x_max=self.L,
                                    moving_boundary_locations=self.moving_boundary_locations)

            self.loss_values, self.accuracy_values, self.Temperature_pred, self.Boundary_pred = train_PINN(
                model, x, model.y_T, model.y_B, self.T_arr_implicit, self.pcm, 25)

            # Show the PINN plots in a new Tkinter window
            self.show_PINN_plots()

        # Update the 'to' parameter of the timescale widget
        self.time_scale.config(to=self.t_arr_final[-1])

        # Call update_plots to update the plots based on the selected solution type
        self.update_plots()

    def update_plots(self):

        # Get the current time index from the timescale widget
        t_idx = int(self.time_scale.get())
        t_idx = t_idx if t_idx < self.T_arr_to_display.shape[1] else self.T_arr_to_display.shape[1] - 1

        # Update line plot 3 using the selected time index
        self.update_line_plot3(self.x_arr2, self.T_arr_to_display[:, t_idx])
        # if self.H_arr_final is None:
        #     self.H_arr_final = self.pcm.calcEnthalpy2(self.T_arr_to_display, self.pcm)

        # print("H values from calcEnthalpy2:", self.H_arr_final)

        self.energy_sufficiency = self.pcm.calcEnergySufficiency(self.H_arr_final)
        self.update_line_plot(self.H_arr_final[:, t_idx], self.T_arr_to_display[:, t_idx])
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

    def show_PINN_plots(self):
        # Create a new Tkinter window
        new_window = tk.Toplevel(self.root)
        new_window.title("PINN Plots")

        # Create a main frame
        main_frame = tk.Frame(new_window)
        main_frame.grid(row=0, column=0)

        # Create a frame for the canvas
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.grid(row=0, column=0)

        # Create a frame for the toolbar
        toolbar_frame = tk.Frame(main_frame)
        toolbar_frame.grid(row=1, column=0)

        # Create a canvas
        canvas = FigureCanvasTkAgg(self.fig_PINN, master=canvas_frame)
        canvas.get_tk_widget().grid(row=0, column=0)

        # Add a toolbar
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        toolbar.pack()

        # Create subplots
        ax1 = self.fig_PINN.add_subplot(151)
        ax2 = self.fig_PINN.add_subplot(152)
        ax2a = self.fig_PINN.add_subplot(153)
        ax3 = self.fig_PINN.add_subplot(154, projection='3d')
        ax4 = self.fig_PINN.add_subplot(155)

        # Plot loss and accuracy for T
        ax1.plot(self.loss_values)
        ax1.set_title('Loss during Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        accuracy_T_values = self.accuracy_values['accuracy_T']

        ax2.plot(accuracy_T_values)
        ax2.set_title('Accuracy T during Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        # New: Plot accuracy for B
        accuracy_B_values = self.accuracy_values['accuracy_B']

        ax2a.plot(accuracy_B_values)
        ax2a.set_title('Accuracy B during Training')
        ax2a.set_xlabel('Epoch')
        ax2a.set_ylabel('Accuracy')

        # Compute expected and actual sizes
        x_dim = len(self.x_arr2)
        t_dim = len(self.t_arr2)
        expected_size = x_dim * t_dim
        actual_size = self.Temperature_pred.shape[0]  # Change this to the actual temperature predictions

        # Compute the minimum size to consider and the new time dimension based on this size
        min_size = min(expected_size, actual_size)
        new_t_dim = min_size // x_dim  # Integer division to get the floor value

        # Slice and reshape Temperature_pred just once based on new_t_dim and x_dim
        self.Temperature_pred_sliced = self.Temperature_pred[:min_size, :]
        self.Temperature_pred_reshaped = self.Temperature_pred_sliced.reshape((x_dim, new_t_dim))
        print(f"Debug: Shape of self.t_arr2: {self.t_arr2.shape}")
        print(f"Debug: Shape of self.Boundary_pred: {self.Boundary_pred.shape}")

        self.Boundary_pred_reshaped = np.reshape(self.Boundary_pred, -1)[:len(self.t_arr2)]
        print(f"Debug: Reshaped self.Boundary_pred: {self.Boundary_pred_reshaped.shape}")

        # Find the minimum and maximum values in Temperature_pred_reshaped
        z_min = np.min(self.Temperature_pred_reshaped)
        z_max = np.max(self.Temperature_pred_reshaped)

        # Set the z-axis limits
        # ax3.set_zlim(z_min, z_max)
        # Change z-axis label formatting to decimal
        ax3.zaxis.set_major_formatter(ScalarFormatter(useMathText=False, useOffset=False))

        X, T = np.meshgrid(self.x_arr2, self.t_arr2)
        ax3.plot_surface(X, T, np.transpose(self.Temperature_pred_reshaped), cmap='coolwarm')
        ax3.set_title('Temperature Distribution (PINN)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_zlabel('Temperature')

        # Plotting the predicted boundary
        ax4.plot(self.t_arr2, self.Boundary_pred_reshaped, label='Predicted Boundary')
        print("Debug: our indices:", self.indices)
        print("Debug: moving_boundary_locations:", self.moving_boundary_locations)
        # Plotting the true boundary
        true_boundary_times = np.array(self.indices) * self.pcm.dt  # assuming self.dt is your time step
        ax4.plot(true_boundary_times, self.moving_boundary_locations, label='True Boundary', linestyle='--')

        ax4.set_title('Boundary Location (PINN)')
        ax4.set_xlabel('t')
        ax4.set_ylabel('Boundary Location')
        ax4.legend()

        canvas.draw()

    def update_line_plot(self, x, y):
        self.line1.set_data(x, y)
        self.ax1.relim()
        # self.ax1.set_xlim([np.min(x), np.max(x)])
        # self.ax1.set_ylim([np.min(y), np.max(y)])
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

        self.line1.set_ydata(self.H_arr_final[:, self.t_idx])
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
    # logging.basicConfig(level=logging.INFO)
    # if int(os.environ.get("PRODUCTION", 0)) == 1:
    #     logging_client = google.cloud.logging.Client()
    #     logging_client.setup_logging()
    app = App()
    app.run()  # This line should be the last one in the main block

