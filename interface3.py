import os
import tkinter as tk
# import google
# import logging
from matplotlib.ticker import ScalarFormatter
from tensorflow.keras.optimizers import Adam
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
        self.root.configure(bg='white')
        self.root.title("Heat Conduction")
        self.root.grid_anchor("center")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Create a frame to hold the scale and the line plot
        self.frame = tk.Frame(self.root)
        self.frame.grid(row=0, column=0, columnspan=3, sticky='n')  # Span the frame across columns

        # Create the scale widget
        self.time_scale = tk.Scale(self.root, from_=0, to=self.t_max, resolution=self.pcm.dt,
                                   orient=tk.HORIZONTAL, command=self.update_time)
        self.time_scale.grid(row=0, column=0, sticky='nsew')

        # Create a variable for the dropdown selection
        self.solution_type = tk.StringVar(self.root)
        self.solution_type.set("Analytical")  # default value

        # Create the dropdown menu for the main solution type selection
        solution_menu = tk.OptionMenu(self.frame, self.solution_type,
                                      "Analytical", "Enthalpy Method", "Implicit", "Numerical", "PINN")
        solution_menu.grid(row=0, column=1, sticky='nsew')
        self.solution_type.trace('w', self.handle_solution_type_change)
        # Initialize gold standard variables
        self.gold_standard = None
        self.gold_standard_temp_array = None
        self.temperature_solutions = {}  # Dictionary to hold temperature arrays for each solution type
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

        self.fig3 = plt.Figure(figsize=(5, 5), dpi=100)
        self.ax3 = self.fig3.add_subplot(111)
        self.ax3.grid(True)
        self.ax3.set_xlabel("x [m]")
        self.ax3.set_ylabel("T [C]")
        self.ax3.set_title("T(x)")
        self.canvas3 = FigureCanvasTkAgg(self.fig3, master=self.root)  # A tk.DrawingArea.
        self.canvas3.draw()
        self.canvas3.get_tk_widget().grid(row=1, column=1, sticky='nsew')
        self.line1, = self.ax1.plot([], [], 'r-', label='Line Plot 1')  # Notice the comma
        self.line3, = self.ax3.plot([], [], 'g-', label='Line Plot 3')  # Notice the comma

        self.ax3.legend()

        # Configure the frame to give all extra space to the line plot
        self.frame.grid_rowconfigure(0, weight=1)
        self.frame.grid_columnconfigure(0, weight=1)

        self.canvas1.get_tk_widget().grid(row=1, column=0, sticky='nsew')
        self.canvas2.get_tk_widget().grid(row=1, column=2, sticky='nsew')

        # Configure the main window to distribute extra space among the widgets
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(2, weight=1)
        self.root.grid_rowconfigure(3, weight=1)
        self.root.grid_columnconfigure(3, weight=1)
        self.fig_PINN = plt.Figure(figsize=(15, 5), dpi=100)
        self.moving_boundary_locations = []
        self.indices = []

        self.calcAll()
        self.update_solution_type()
        self.update_plots()

    def handle_gold_standard_selection(self, selected_solution):
        self.gold_standard = selected_solution
        self.set_gold_standard_temp_array(self.gold_standard)
        self.prepare_PINN_model_and_train()

    def set_gold_standard_temp_array(self, selected_solution):
        self.gold_standard_temp_array = self.temperature_solutions.get(selected_solution, None)
        self.temp_mask_array = self.temp_mask_arrays.get(selected_solution, None)
        self.bound_mask_array = self.bound_mask_arrays.get(selected_solution, None)
        self.moving_boundary_indices_to_display = self.moving_boundary_indices.get(selected_solution, None)

    def prompt_for_gold_standard(self):
        self.gold_standard_window = GoldStandardSelectionWindow(self.root, self.handle_gold_standard_selection)
        self.gold_standard_window.mainloop()

    def handle_solution_type_change(self, *args):
        selected_solution = self.solution_type.get()
        if (selected_solution == "PINN"):
            # Prompt the user to select the gold standard solution type
            self.prompt_for_gold_standard()
        else:
            # Handle other solution types normally
            self.update_solution_type()
            self.update_plots()

    def calcAll(self):
        print("Debug: Starting calcAll")

        # Initialize attributes
        self.moving_boundary_indices = {}
        self.temp_mask_arrays = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
            "Enthalpy Method": None
        }
        self.bound_mask_arrays = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
            "Enthalpy Method": None
        }
        self.temperature_solutions = {
            "Numerical": None,
            "Analytical": None,
            "Implicit": None,
            "Enthalpy Method": None
        }

        # Generate data
        print("Debug: Generating data")
        self.x, self.y_T, self.y_B, self.x_boundary, self.x_grid, self.t_grid, self.T_arr, self.H_arr = self.pcm.generate_data(
            self.L, self.t_max)
        nx = len(self.x_grid)
        nt = len(self.t_grid)

        print(f"Debug: nx = {nx}, nt = {nt}")
        print(f"Debug: x_grid shape = {self.x_grid.shape}, t_grid shape = {self.t_grid.shape}")

        self.initial_data_T = (self.x, self.y_T)
        self.initial_data_B = (self.x_boundary, self.y_B)

        # Numerical solution
        print("Debug: Calculating numerical solution")
        self.T_arr_numerical, self.temp_mask_arrays["Numerical"], self.bound_mask_arrays[
            "Numerical"], moving_boundary_indices_numerical = self.pcm.explicitNumerical(
            self.x_grid, self.t_grid, self.T_arr.copy(), self.pcm, self.temp_mask_arrays["Numerical"],
            self.bound_mask_arrays["Numerical"])
        self.temperature_solutions["Numerical"] = self.T_arr_numerical
        print(f"Debug: T_arr_numerical shape = {self.T_arr_numerical.shape}")
        print(f"Debug: temp_mask_arrays[\"Numerical\"] shape = {self.temp_mask_arrays['Numerical'].shape}")
        print(f"Debug: bound_mask_arrays[\"Numerical\"] shape = {self.bound_mask_arrays['Numerical'].shape}")
        print(f"Debug: temp_mask_arrays[\"Numerical\"]:\n{self.temp_mask_arrays['Numerical']}")
        print(f"Debug: bound_mask_arrays[\"Numerical\"]:\n{self.bound_mask_arrays['Numerical']}")

        # Analytical solution
        print("Debug: Calculating analytical solution")
        self.T_arr_analytical = self.pcm.analyticalSol(self.x_grid, self.t_grid, self.pcm)
        self.temp_mask_arrays["Analytical"], self.bound_mask_arrays["Analytical"] = compute_mask_arrays(
            self.T_arr_analytical[:, -1], self.pcm)
        self.temperature_solutions["Analytical"] = self.T_arr_analytical
        print(f"Debug: T_arr_analytical shape = {self.T_arr_analytical.shape}")
        print(f"Debug: temp_mask_arrays[\"Analytical\"] shape = {self.temp_mask_arrays['Analytical'].shape}")
        print(f"Debug: bound_mask_arrays[\"Analytical\"] shape = {self.bound_mask_arrays['Analytical'].shape}")
        print(f"Debug: temp_mask_arrays[\"Analytical\"]:\n{self.temp_mask_arrays['Analytical']}")
        print(f"Debug: bound_mask_arrays[\"Analytical\"]:\n{self.bound_mask_arrays['Analytical']}")

        try:
            print("Debug: Calculating moving boundary indices for analytical solution")
            moving_boundary_indices_analytical = self.pcm.calculate_boundary_indices(self.x_grid, self.L, self.pcm.dt,
                                                                                     T=self.T_arr_analytical,
                                                                                     T_m=self.pcm.T_m,
                                                                                     mode='moving_boundary')
            print(f"Debug: Moving boundary indices (analytical) = {moving_boundary_indices_analytical}")
        except Exception as e:
            print(f"Error in calculating moving boundary indices (analytical): {e}")
            return

        # Implicit solution
        print("Debug: Calculating implicit solution")
        self.T_arr_implicit, self.H_arr_final, self.temp_mask_arrays["Implicit"], self.bound_mask_arrays[
            "Implicit"], moving_boundary_indices_implicit = self.pcm.implicitSol(
            self.x_grid, self.t_grid, self.T_arr.copy(), self.H_arr.copy(), self.pcm, self.temp_mask_arrays["Implicit"],
            self.bound_mask_arrays["Implicit"])
        self.temperature_solutions["Implicit"] = self.T_arr_implicit
        print(f"Debug: T_arr_implicit shape = {self.T_arr_implicit.shape}")
        print(f"Debug: temp_mask_arrays[\"Implicit\"] shape = {self.temp_mask_arrays['Implicit'].shape}")
        print(f"Debug: bound_mask_arrays[\"Implicit\"] shape = {self.bound_mask_arrays['Implicit'].shape}")
        print(f"Debug: temp_mask_arrays[\"Implicit\"]:\n{self.temp_mask_arrays['Implicit']}")
        print(f"Debug: bound_mask_arrays[\"Implicit\"]:\n{self.bound_mask_arrays['Implicit']}")

        # Enthalpy method solution
        print("Debug: Calculating enthalpy method solution")
        try:
            self.T_arr_enth, self.H_arr_enth, self.temp_mask_arrays["Enthalpy Method"], self.bound_mask_arrays[
                "Enthalpy Method"], moving_boundary_indices_enthalpy = self.pcm.solve_stefan_problem_enthalpy(
                self.pcm, self.x_grid, self.t_grid, self.T_arr.copy(), self.H_arr.copy(),
                self.temp_mask_arrays["Enthalpy Method"], self.bound_mask_arrays["Enthalpy Method"])
            self.temperature_solutions["Enthalpy Method"] = self.T_arr_enth
            print(f"Debug: T_arr_enth shape = {self.T_arr_enth.shape}")
            print(
                f"Debug: temp_mask_arrays[\"Enthalpy Method\"] shape = {self.temp_mask_arrays['Enthalpy Method'].shape}")
            print(
                f"Debug: bound_mask_arrays[\"Enthalpy Method\"] shape = {self.bound_mask_arrays['Enthalpy Method'].shape}")
            print(f"Debug: temp_mask_arrays[\"Enthalpy Method\"]:\n{self.temp_mask_arrays['Enthalpy Method']}")
            print(f"Debug: bound_mask_arrays[\"Enthalpy Method\"]:\n{self.bound_mask_arrays['Enthalpy Method']}")
        except Exception as e:
            print(f"Error in calculating enthalpy method solution: {e}")
            return

        # Update moving boundary indices
        print("Debug: Updating moving boundary indices")
        self.moving_boundary_indices.update({
            "Analytical": moving_boundary_indices_analytical,
            "Numerical": moving_boundary_indices_numerical,
            "Implicit": moving_boundary_indices_implicit,
            "Enthalpy Method": moving_boundary_indices_enthalpy
        })

        print("Debug: calcAll finished successfully")

    def update_solution_type(self, *args):
        selected_solution = self.solution_type.get()
        print(f"Debug: Selected solution type = {selected_solution}")

        if selected_solution == "Analytical":
            self.T_arr_to_display = self.T_arr_analytical
            self.temp_mask_array = self.temp_mask_arrays["Analytical"]
            self.bound_mask_array = self.bound_mask_arrays["Analytical"]
            self.moving_boundary_indices_to_display = self.moving_boundary_indices["Analytical"]
        elif selected_solution == "Numerical":
            self.T_arr_to_display = self.T_arr_numerical
            self.temp_mask_array = self.temp_mask_arrays["Numerical"]
            self.bound_mask_array = self.bound_mask_arrays["Numerical"]
            self.moving_boundary_indices_to_display = self.moving_boundary_indices["Numerical"]
        elif selected_solution == "Implicit":
            self.T_arr_to_display = self.T_arr_implicit
            self.temp_mask_array = self.temp_mask_arrays["Implicit"]
            self.bound_mask_array = self.bound_mask_arrays["Implicit"]
            self.moving_boundary_indices_to_display = self.moving_boundary_indices["Implicit"]
        elif selected_solution == "Enthalpy Method":
            self.T_arr_to_display = self.T_arr_enth
            self.temp_mask_array = self.temp_mask_arrays["Enthalpy Method"]
            self.bound_mask_array = self.bound_mask_arrays["Enthalpy Method"]
            self.moving_boundary_indices_to_display = self.moving_boundary_indices["Enthalpy Method"]
        elif selected_solution == "PINN":
            self.prompt_for_gold_standard()

        print(f"Debug: self.T_arr_to_display shape = {self.T_arr_to_display.shape}")
        print(f"Debug: self.temp_mask_array shape = {self.temp_mask_array.shape}")
        print(f"Debug: self.bound_mask_array shape = {self.bound_mask_array.shape}")
        print(f"Debug: self.moving_boundary_indices_to_display shape = {self.moving_boundary_indices_to_display.shape}")

        self.update_plots()

    def prepare_PINN_model_and_train(self):
        self.x_input = self.x
        self.x_boundary_input = self.x_boundary
        self.boundary_indices = self.moving_boundary_indices_to_display

        self.batch_size = min(1024, self.x_input.shape[0])
        self.moving_boundary_locations = self.moving_boundary_indices_to_display

        self.indices = list(range(len(self.t_grid)))
        self.true_boundary_times = np.array(self.indices) * self.pcm.dt

        if self.bound_mask_array is None or self.temp_mask_array is None:
            print("Error: bound_mask_array or temp_mask_array is None")
            return

        self.y_T = self.gold_standard_temp_array.flatten()
        self.y_B = np.zeros_like(self.y_T)

        self.temp_mask_array = np.resize(self.temp_mask_array.flatten(), self.y_T.shape)
        self.bound_mask_array = np.resize(self.bound_mask_array.flatten(), self.y_T.shape)

        min_length = min(len(self.x_input), len(self.x_boundary_input), len(self.y_T), len(self.y_B))
        self.x_input = self.x_input[:min_length, :]
        self.x_boundary_input = self.x_boundary_input[:min_length, :]
        self.y_T = self.y_T[:min_length]
        self.y_B = self.y_B[:min_length]
        self.temp_mask_array = self.temp_mask_array[:min_length]
        self.bound_mask_array = self.bound_mask_array[:min_length]

        print(f"Debug: x_input shape = {self.x_input.shape}")
        print(f"Debug: x_boundary_input shape = {self.x_boundary_input.shape}")
        print(f"Debug: y_T shape = {self.y_T.shape}")
        print(f"Debug: y_B shape = {self.y_B.shape}")
        print(f"Debug: temp_mask_array shape = {self.temp_mask_array.shape}")
        print(f"Debug: bound_mask_array shape = {self.bound_mask_array.shape}")
        print(f"Debug: temp_mask_array: {self.temp_mask_array}")
        print(f"Debug: bound_mask_array: {self.bound_mask_array}")

        self.model = CustomPINNModel(
            input_dim=2,
            output_dim=1,
            alpha=self.pcm.alpha2,
            T_m=self.pcm.T_m,
            T_a=self.pcm.T_a,
            boundary_indices=self.boundary_indices,
            x_arr=self.x_grid,
            t_arr=self.t_grid,
            batch_size=self.batch_size,
            y_T=self.y_T,
            y_B=self.y_B,
            x_max=self.L,
            bound_mask_array=self.bound_mask_array,
            temp_mask_array=self.temp_mask_array,
            initial_data_T=self.initial_data_T,
            initial_data_B=self.initial_data_B,
            moving_boundary_locations=self.moving_boundary_locations,
            pcm=self.pcm,
            x_input=self.x_input,
            gold_standard=self.gold_standard_temp_array
        )

        # Initialize accuracy values
        self.scaled_accuracy_values = {'scaled_accuracy_T': [], 'scaled_accuracy_B': []}
        self.raw_accuracy_values = {'raw_accuracy_T': [], 'raw_accuracy_B': []}

        self.loss_values, accuracy_values, self.Temperature_pred, self.Boundary_pred = train_PINN(
            model=self.model,
            x=self.x_input,
            x_boundary=self.x_boundary_input,
            y_T=self.y_T,
            y_B=self.y_B,
            epochs=25,
            mask_T=self.temp_mask_array,
            mask_B=self.bound_mask_array,
            batch_size=self.batch_size
        )

        self.scaled_accuracy_values = {
            'scaled_accuracy_T': accuracy_values['scaled_accuracy_T'],
            'scaled_accuracy_B': accuracy_values['scaled_accuracy_B']
        }
        self.raw_accuracy_values = {
            'raw_accuracy_T': accuracy_values['raw_accuracy_T'],
            'raw_accuracy_B': accuracy_values['raw_accuracy_B']
        }

        if self.loss_values is None:
            print("Training was unsuccessful.")
            return

        print(f"Training completed with loss: {self.loss_values[-1]}")
        self.show_PINN_plots()

    def update_plots(self):
        t_idx = int(self.time_scale.get())
        t_idx = min(t_idx, self.T_arr_to_display.shape[1] - 1)

        # Update T(x) plot
        self.update_line_plot3(self.x_grid, self.T_arr_to_display[:, t_idx])

        # Update 3D surface plot
        self.update_surface_plot(self.x_grid, self.t_grid, self.T_arr_to_display)

        # Compute temperature from enthalpy for the current time step
        H_current = self.H_arr[:, t_idx]
        T_current = self.pcm.update_temperature(H_current, self.pcm)

        # Update T(E) plot
        self.update_line_plot(H_current, T_current)

        self.canvas1.draw()
        self.canvas2.draw()
        self.canvas3.draw()

    def show_PINN_plots(self, show_gold_standard=True, custom_cmap='coolwarm'):
        # Ensure the necessary data is available
        if self.Temperature_pred is None or self.Boundary_pred is None:
            print("Prediction data is not available.")
            return

        if show_gold_standard and (self.moving_boundary_locations is None or self.indices is None):
            print("Gold standard data is not set.")
            return

        # Create a new Tkinter window for the plots
        new_window = tk.Toplevel(self.root)
        new_window.title("PINN Plots")

        # Create a main frame for all widgets
        main_frame = tk.Frame(new_window)
        main_frame.grid(row=0, column=0)

        # Create a frame for the canvas where plots will be shown
        canvas_frame = tk.Frame(main_frame)
        canvas_frame.grid(row=0, column=0)

        # Create a frame for the toolbar (for navigation and tools)
        toolbar_frame = tk.Frame(main_frame)
        toolbar_frame.grid(row=1, column=0)

        # Create a canvas on which the plots will be drawn
        canvas = FigureCanvasTkAgg(self.fig_PINN, master=canvas_frame)
        canvas.get_tk_widget().grid(row=0, column=0)

        # Add a toolbar to the canvas frame
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()
        toolbar.pack()

        # Create subplots on the figure for different types of plots
        ax1 = self.fig_PINN.add_subplot(151)
        ax2 = self.fig_PINN.add_subplot(152)
        ax2a = self.fig_PINN.add_subplot(153)
        ax3 = self.fig_PINN.add_subplot(154, projection='3d')
        ax4 = self.fig_PINN.add_subplot(155)

        # Plot Loss and Accuracy
        ax1.plot(self.loss_values)
        ax1.set_title('Loss during Training')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')

        ax2.plot(self.scaled_accuracy_values['scaled_accuracy_T'], label='Scaled Accuracy T')
        ax2.plot(self.raw_accuracy_values['raw_accuracy_T'], label='Raw Accuracy T', linestyle='--')
        ax2.set_title('Accuracy T during Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()

        ax2a.plot(self.scaled_accuracy_values['scaled_accuracy_B'], label='Scaled Accuracy B')
        ax2a.plot(self.raw_accuracy_values['raw_accuracy_B'], label='Raw Accuracy B', linestyle='--')
        ax2a.set_title('Accuracy B during Training')
        ax2a.set_xlabel('Epoch')
        ax2a.set_ylabel('Accuracy')
        ax2a.legend()

        # Reshape and plot the temperature distribution
        x_dim, t_dim = len(self.x_grid), len(self.t_grid)
        Temperature_pred_reshaped = self.Temperature_pred.reshape((x_dim, t_dim))
        X, T = np.meshgrid(self.x_grid, self.t_grid, indexing='ij')
        ax3.plot_surface(X, T, Temperature_pred_reshaped, cmap=custom_cmap)
        ax3.set_title('Temperature Distribution (PINN)')
        ax3.set_xlabel('x')
        ax3.set_ylabel('t')
        ax3.set_zlabel('Temperature')

        # Reshape Boundary_pred to match the dimensions of x_grid and t_grid
        Boundary_pred_reshaped = self.Boundary_pred.reshape((x_dim, t_dim))
        # Select the last time step
        Boundary_last_time_step = Boundary_pred_reshaped[:, -1]

        # Now, plot the boundary prediction for the last time step
        ax4.plot(self.x_grid, Boundary_last_time_step, label='Predicted Boundary at Last Time Step')
        if show_gold_standard and self.moving_boundary_locations is not None:
            true_boundary_times = np.array(self.indices) * self.pcm.dt
            ax4.plot(true_boundary_times, self.moving_boundary_locations, label='Gold Standard Boundary',
                     linestyle='--')
        ax4.set_title('Boundary Location (PINN) at Last Time Step')
        ax4.set_xlabel('x')
        ax4.set_ylabel('Boundary Location')
        ax4.legend()

        # Draw the canvas with all the plots
        canvas.draw()

    def update_line_plot(self, H_arr, T_arr):
        self.line1.set_data(H_arr, T_arr)
        self.ax1.relim()
        self.ax1.autoscale_view()
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

        # Update the T(x) plot
        self.line3.set_ydata(self.T_arr_to_display[:, self.t_idx])
        self.ax3.relim()
        self.ax3.autoscale_view()
        self.canvas3.draw()

    def update_surface_plot(self, x_grid, t_arr_final, T_arr_to_display):
        self.ax2.cla()
        x_dim, t_dim = len(x_grid), len(t_arr_final)
        print(
            f"Debug: update_surface_plot - x_dim: {x_dim}, t_dim: {t_dim}, T_arr_to_display shape: {T_arr_to_display.shape}")

        try:
            if T_arr_to_display.shape[1] != t_dim:
                print(f"Warning: T_arr_to_display shape mismatch, reshaping to ({x_dim}, {t_dim})")
                T_arr_to_display = T_arr_to_display[:, :t_dim]

            T_reshaped = T_arr_to_display.reshape((x_dim, t_dim))
            X, T = np.meshgrid(x_grid, t_arr_final, indexing='ij')
            self.ax2.plot_surface(X, T, T_reshaped, cmap='coolwarm')
            self.ax2.set_title("3D Surface Plot")
            self.ax2.set_xlabel("x")
            self.ax2.set_ylabel("t")
            self.ax2.set_zlabel("T")
            self.canvas2.draw()

            print(
                f"Debug: update_surface_plot - X shape: {X.shape}, T shape: {T.shape}, T_reshaped shape: {T_reshaped.shape}")
        except Exception as e:
            print(f"Error in update_surface_plot: {e}")


    def run(self):
        self.root.mainloop()

    def on_closing(self):
        print("Main window is closing")
        self.root.destroy()


class GoldStandardSelectionWindow(tk.Toplevel):
    def __init__(self, parent, on_submit=None):
        super().__init__(parent)
        self.title("Select Gold Standard Solution")
        self.on_submit = on_submit  # Callback function from the parent

        # Exclude the PINN option from the selection
        self.solution_types = ["Analytical", "Enthalpy Method", "Implicit", "Numerical"]

        # Create a label
        label = tk.Label(self, text="Select a solution type to serve as a gold standard:")
        label.grid(row=0, column=0)

        # Create a StringVar for the dropdown
        self.selected_option = tk.StringVar(self)
        self.selected_option.set(self.solution_types[0])  # default value

        # Create the dropdown menu
        self.dropdown = tk.OptionMenu(self, self.selected_option, *self.solution_types)
        self.dropdown.grid(row=1, column=0)

        # Create submit button
        submit_button = tk.Button(self, text="Submit", command=self.submit)
        submit_button.grid(row=2, column=0)

    def submit(self):
        # Get the selected gold standard solution
        self.selected_solution = self.selected_option.get()

        # If a callback function was provided, call it with the selected solution
        if self.on_submit is not None:
            self.on_submit(self.selected_solution)

        # Close the window
        self.destroy()


if __name__ == '__main__':
    # logging.basicConfig(level=logging.INFO)
    # if int(os.environ.get("PRODUCTION", 0)) == 1:
    #     logging_client = google.cloud.logging.Client()
    #     logging_client.setup_logging()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    app = App()
    app.run()  # This line should be the last one in the main block
