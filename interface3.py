import os
import tkinter as tk
# import google
# import logging
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
        # print(f"Debug: User choice for input window = {choice}")

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
        self.root.geometry("1200x800")
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

    def handle_gold_standard_selection(self, selected_solution):
        # Update the application state with the selected solution
        self.gold_standard = selected_solution

        # Set the corresponding temperature array based on the selected gold standard
        self.set_gold_standard_temp_array(self.gold_standard)

        # Now, prepare and start the PINN calculations
        self.prepare_PINN_model_and_train()

    def set_gold_standard_temp_array(self, selected_solution):
        self.gold_standard_temp_array = self.temperature_solutions.get(selected_solution, None)

    def prompt_for_gold_standard(self):
        # Create a new window for the user to select the gold standard
        self.gold_standard_window = GoldStandardSelectionWindow(self.root, self.handle_gold_standard_selection)
        self.gold_standard_window.mainloop()  # This will now call handle_gold_standard_selection when the user submits

    def handle_solution_type_change(self, *args):
        selected_solution = self.solution_type.get()
        if selected_solution == "PINN":
            # Prompt the user to select the gold standard solution type
            self.prompt_for_gold_standard()
        else:
            # Handle other solution types normally
            self.update_solution_type()
            self.update_plots()

    def calcAll(self):
        # Generate data
        self.x, self.y_T, self.y_B, self.x_boundary, self.x_grid, self.t_grid = self.pcm.generate_data(self.L,
                                                                                                            self.t_max)

        # Adjust dimensions of x to align with y_T, y_B for PINN
        self.x = np.squeeze(self.x)

        # Reshape y_T_flat to 2D for traditional methods, assuming y_T_flat is ordered by x then t
        nx = len(self.x_grid)
        nt = len(self.t_grid)
        y_T_2D = self.y_T.reshape(nx, nt)

        # Now that y_T_2D is in the correct shape, compute mask arrays
        self.mask_array_T, self.mask_array_B = compute_mask_arrays(y_T_2D, self.pcm)
        # print("Debug: Shape of flat_mask_array_T:", flat_mask_array_T.shape)
        # print("Debug: Shape of flat_mask_array_B:", flat_mask_array_B.shape)

        # # Reshape the flat mask arrays to 2D for use in traditional methods
        # self.mask_array_T = flat_mask_array_T.reshape(len(self.x_grid), nt)
        # self.mask_array_B = flat_mask_array_B.reshape(len(self.x_grid), nt)

        # Store initial and boundary conditions for PINN
        self.initial_data_T = (self.x, self.y_T)
        self.initial_data_B = (self.x_boundary, self.y_B)
        # self.mask_array_T_incremental = np.zeros((self.y_T.shape[0], 1))
        self.T_arr_numerical, self.temp_mask_array_numerical, self.bound_mask_array_numerical = \
            self.pcm.explicitNumerical(self.x_grid, self.t_max, self.pcm, phase_mask_array=self.mask_array_T,
                                      boundary_mask_array=self.mask_array_B)

        self.T_arr_analytical, self.temp_mask_array_analytical, self.bound_mask_array_analytical = \
            self.pcm.analyticalSol(self.x_grid, self.t_grid, self.pcm, phase_mask_array=self.mask_array_T,
                                 bound_mask_array=self.mask_array_B)

        self.T_arr_implicit, self.H_arr_final, self.t_arr4, self.temp_mask_array_implicit, self.bound_mask_array_implicit = \
            self.pcm.implicitSol(self.x_grid, self.t_max, self.pcm, phase_mask_array=self.mask_array_T,
                                 boundary_mask_array=self.mask_array_B)

        self.T_arr_enth, self.H_arr_enth, self.temp_mask_array_stefan, self.bound_mask_array_stefan = \
            self.pcm.solve_stefan_problem_enthalpy(self.pcm, self.L, self.t_max, phase_mask_array=self.mask_array_T,
                                                   boundary_mask_array=self.mask_array_B)
        print(f"T_arr_enth = {self.T_arr_enth}")
        print(f"H_arr_enth = {self.H_arr_enth}")
        print(f"temp_mask_array_stefan = {self.temp_mask_array_stefan}")
        print(f"bound_mask_array_stefan = {self.bound_mask_array_stefan}")
        # self.H_arr_final = self.H_arr_enth
        # print(f"Debug: self.x_grid length = {len(self.x_grid)}")
        # print(f"Debug: self.t_grid length = {len(self.t_grid)}")
        # Store temperature arrays for each solution type
        self.temperature_solutions = {
            "Analytical": self.T_arr_analytical,
            "Numerical": self.T_arr_numerical,
            "Implicit": self.T_arr_implicit,
            "Enthalpy Method": self.T_arr_enth,
            # Add other solution types if necessary
        }

        moving_boundary_indices = self.pcm.calculate_moving_boundary_indices(self.T_arr_analytical, self.pcm.T_m)
        print("Debug: Moving boundary indices = ", moving_boundary_indices)
        # Convert indices to actual spatial locations if needed
        dx = self.pcm.dx

        for i, val in enumerate(moving_boundary_indices):
            if val is not None:
                self.moving_boundary_locations.append(val * dx)
                self.indices.append(i)

        self.moving_boundary_locations = np.array(self.moving_boundary_locations)
        self.indices = np.array(self.indices)
        # self.mask_array = compute_mask_array(self.x_grid, self.L)
        self.update_solution_type()
        self.update_plots()


    def update_solution_type(self, *args):

        selected_solution = self.solution_type.get()

        # print(f"Debug: Selected solution type = {selected_solution}")
        self.t_arr_final = self.t_grid
        if selected_solution == "Analytical":
            self.T_arr_to_display = self.T_arr_analytical

            self.temp_mask_array = self.temp_mask_array_analytical
            self.bound_mask_array = self.bound_mask_array_analytical
        elif selected_solution == "Numerical":
            self.T_arr_to_display = self.T_arr_numerical
            self.temp_mask_array = self.temp_mask_array_numerical
            self.bound_mask_array = self.bound_mask_array_numerical
        elif selected_solution == "Implicit":
            self.T_arr_to_display = self.T_arr_implicit
            self.t_arr_final = self.t_arr4
            self.temp_mask_array = self.temp_mask_array_implicit
            self.bound_mask_array = self.bound_mask_array_implicit
        elif selected_solution == "Enthalpy Method":
            self.T_arr_to_display = self.T_arr_enth
            self.temp_mask_array = self.temp_mask_array_stefan
            self.bound_mask_array = self.bound_mask_array_stefan
        elif selected_solution == 'PINN':
            # Prompt the user to select a gold standard solution type
            gold_standard_window = GoldStandardSelectionWindow(self.root)
            gold_standard_window.mainloop()
            gold_standard_solution = gold_standard_window.selected_solution

            # Check if a gold standard was selected
            if gold_standard_solution:
                print(f"Selected gold standard solution type: {gold_standard_solution}")

                # Set the gold standard temperature array based on the user's choice
                self.set_gold_standard_temp_array(gold_standard_solution)

                # Call the method to prepare and start PINN calculations
                self.prepare_PINN_model_and_train()

                # Update the 'to' parameter of the timescale widget and plots
                self.time_scale.config(to=self.t_arr_final[-1])

            else:
                print("No gold standard solution was selected. PINN training aborted.")



    def prepare_PINN_model_and_train(self):
        # Validate the necessary components
        if self.gold_standard_temp_array is None:
            print("Gold standard data is not set.")
            return

        # Reshape inputs for the model
        self.x_input = self.x.reshape(-1, 2)  # Flatten x for model input
        self.x_boundary_input = self.x_boundary.reshape(-1, 2)  # Flatten x_boundary for model input

        self.boundary_indices = self.pcm.calculate_boundary_indices(
            self.x, self.L, self.pcm.dt, T=self.gold_standard_temp_array, T_m=self.pcm.T_m, mode='initial')

        # Define the batch size for training
        self.batch_size = 64

        # Initialize the PINN model
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
            moving_boundary_locations=self.moving_boundary_locations
        )

        # Call the train_PINN function to train the model
        self.loss_values, self.accuracy_values, self.Temperature_pred, self.Boundary_pred = train_PINN(
            model=self.model,
            x=self.x_input,
            x_boundary=self.x_boundary_input,
            y_T=self.y_T,
            y_B=self.y_B,
            T_arr_display=self.gold_standard,
            pcm=self.pcm,
            epochs=25,
            mask_T=self.mask_array_T.ravel(),
            mask_B=self.mask_array_B.ravel(),
            batch_size=self.batch_size
        )

        # Check if training was successful before proceeding
        if self.loss_values is None:
            print("Training was unsuccessful.")
            return

        # Print or plot results as needed
        print(f"Training completed with loss: {self.loss_values[-1]}")
        self.show_PINN_plots()

    def update_plots(self):

        # Get the current time index from the timescale widget
        t_idx = int(self.time_scale.get())
        t_idx = t_idx if t_idx < self.T_arr_to_display.shape[1] else self.T_arr_to_display.shape[1] - 1

        # Update line plot 3 using the selected time index
        self.update_line_plot3(self.x_grid, self.T_arr_to_display[:, t_idx])
        # if self.H_arr_final is None:
        #     self.H_arr_final = self.pcm.calcEnthalpy2(self.T_arr_to_display, self.pcm)

        # print("H values from calcEnthalpy2:", self.H_arr_final)

        self.energy_sufficiency = self.pcm.calcEnergySufficiency(self.H_arr_final)
        self.update_line_plot(self.H_arr_final[:, t_idx], self.T_arr_to_display[:, t_idx])
        self.update_surface_plot(self.x_grid, self.t_arr_final, self.T_arr_to_display)
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

        ax2.plot(self.accuracy_values['accuracy_T'])
        ax2.set_title('Accuracy T during Training')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')

        ax2a.plot(self.accuracy_values['accuracy_B'])
        ax2a.set_title('Accuracy B during Training')
        ax2a.set_xlabel('Epoch')
        ax2a.set_ylabel('Accuracy')

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
        Boundary_last_time_step = Boundary_pred_reshaped[:,
                                  -1]  # This selects the last column, representing the last time step

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
