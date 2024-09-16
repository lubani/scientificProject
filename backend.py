import math
from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import splu
from tensorflow.keras.layers import Flatten  # Use the Flatten layer from tensorflow.keras
from scipy.linalg import lu_factor, lu_solve
from scipy.special import erf
from scipy.sparse import lil_matrix, csc_matrix



# def compute_mask_arrays(T, cls, tolerance=100, phase_mask=None, boundary_mask=None):
#     if phase_mask is None:
#         phase_mask = np.zeros_like(T, dtype=int)
#     if boundary_mask is None:
#         boundary_mask = np.zeros_like(T, dtype=int)
#
#     T_minus = cls.T_m - tolerance
#     T_plus = cls.T_m + tolerance
#
#     # Initialize phase mask
#     phase_mask[:] = 0
#     phase_mask[T < T_minus] = 0  # Solid phase
#     phase_mask[(T >= T_minus) & (T <= T_plus)] = 1  # Phase transition (melting/freezing)
#     phase_mask[T > T_plus] = 2  # Liquid phase
#
#     # Initialize boundary mask
#     boundary_mask[:] = 0
#     boundary_mask[phase_mask == 1] = 1  # Mark where phase change occurs
#
#     # Debugging information
#     # print(f"T_minus: {T_minus}, T_plus: {T_plus}")
#     # print(f"phase_mask: {np.unique(phase_mask, return_counts=True)}")
#     # print(f"boundary_mask: {np.unique(boundary_mask, return_counts=True)}")
#
#     return phase_mask, boundary_mask
def compute_mask_arrays(T, H, cls, tolerance=10, phase_mask=None, boundary_mask=None):
    if phase_mask is None:
        phase_mask = np.zeros_like(T, dtype=int)
    if boundary_mask is None:
        boundary_mask = np.zeros_like(T, dtype=int)

    T_minus = cls.T_m - tolerance
    T_plus = cls.T_m + tolerance

    # Calculate the latent heat enthalpy value
    H_latent = cls.LH * cls.rho

    # Initialize phase mask based on enthalpy and temperature
    phase_mask[:] = 0
    phase_mask[
        (H >= H_latent - tolerance) & (H <= H_latent + tolerance) & (T >= T_minus) & (T <= T_plus)] = 1  # Mushy zone
    phase_mask[H > H_latent + tolerance] = 2  # Liquid phase
    phase_mask[H < H_latent - tolerance] = 0  # Solid phase

    # Initialize boundary mask based on phase change boundaries
    boundary_mask[:] = 0
    boundary_mask[phase_mask == 1] = 1  # Mark where phase change occurs

    # Debugging prints to check mask values
    print(f"compute_mask_arrays: phase_mask counts = {np.bincount(phase_mask.flatten())}")
    print(f"compute_mask_arrays: boundary_mask counts = {np.bincount(boundary_mask.flatten())}")

    return phase_mask, boundary_mask


class PCM(ABC):

    @abstractmethod
    def calcThermalConductivity(self, temp):
        pass

    @abstractmethod
    def calcSpecificHeat(self, temp):
        pass

    @abstractmethod
    def generate_data(self, x_max, t_max):
        pass

    @abstractmethod
    def implicitSol(self, x_arr, t_arr, T_arr, H_arr, cls, phase_mask_array=None, boundary_mask_array=None):
        pass

    @abstractmethod
    def explicitNumerical(self, x_arr, t_arr, T_arr, cls, phase_mask_array=None, boundary_mask_array=None):
        pass

    @abstractmethod
    def heat_source_function(self, x, t, cycle_duration, heat_source_max):
        pass

    def alpha(self, k, c, rho):
        # Check for division by zero or small number
        if c * rho == 0 or k == 0:
            raise ValueError("c * rho should not be zero")
        return k / (c * rho)

    def solve_stefan_problem_enthalpy(self, cls, x_arr, t_arr, T_arr, H_arr, temp_mask_array=None,
                                      bound_mask_array=None):
        print("Debug: Entering solve_stefan_problem_enthalpy")

        if temp_mask_array is None or bound_mask_array is None:
            temp_mask_array = np.zeros((len(x_arr), len(t_arr)), dtype=int)
            bound_mask_array = np.zeros((len(x_arr), len(t_arr)), dtype=int)

        boundary_indices = np.full(len(t_arr), -1, dtype=int)

        # Initialize temperature with analytical solution for smooth start
        T_initial = self.analyticalSol(x_arr, t_arr[:1], cls)[:, 0]
        T_arr[:, 0] = T_initial
        H_arr[:, 0] = cls.calcEnthalpy2(T_initial, cls)

        for t_idx in range(1, len(t_arr)):
            H_old = H_arr[:, t_idx - 1]
            T_old = T_arr[:, t_idx - 1]

            # Calculate thermal properties
            k_vals = np.array([cls.calcThermalConductivity(T_old[i]) for i in range(len(x_arr))])
            c_vals = np.array([cls.calcSpecificHeat(T_old[i]) for i in range(len(x_arr))])
            alpha_vals = k_vals / (cls.rho * c_vals)
            lmbda_vals = cls.dt / cls.dx ** 2 * alpha_vals

            # Apply heat source term
            heat_source = np.array([cls.heat_source_function(x, t_arr[t_idx]) for x in x_arr])

            # Update enthalpy considering heat source
            H_new_internal = H_old[1:-1] + (cls.dt / cls.dx ** 2) * (
                    lmbda_vals[1:-1] * (H_old[2:] - H_old[1:-1]) - lmbda_vals[:-2] * (H_old[1:-1] - H_old[:-2])
            ) + cls.dt * heat_source[1:-1]

            # Apply boundary conditions to H_new_internal
            H_new = np.zeros_like(H_old)
            H_new[1:-1] = H_new_internal
            H_new[0] = cls.T_m + 10  # Set the boundary condition at the first point
            H_new[-1] = H_old[-1]  # Keep the last value as in the previous time step

            # Update temperature from the new enthalpy
            T_new = self.update_temperature(H_new, cls)

            # Store the new values in the arrays
            H_arr[:, t_idx] = H_new
            T_arr[:, t_idx] = T_new

            # Update phase mask based on the new temperature
            temp_mask_array[:, t_idx], bound_mask_array[:, t_idx] = compute_mask_arrays(T_new, H_new, cls)

            # Calculate the boundary index for the current time step
            phase_change_indices = np.where(np.abs(T_new - cls.T_m) <= 100)[0]  # Adjust tolerance as needed
            if phase_change_indices.size > 0:
                boundary_indices[t_idx] = phase_change_indices[0]

        # print(f"Debug: solve_stefan_problem_enthalpy - T_arr shape: {T_arr.shape}, H_arr shape: {H_arr.shape}")
        return T_arr, H_arr, temp_mask_array, bound_mask_array, boundary_indices

    def update_gamma(self, cls, temp, dt_current):
        k_max = np.max([cls.calcThermalConductivity(t) for t in temp])
        c_max = np.max(cls.calcSpecificHeat(temp))
        alpha_max = k_max / (cls.rho * c_max)
        gamma = alpha_max * dt_current / cls.dx ** 2
        return gamma

    def initialize_enthalpy_temperature_arrays(self, x_arr, cls, t_steps):
        # Initialize the temperature array
        T = np.ones((len(x_arr), t_steps)) * cls.T_a  # Start all cells at ambient temperature
        T[0, :] = cls.T_m + 10  # Set the first spatial cell at the first time step to T_m + 10

        # Introduce a slight gradient or perturbation to the initial temperature array
        # for i in range(1, len(x_arr)):
        #     T[i, 0] = cls.T_a + np.random.uniform(0, 5)  # Add a small random perturbation

        # Initialize the enthalpy array
        H = self.initial_enthalpy(x_arr, cls, t_steps)
        return T, H

    def calculate_dt(self, cls, max_k=None, safety_factor=0.4, dt_multiplier=1.0):
        # Determine max_k based on the melting temperature if not provided
        if max_k is None:
            max_k = max(cls.calcThermalConductivity(cls.T_m), cls.calcThermalConductivity(cls.T_a))

        max_alpha = max_k / (cls.rho * max(cls.c_solid, cls.c_liquid))
        max_dt = (safety_factor * cls.dx ** 2) / (max_alpha * 2)

        # Apply the multiplier to adjust dt
        calculated_dt = max_dt * dt_multiplier
        print(f"Calculated dt = {calculated_dt}")

        return calculated_dt

    def update_enthalpy_temperature(self, H_current, cls, gamma, x_arr):
        dH = gamma * (np.roll(H_current, -1) - 2 * H_current + np.roll(H_current, 1))
        H_next = H_current + dH
        H_next[0] = H_current[0]  # Reapply left boundary condition
        H_next[-1] = H_current[-1]  # Reapply right boundary condition
        T_next = self.update_temperature(H_next, cls)
        return H_next, T_next

    def update_temperature(self, H_new, cls):
        T_new = np.zeros_like(H_new)
        H_m = cls.calcEnthalpy2(cls.T_m, cls)  # Enthalpy at melting temperature
        delta_H = cls.LH / 10  # Smoothing region around the phase transition

        for i in range(len(H_new)):
            if H_new[i] <= 0:
                T_new[i] = cls.T_a  # Ensure temperature is not below ambient
            elif H_new[i] <= H_m - delta_H:
                T_new[i] = max(H_new[i] / (cls.rho * cls.c_solid), cls.T_a)
            elif H_new[i] <= H_m + cls.LH + delta_H:
                # Smooth transition around the phase change region
                T_new[i] = cls.T_m + (H_new[i] - H_m) * (1 / (cls.rho * cls.c_liquid))
            else:
                T_new[i] = (H_new[i] - cls.LH - H_m) / (cls.rho * cls.c_liquid) + cls.T_m

        print(f"Debug: Updated temperatures T_new: {T_new}")
        return T_new

    def update_phase_mask(self, temperature_array, cls):
        tolerance = 10.0  # Wider tolerance around the melting point

        # Phase mask: 0 for solid, 1 for liquid, 2 for mushy zone
        phase_mask = np.select(
            [temperature_array < cls.T_m - tolerance, temperature_array > cls.T_m + tolerance],
            [0, 1],  # 0 for solid, 1 for liquid
            default=2  # 2 for mushy zone (near melting point)
        )

        # Boundary mask: Detect sharp temperature gradients indicating a phase boundary
        gradient = np.gradient(temperature_array)
        gradient_threshold = 10  # Lower threshold for detecting smoother changes in temperature
        boundary_mask = np.where(np.abs(gradient) > gradient_threshold, 1, 0)

        return phase_mask, boundary_mask

    def calculate_boundary_indices(self, x, x_max, dt, T=None, T_m=None, tolerance=100, mode='initial', atol=1e-8,
                                   rtol=1e-5):
        if mode == 'initial':
            boundary_indices = {'condition1': [], 'condition2': []}
            dt_indices = np.isclose(x[:, 1], dt, atol=atol, rtol=rtol)
            boundary_indices['condition1'] = np.where(dt_indices & np.isclose(x[:, 0], 0, atol=atol, rtol=rtol))[0]
            boundary_indices['condition2'] = np.where(dt_indices & ~np.isclose(x[:, 0], x_max, atol=atol, rtol=rtol))[0]
            return boundary_indices

        elif mode == 'moving_boundary':
            if T is None or T_m is None:
                raise ValueError(
                    "Temperature array T and melting point T_m must be provided for 'moving_boundary' mode.")

            moving_boundary_indices = np.full(T.shape[1], -1, dtype=int)

            for n in range(T.shape[1]):
                phase_change_indices = np.where(np.abs(T[:, n] - T_m) <= tolerance)[0]
                if phase_change_indices.size > 0:
                    moving_boundary_indices[n] = phase_change_indices[0]

            return moving_boundary_indices

        else:
            raise ValueError("Invalid mode. Choose between 'initial' and 'moving_boundary'.")

    def calculate_moving_boundary_indices(self, T_arr, T_m):
        boundary_indices = np.full(T_arr.shape[1], -1, dtype=int)
        for t_idx in range(T_arr.shape[1]):
            phase_change_indices = np.where(np.abs(T_arr[:, t_idx] - T_m) < 1e-2)[0]
            if phase_change_indices.size > 0:
                boundary_indices[t_idx] = phase_change_indices[0]
            else:
                boundary_indices[t_idx] = -1
            # print(f"Debug: Time step {t_idx} - phase_change_indices: {phase_change_indices}")
            # print(f"Debug: Time step {t_idx} - boundary_indices: {boundary_indices[t_idx]}")
        return boundary_indices

    def inverseEnthalpy2(self, H, cls):
        T = np.full_like(H, cls.T_m)
        T[H < 0] = cls.T_a + H[H < 0] / (cls.rho * cls.c)
        T[H > cls.LH] = cls.T_a + (H[H > cls.LH] - cls.LH) / (cls.rho * cls.c)
        return T

    def initial_enthalpy(self, x_arr, cls, t_steps):
        H_arr = np.zeros((len(x_arr), t_steps), dtype=np.float64)
        for i, x in enumerate(x_arr):
            T_initial = cls.T_a if i != 0 else cls.T_m + 10  # Initial temperature condition
            if T_initial < cls.T_m:
                initial_H = cls.rho * cls.c_solid * (T_initial - cls.T_a)
            elif T_initial == cls.T_m:
                initial_H = 0  # At the melting point, no sensible heat change, only phase change
            else:
                initial_H = cls.LH + cls.rho * cls.c_liquid * (T_initial - cls.T_m)
            H_arr[i, :] = initial_H
        return H_arr

    def calcEnthalpy2(self, T, cls, epsilon=0.01, smoothing='h'):
        T_minus = cls.T_m - epsilon
        T_plus = cls.T_m + epsilon

        def smoothed_enthalpy(T):
            return cls.rho * cls.c * (T_minus - cls.T_a) + ((cls.LH / (2 * epsilon)) * (T - T_minus))

        if smoothing == 'erf':
            # Use error function smoothing
            eta = 0.5 * (1 + erf((T - cls.T_m) / (np.sqrt(2) * epsilon)))
            H = eta * (cls.LH + cls.rho * cls.c * (T - cls.T_m)) + (1 - eta) * cls.rho * cls.c * (T - cls.T_a)
        elif smoothing == 'linear':
            # Use linear smoothing
            H = np.where(T < T_minus, cls.rho * cls.c * (T - cls.T_a),
                         np.where(T > T_plus, cls.LH + cls.rho * cls.c * (T - cls.T_m),
                                  smoothed_enthalpy(T)))
        elif smoothing == 'h':
            # Use h-smoothing method
            H = np.where(T < T_minus, cls.rho * cls.c * (T - cls.T_a),
                         np.where(T > T_plus, cls.LH + cls.rho * cls.c * (T - cls.T_m),
                                  smoothed_enthalpy(T)))
        else:
            raise ValueError("Unknown smoothing method: choose 'erf', 'linear', or 'h'.")

        return H

    def calcEnthalpy(self, x_arr, t_max, cls):
        num_points = len(x_arr)
        T = np.full(num_points, cls.T_a)  # temperature array
        c = np.array([self.calcSpecificHeat(temp) for temp in T])
        H = c * T  # enthalpy array
        H = np.array(H, dtype=np.float64)
        c = np.array(c, dtype=np.float64)

        t_vals = np.linspace(cls.dt, t_max, num_points)  # Time values

        # Time evolution
        for t in range(len(t_vals) - 1):
            # Build the system of equations for the backward Euler method
            A = np.eye(num_points)
            b = np.copy(H)

            for i in range(1, num_points - 1):  # interior points
                k = self.calcThermalConductivity(T[i])
                A[i, i - 1] = -cls.dt * k / (cls.rho * cls.dx ** 2)
                A[i, i] += 2 * cls.dt * k / (cls.rho * cls.dx ** 2)
                A[i, i + 1] = -cls.dt * k / (cls.rho * cls.dx ** 2)

            # Solve the system of equations
            H_new = np.linalg.solve(A, b)

            # Calculate temperature from enthalpy
            for i in range(len(H_new)):
                if H_new[i] < c[i] * cls.T_m:
                    T[i] = H_new[i] / c[i]
                elif H_new[i] < c[i] * cls.T_m + cls.LH:
                    T[i] = cls.T_m
                else:
                    T[i] = (H_new[i] - cls.LH) / c[i]

            H = H_new

        return H, T

    def calcEnergySufficiency(self, H_vals):
        E_vals = H_vals  # Enthalpy is the energy in kJ

        P_settlement = 50  # Power requirement for the settlement in kW
        time_hours = 14.75 * 24  # Half of a lunar day-night cycle (only day or night) in hours

        # Energy needed for the settlement over the adjusted lunar day-night cycle directly in kJ
        E_settlement = P_settlement * time_hours * 3.6e3  # Convert kWh to kJ

        # Total energy stored in the regolith over the adjusted cycle, assuming E_vals is already in kJ
        E_regolith_total = np.sum(E_vals)  # Directly use the values without further conversion

        # Formulate the result string
        result_string = f"Energy needed: {E_settlement:.0f} kJ, Energy calculated: {E_regolith_total:.0f} kJ. "
        if E_regolith_total >= E_settlement:
            result_string += "The thermal energy in the regolith is sufficient to power the settlement."
        else:
            result_string += "The thermal energy in the regolith is not sufficient to power the settlement."

        return result_string

    def analyticalSol(self, x_val, t_arr, cls):
        T_initial = cls.T_a
        T_final = cls.T_m
        T = np.full((len(x_val), len(t_arr)), T_initial, dtype=np.float64)
        T[0, :] = T_final

        for t_idx, t_val in enumerate(t_arr):
            if t_val > 0:
                alpha2 = self.alpha(cls.k, cls.c, cls.rho)
                x_term = x_val / (2 * np.sqrt(alpha2 * t_val))
                T[:, t_idx] = T_initial + (T_final - T_initial) * (1 - erf(x_term))
            else:
                T[:, t_idx] = T_initial

        return T


class customPCM(PCM):
    T_m, LH, k, c, rho = None, None, None, None, None

    def __init__(self, k, c, rho, T_m, LH):
        customPCM.k, customPCM.c, customPCM.rho, customPCM.T_m, customPCM.LH = k, c, rho, T_m, LH
        self.dx = 0.1
        self.alpha2 = self.alpha(customPCM.k, customPCM.c, customPCM.rho)
        self.dt = (0.4 * self.dx ** 2) * self.alpha2
        print(f'alpha = {self.alpha2}')
        self.lmbda = self.dt / self.dx ** 2

    def calcThermalConductivity(self, temp):
        return customPCM.k

    def calcSpecificHeat(self, temp):
        return customPCM.c


class Regolith(PCM):
    T_m = 1373  # Melting temperature in Kelvin
    LH = 1429   # Latent heat in J/kg
    rho = 1.8   # Density in kg/m³
    T_a = 253   # Ambient temperature in Kelvin
    cycle_duration = 708  # Lunar cycle duration

    def __init__(self):
        self.dx = 0.15  # Spatial step

        # Calculate specific heats for solid and liquid states
        self.c_solid = self.calcSpecificHeat(Regolith.T_a)
        self.c_liquid = self.calcSpecificHeat(Regolith.T_m)

        # Calculate thermal conductivity at melting temperature
        self.k = self.calcThermalConductivity(Regolith.T_m)

        # Calculate dt using the calculate_dt method
        self.dt = self.calculate_dt(cls=self, safety_factor=0.4, dt_multiplier=1.0)

        # Use specific heat for the molten state in further calculations
        self.c = self.c_liquid
        self.alpha2 = self.alpha(self.k, self.c, Regolith.rho)
        self.lmbda = self.dt / (self.dx ** 2)

        # Set the solar incidence angle
        self.solar_incidence_angle = 45  # degrees

        # Define heat_source_max as an attribute for Regolith
        self.heat_source_max = 100

        print(f"Calculated dt = {self.dt}")
        print(f'alpha = {self.alpha2}')

    def generate_data(self, x_max, t_max):
        self.x_max = x_max  # Save x_max as an instance attribute
        x_grid = np.arange(0, x_max, self.dx)
        t_grid = np.arange(self.dt, t_max, self.dt)  # Exclude the final time step

        # Ensure t_grid is not empty
        if len(t_grid) == 0:
            raise ValueError("t_grid has no elements. Ensure t_max > dt and dt is reasonably small.")

        X, T = np.meshgrid(x_grid, t_grid, indexing='ij')
        x_features = np.column_stack([X.ravel(), T.ravel()])

        # Initialize temperature (y_T) and boundary (y_B) arrays
        y_T = np.full(x_features.shape[0], self.T_a, dtype=np.float64)  # Ambient temperature for all
        y_B = np.zeros_like(y_T, dtype=np.float64)  # Initialize boundary array as zeros
        x_boundary = np.zeros_like(x_features, dtype=np.float64)  # Initialize boundary features as zeros

        # Set initial boundary condition at x=0 (e.g., higher temperature at the boundary)
        boundary_condition_indices = x_features[:, 0] == 0
        y_T[boundary_condition_indices] = self.T_m + 10.0  # Temperature at the boundary
        y_B[boundary_condition_indices] = 1.0  # Mark boundary location

        # Use provided methods to initialize temperature and enthalpy arrays
        T_arr, H_arr = self.initialize_enthalpy_temperature_arrays(x_grid, self, len(t_grid))

        # Debugging prints to visualize initial conditions
        print(f"Initial T_arr shape: {T_arr.shape}")
        print(f"Initial H_arr shape: {H_arr.shape}")
        print(f"Initial boundary condition indices: {np.sum(boundary_condition_indices)}")

        return x_features, y_T, y_B, x_boundary, x_grid, t_grid, T_arr, H_arr

    def explicitNumerical(self, x_arr, t_arr, T_arr, cls, phase_mask_array=None, boundary_mask_array=None):
        nx = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None or boundary_mask_array is None:
            phase_mask_array = np.zeros((nx, num_timesteps), dtype=int)
            boundary_mask_array = np.zeros((nx, num_timesteps), dtype=int)

        alpha = cls.alpha(cls.k, cls.c, cls.rho)
        max_increase = 1000.0  # Maximum temperature increase allowed
        max_temperature = 4000.0  # Maximum temperature allowed in the system
        moving_boundary_indices = np.full(num_timesteps, -1, dtype=int)

        # Initial conditions
        T_arr[:, 0] = cls.T_a
        T_arr[0, :] = cls.T_m + 10.0

        for timestep in range(1, num_timesteps):
            T_old = T_arr[:, timestep - 1]

            # Compute diffusive term with proper boundary handling
            diffusive_term = np.zeros_like(T_old)
            diffusive_term[1:-1] = T_old[2:] - 2 * T_old[1:-1] + T_old[:-2]
            # Apply boundary conditions
            diffusive_term[0] = 0
            diffusive_term[-1] = 0

            T_new = T_old + (alpha * cls.dt / cls.dx ** 2) * diffusive_term

            # Update temperature at the boundary with heat source
            heat_source = cls.heat_source_function(0, t_arr[timestep], cls.cycle_duration, heat_source_max=10)
            if heat_source > 0:
                T_new[0] += min(heat_source / (cls.rho * cls.c), max_increase)

            T_new[-1] = T_old[-1]  # Keep the last cell's temperature constant

            # Clip temperatures to prevent numerical issues
            T_new = np.clip(T_new, cls.T_a, max_temperature)

            T_arr[:, timestep] = T_new

            # Update phase and boundary masks
            phase_mask, boundary_mask = self.update_phase_mask(T_new, cls)
            phase_mask_array[:, timestep] = phase_mask
            boundary_mask_array[:, timestep] = boundary_mask

            # Update moving boundary indices
            moving_boundary_indices[timestep] = self.calculate_boundary_indices(
                x=x_arr,
                x_max=x_arr[-1],
                dt=cls.dt,
                T=T_arr,
                T_m=cls.T_m,
                mode='moving_boundary',
                tolerance=100
            )[timestep]

        return T_arr, phase_mask_array, boundary_mask_array, moving_boundary_indices

    def implicitSol(self, x_arr, t_arr, T_arr, H_arr, cls, phase_mask_array=None, boundary_mask_array=None):
        num_segments = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None:
            phase_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)
        if boundary_mask_array is None:
            boundary_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)

        moving_boundary_indices = np.full(num_timesteps, -1, dtype=int)

        # Initial condition: Set the first time step
        T_arr[:, 0] = cls.T_a  # Ambient temperature for all spatial cells
        T_arr[0, :] = cls.T_m + 10.0  # Higher temperature at the first spatial cell

        for time_step in range(1, num_timesteps):
            T_old = T_arr[:, time_step - 1]

            # Heat source function specific to regolith
            heat_source = cls.heat_source_function(x_arr, t_arr[time_step], cls.cycle_duration, cls.heat_source_max)

            # Explicit update for the nonlinear heat source term
            T_explicit = T_old + cls.dt * heat_source / (cls.rho * cls.c)

            # Calculate thermal properties
            k_vals = np.array([cls.calcThermalConductivity(T_old[i]) for i in range(num_segments)])
            c_vals = np.array([cls.calcSpecificHeat(T_old[i]) for i in range(num_segments)])
            rho = cls.rho
            alpha_vals = k_vals / (c_vals * rho)
            lmbda_vals = cls.dt / (cls.dx ** 2) * alpha_vals

            # Construct tridiagonal matrix A for the implicit diffusion term
            A = lil_matrix((num_segments, num_segments))
            A.setdiag(1 + 2 * lmbda_vals)
            A.setdiag(-lmbda_vals[1:], -1)
            A.setdiag(-lmbda_vals[:-1], 1)

            # Apply boundary conditions
            A[0, :] = 0
            A[0, 0] = 1

            # Do not fix the temperature at the rightmost cell, allowing it to evolve
            A[-1, -2] = -lmbda_vals[-1]  # Adjust the last cell in the matrix A
            A[-1, -1] = 1 + lmbda_vals[-1]

            # Convert A to CSC format for efficient solving
            A = A.tocsc()

            try:
                # Solve the system using LU decomposition
                lu = splu(A)
                T_new = lu.solve(T_explicit)

                # Calculate new enthalpy from updated temperature
                H_new = cls.calcEnthalpy2(T_new, cls)
                T_arr[:, time_step] = T_new
                H_arr[:, time_step] = H_new

                # Compute masks
                phase_mask, boundary_mask = self.update_phase_mask(T_new, cls)
                phase_mask_array[:, time_step] = phase_mask
                boundary_mask_array[:, time_step] = boundary_mask

                moving_boundary_indices[time_step] = self.calculate_boundary_indices(
                    x_arr, self.x_max, cls.dt, T=T_arr, T_m=cls.T_m, mode='moving_boundary', tolerance=100
                )[time_step]
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        return T_arr, H_arr, phase_mask_array, boundary_mask_array, moving_boundary_indices

    # def calcThermalConductivity(self, temp):
    #     C1 = 1.281e-2
    #     C2 = 4.431e-4
    #     epsilon = 1e-6  # Small value to avoid zero division
    #
    #     if isinstance(temp, (list, np.ndarray)):
    #         k_granular = C1 + C2 * (np.array(temp, dtype=np.float64) + epsilon) ** (-3)
    #     else:
    #         k_granular = C1 + C2 * (float(temp) + epsilon) ** (-3)
    #
    #     k_molten_end = 2.5  # Thermal conductivity for molten regolith
    #
    #     if isinstance(temp, (list, np.ndarray)):
    #         k_final = np.where(temp < self.T_m, k_granular, k_molten_end)
    #     else:
    #         k_final = k_granular if temp < self.T_m else k_molten_end
    #
    #     return 1000.0 * k_final

    # def calcSpecificHeat(self, temp):
    #     specific_heat = -1848.5 + 1047.41 * np.log(temp)
    #     return specific_heat

    def calcThermalConductivity(self, temp):
        k_solid = 0.01  # W/m·K for solid regolith
        k_molten = 2.5  # W/m·K for molten regolith

        if isinstance(temp, (list, np.ndarray)):
            k_final = np.where(temp < self.T_m, k_solid, k_molten)
        else:
            k_final = k_solid if temp < self.T_m else k_molten

        return k_final

    def calcSpecificHeat(self, temp):
        c_solid = 0.8  # J/kg·K for solid regolith
        c_molten = 1.2  # J/kg·K for molten regolith

        if isinstance(temp, (list, np.ndarray)):
            c_final = np.where(temp < self.T_m, c_solid, c_molten)
        else:
            c_final = c_solid if temp < self.T_m else c_molten

        return c_final

    def heat_source_function(self, x, t, cycle_duration, heat_source_max):
        tube_radius = 0.1  # radius of the tube

        solar_constant = 1361  # W/m^2
        solar_flux = solar_constant * np.cos(np.deg2rad(self.solar_incidence_angle))  # Adjusted for incidence angle

        # Simulate cyclic heat source variation over the lunar cycle
        cyclic_variation = (1 + np.sin(2 * np.pi * t / cycle_duration)) / 2

        # Calculate the heat flux from the tube with a sinusoidal time dependency
        distance_from_tube = np.abs(x)
        heat_flux_tube = heat_source_max * cyclic_variation * np.exp(-distance_from_tube / tube_radius)

        # Total heat flux is the sum of the tube heat flux and the solar flux
        total_heat_flux = heat_flux_tube + solar_flux

        return total_heat_flux


class Iron(PCM):
    T_m, LH, rho, T_a = 1810, 247000, 7870, 293
    c_solid = 0.449  # specific heat of solid iron in J/g°C
    c_liquid = 0.82  # specific heat of liquid iron in J/g°C
    cycle_duration = 86400  # seconds for a day
    heat_source_max = 1000  # example value in W/m^3

    def __init__(self):
        self.dx = 0.05  # Spatial discretization
        self.k = self.calcThermalConductivity(Iron.T_a)  # Thermal conductivity
        self.c = self.calcSpecificHeat(Iron.T_a)  # Specific heat
        self.alpha2 = self.alpha(self.k, self.c, Iron.rho)  # Thermal diffusivity

        # Stability condition for dt calculation
        self.dt = (0.5 * self.dx ** 2) / self.alpha2

        print("dt = ", self.dt)
        print(f'alpha = {self.alpha2}')

        self.lmbda = self.dt / (self.dx ** 2)  # Used for numerical methods

    def generate_data(self, x_max, t_max):
        self.x_max = x_max  # Save x_max as an instance attribute
        x_grid = np.arange(0, x_max, self.dx)
        t_grid = np.arange(self.dt, t_max, self.dt)  # Exclude the final time step
        # print(f"Debug from generate_data: x_grid shape = {x_grid.shape}")
        # print(f"Debug from generate_data: t_grid shape = {t_grid.shape}")

        X, T = np.meshgrid(x_grid, t_grid, indexing='ij')
        x_features = np.column_stack([X.ravel(), T.ravel()])

        y_T = np.full(x_features.shape[0], self.T_a, dtype=np.float64)
        boundary_condition_indices = x_features[:, 0] == 0
        y_T[boundary_condition_indices] = self.T_m + 10.0

        y_B = y_T.copy()
        x_boundary = x_features.copy()

        T_arr = np.full((len(x_grid), len(t_grid)), self.T_a)
        H_arr = self.initial_enthalpy(x_grid, self, len(t_grid))

        # print(f"Debug: x_features shape = {x_features.shape}")
        # print(f"Debug: y_T shape = {y_T.shape}, y_B shape = {y_B.shape}")

        return x_features, y_T, y_B, x_boundary, x_grid, t_grid, T_arr, H_arr

    def explicitNumerical(self, x_arr, t_arr, T_arr, cls, phase_mask_array=None, boundary_mask_array=None):
        nx = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None or boundary_mask_array is None:
            phase_mask_array = np.zeros((nx, num_timesteps), dtype=int)
            boundary_mask_array = np.zeros((nx, num_timesteps), dtype=int)

        alpha = cls.alpha(cls.k, cls.c, cls.rho)
        moving_boundary_indices = np.full(num_timesteps, -1, dtype=int)  # Initialize moving_boundary_indices

        T_minus = cls.T_m - 100  # Assuming the default tolerance of 100
        T_plus = cls.T_m + 100

        for timestep in range(1, num_timesteps):
            T_old = T_arr[:, timestep - 1]
            diffusive_term = (np.roll(T_old, -1) - 2 * T_old + np.roll(T_old, 1))
            T_new = T_old + (alpha * cls.dt / cls.dx ** 2) * diffusive_term

            T_new[0] = cls.T_m + 10.0  # Fix the left boundary to T_m + 10.0
            T_new[-1] = T_old[-1]  # Maintain the last cell's temperature

            T_arr[:, timestep] = T_new

            # Directly compute the phase mask and boundary mask in the loop
            phase_mask_array[:, timestep] = np.where(T_new < T_minus, 0, np.where(T_new <= T_plus, 1, 2))
            boundary_mask_array[:, timestep] = np.where(phase_mask_array[:, timestep] == 1, 1, 0)

        if np.all(moving_boundary_indices == -1):  # Check if moving_boundary_indices was never updated
            x_features = np.column_stack([np.tile(x_arr, len(t_arr)), np.repeat(t_arr, len(x_arr))])
            moving_boundary_indices = self.calculate_boundary_indices(
                x_features,
                x_arr[-1],
                cls.dt,
                T=T_arr,
                T_m=cls.T_m,
                mode='moving_boundary'
            )

        return T_arr, phase_mask_array, boundary_mask_array, moving_boundary_indices

    def implicitSol(self, x_arr, t_arr, T_arr, H_arr, cls, phase_mask_array=None, boundary_mask_array=None):
        num_segments = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None:
            phase_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)
        if boundary_mask_array is None:
            boundary_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)

        moving_boundary_indices = np.full(num_timesteps, -1, dtype=int)

        # Initial condition: Set the first time step
        T_arr[:, 0] = cls.T_a  # Ambient temperature for all spatial cells
        T_arr[0, :] = cls.T_m + 10.0  # Higher temperature at the first spatial cell

        for time_step in range(1, num_timesteps):
            T_old = T_arr[:, time_step - 1]

            # Heat source function specific to iron
            heat_source = cls.heat_source_function(x_arr, t_arr[time_step])

            # Explicit update for the nonlinear heat source term
            T_explicit = T_old + cls.dt * heat_source / (cls.rho * cls.c)

            # Calculate thermal properties
            k_vals = np.array([cls.calcThermalConductivity(T_old[i]) for i in range(num_segments)])
            c_vals = np.array([cls.calcSpecificHeat(T_old[i]) for i in range(num_segments)])
            rho = cls.rho
            alpha_vals = k_vals / (c_vals * rho)
            lmbda_vals = cls.dt / (cls.dx ** 2) * alpha_vals

            # Construct tridiagonal matrix A for the implicit diffusion term
            A = lil_matrix((num_segments, num_segments))
            A.setdiag(1 + 2 * lmbda_vals)
            A.setdiag(-lmbda_vals[1:], -1)
            A.setdiag(-lmbda_vals[:-1], 1)

            # Apply boundary conditions
            A[0, :] = 0
            A[0, 0] = 1

            # Do not fix the temperature at the rightmost cell, allowing it to evolve
            A[-1, -2] = -lmbda_vals[-1]  # Adjust the last cell in the matrix A
            A[-1, -1] = 1 + lmbda_vals[-1]

            # Convert A to CSC format for efficient solving
            A = A.tocsc()

            try:
                # Solve the system using LU decomposition
                lu = splu(A)
                T_new = lu.solve(T_explicit)

                # Calculate new enthalpy from updated temperature
                H_new = cls.calcEnthalpy2(T_new, cls)
                T_arr[:, time_step] = T_new
                H_arr[:, time_step] = H_new

                # Compute masks
                phase_mask, boundary_mask = self.update_phase_mask(T_new, cls)
                phase_mask_array[:, time_step] = phase_mask
                boundary_mask_array[:, time_step] = boundary_mask

                moving_boundary_indices[time_step] = self.calculate_boundary_indices(
                    x_arr, self.x_max, cls.dt, T=T_arr, T_m=cls.T_m, mode='moving_boundary', tolerance=100
                )[time_step]
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        return T_arr, H_arr, phase_mask_array, boundary_mask_array, moving_boundary_indices

    def calcThermalConductivity(self, temp):
        """Calculate the thermal conductivity of iron based on its phase (solid or liquid).

        Args:
            temp (float): Temperature in degrees Celsius.

        Returns:
            float: Thermal conductivity in W/m·K.
        """
        if temp < self.T_m:
            return 73  # Thermal conductivity for solid iron in W/m·K
        else:
            return 35  # Thermal conductivity for liquid iron in W/m·K

    def calcSpecificHeat(self, temp):
        return np.where(temp < Iron.T_m, Iron.c_solid, Iron.c_liquid)

    def heat_source_function(self, x, t):
        half_cycle = self.cycle_duration / 2
        if 0 <= t % self.cycle_duration < half_cycle:
            return self.heat_source_max
        else:
            return 0
