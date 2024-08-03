import math
from abc import ABC, abstractmethod
import numpy as np
from scipy.sparse import diags, csc_matrix
from scipy.sparse.linalg import splu
from tensorflow.keras.layers import Flatten  # Use the Flatten layer from tensorflow.keras
from scipy.linalg import lu_factor, lu_solve
from scipy.special import erf


def compute_mask_arrays(T, cls, tolerance=100, phase_mask=None, boundary_mask=None):
    if phase_mask is None:
        phase_mask = np.zeros_like(T, dtype=int)
    if boundary_mask is None:
        boundary_mask = np.zeros_like(T, dtype=int)

    phase_mask[:] = 0
    phase_mask[T < cls.T_m - tolerance] = 0
    phase_mask[(T >= cls.T_m - tolerance) & (T <= cls.T_m + tolerance)] = 1
    phase_mask[T > cls.T_m + tolerance] = 2

    boundary_mask[:] = 0
    boundary_mask[phase_mask == 1] = 1

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
            temp_mask_array[:, t_idx], bound_mask_array[:, t_idx] = compute_mask_arrays(T_new, cls)

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
        T = np.ones((len(x_arr), t_steps)) * cls.T_a
        T[0, :] = cls.T_m + 10  # Set the first cell temperature to T_m + 10

        # Initialize the enthalpy array
        H = self.initial_enthalpy(x_arr, cls, t_steps)
        return T, H

    def calculate_dt(self, cls, max_k, safety_factor=0.4):
        max_alpha = max_k / (cls.rho * max(cls.c_solid, cls.c_liquid))
        max_dt = (safety_factor * cls.dx ** 2) / (max_alpha * 2)
        if max_dt <= 0:
            raise ValueError("Non-positive max_dt calculated, check input parameters.")
        return min(max_dt, cls.dt)

    def update_enthalpy_temperature(self, H_current, cls, gamma, x_arr):
        dH = gamma * (np.roll(H_current, -1) - 2 * H_current + np.roll(H_current, 1))
        H_next = H_current + dH
        H_next[0] = H_current[0]  # Reapply left boundary condition
        H_next[-1] = H_current[-1]  # Reapply right boundary condition
        T_next = self.update_temperature(H_next, cls)
        return H_next, T_next

    def update_temperature(self, enthalpy, cls, epsilon=10):
        T_minus = cls.T_m - epsilon
        T_plus = cls.T_m + epsilon

        # Calculate temperature based on enthalpy
        temperature = np.piecewise(
            enthalpy,
            [
                enthalpy < 0,
                (enthalpy >= 0) & (enthalpy <= cls.LH),
                enthalpy > cls.LH
            ],
            [
                lambda H: cls.T_a + H / (cls.rho * cls.c_solid),
                # Temperature rising from ambient as enthalpy increases
                lambda H: cls.T_m,  # Temperature remains constant in the mushy zone
                lambda H: cls.T_m + (H - cls.LH) / (cls.rho * cls.c_liquid)  # Temperature rising in liquid phase
            ]
        )

        return temperature

    def update_phase_mask(self, temperature_array, cls):
        tolerance = 0.5  # Narrow tolerance around the melting point
        return np.select(
            [temperature_array < cls.T_m - tolerance, temperature_array > cls.T_m + tolerance],
            [0, 1],  # 0 for solid, 1 for liquid
            default=2  # 2 for mushy zone (near melting point)
        )

    def calculate_boundary_indices(self, x, x_max, dt, T=None, T_m=None, tolerance=100, mode='initial', atol=1e-8,
                                   rtol=1e-5):
        if mode == 'initial':
            boundary_indices = {'condition1': [], 'condition2': []}
            dt_indices = np.isclose(x[:, 1], dt, atol=atol, rtol=rtol)
            boundary_indices['condition1'] = np.where(dt_indices & np.isclose(x[:, 0], 0, atol=atol, rtol=rtol))[0]
            boundary_indices['condition2'] = np.where(dt_indices & ~np.isclose(x[:, 0], x_max, atol=atol, rtol=rtol))[0]
            boundary_indices['condition1'] = np.asarray(boundary_indices['condition1'])
            boundary_indices['condition2'] = np.asarray(boundary_indices['condition2'])
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
            print(f"Debug: Time step {t_idx} - phase_change_indices: {phase_change_indices}")
            print(f"Debug: Time step {t_idx} - boundary_indices: {boundary_indices[t_idx]}")
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

    def calcEnthalpy2(self, T, cls, epsilon=0.01):
        T_minus = cls.T_m - epsilon
        T_plus = cls.T_m + epsilon

        def smoothed_enthalpy(T):
            if T < T_minus:
                return cls.rho * cls.c * (T - cls.T_a)
            elif T > T_plus:
                return cls.LH + cls.rho * cls.c * (T - cls.T_m)
            else:
                return cls.rho * cls.c * (T_minus - cls.T_a) + ((cls.LH / (2 * epsilon)) * (T - T_minus))

        H = np.piecewise(T, [T < T_minus, (T >= T_minus) & (T <= T_plus), T > T_plus],
                         [lambda T: cls.rho * cls.c * (T - cls.T_a),
                          lambda T: smoothed_enthalpy(T),
                          lambda T: cls.LH + cls.rho * cls.c * (T - cls.T_m)])

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
        E_vals = H_vals  # Enthalpy is the energy

        P_settlement = 100  # Power requirement for the settlement in kW
        time_hours = 29.5 * 24  # Full lunar day-night cycle in hours

        # Energy needed for the settlement over the full lunar day-night cycle
        E_settlement = P_settlement * time_hours  # in kWh
        # E_settlement *= 3.6e6  # convert to J

        # Total energy stored in the regolith over the full lunar day-night cycle
        E_regolith_total = np.sum(E_vals)  # in J

        # Conclusion
        if E_regolith_total >= E_settlement:
            return "The thermal energy in the regolith is sufficient to power the settlement."
        else:
            return "The thermal energy in the regolith is not sufficient to power the settlement."

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
    rho = 1.5  # Adjusted density in kg/m³
    T_a = 253   # Ambient temperature in Kelvin
    cycle_duration = 2551443  # seconds for a lunar day

    def __init__(self):
        self.dx = 0.2
        self.k = self.calcThermalConductivity(Regolith.T_m)
        self.c = self.calcSpecificHeat(Regolith.T_m)
        self.alpha2 = self.alpha(self.k, self.c, Regolith.rho)
        self.dt = (0.4 * self.dx ** 2) * self.alpha2
        print("dt = ", self.dt)
        print(f'alpha = {self.alpha2}')
        self.lmbda = self.dt / (self.dx ** 2)

        # Define specific heats for solid and liquid phases
        self.c_solid = self.calcSpecificHeat(Regolith.T_a)
        self.c_liquid = self.calcSpecificHeat(Regolith.T_m)

    def generate_data(self, x_max, t_max):
        self.x_max = x_max  # Save x_max as an instance attribute
        x_grid = np.arange(0, x_max, self.dx)
        t_grid = np.arange(self.dt, t_max, self.dt)
        print(f"x_grid size: {len(x_grid)}, t_grid size: {len(t_grid)}")

        X, T = np.meshgrid(x_grid, t_grid, indexing='ij')
        x_features = np.column_stack([X.ravel(), T.ravel()])

        y_T = np.full(x_features.shape[0], Regolith.T_a, dtype=np.float64)
        y_B = y_T.copy()
        x_boundary = np.zeros_like(x_features, dtype=np.float64)

        # Set initial boundary condition at x=0
        boundary_condition_indices = x_features[:, 0] == 0
        y_T[boundary_condition_indices] = Regolith.T_m + 10.0

        T_arr = np.full((len(x_grid), len(t_grid)), Regolith.T_a)
        H_arr = self.initial_enthalpy(x_grid, self, len(t_grid))

        return x_features, y_T, y_B, x_boundary, x_grid, t_grid, T_arr, H_arr

    def explicitNumerical(self, x_arr, t_arr, T_arr, cls, phase_mask_array=None, boundary_mask_array=None):
        nx = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None or boundary_mask_array is None:
            phase_mask_array = np.zeros((nx, num_timesteps), dtype=int)
            boundary_mask_array = np.zeros((nx, num_timesteps), dtype=int)

        alpha = cls.alpha(cls.k, cls.c, cls.rho)
        moving_boundary_indices = None

        for timestep in range(1, num_timesteps):
            T_old = T_arr[:, timestep - 1]
            diffusive_term = (np.roll(T_old, -1) - 2 * T_old + np.roll(T_old, 1))
            T_new = T_old + (alpha * cls.dt / cls.dx ** 2) * diffusive_term

            # Heat source effect at x=0 based on lunar cycle
            heat_source = cls.heat_source_function(0, t_arr[timestep])
            T_new[0] = Regolith.T_m + heat_source / (cls.rho * cls.c) if heat_source > 0 else T_new[0]

            T_new[-1] = T_old[-1]

            T_arr[:, timestep] = T_new
            phase_mask_array[:, timestep], boundary_mask_array[:, timestep] = compute_mask_arrays(T_new, cls)

        if moving_boundary_indices is None:
            x_features = np.column_stack([np.tile(x_arr, len(t_arr)), np.repeat(t_arr, len(x_arr))])
            moving_boundary_indices = self.calculate_boundary_indices(x_features, x_arr[-1], cls.dt, T=T_arr,
                                                                      T_m=cls.T_m, mode='moving_boundary')

        return T_arr, phase_mask_array, boundary_mask_array, moving_boundary_indices

    def implicitSol(self, x_arr, t_arr, T_arr, H_arr, cls, phase_mask_array=None, boundary_mask_array=None):
        num_segments = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None:
            phase_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)
        if boundary_mask_array is None:
            boundary_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)

        moving_boundary_indices = np.full(num_timesteps, -1, dtype=int)

        # Set initial temperature and enthalpy
        T_initial = np.full(num_segments, cls.T_a)
        T_initial[0] = cls.T_m  # Boundary condition at the first cell
        T_arr[:, 0] = T_initial
        H_arr[:, 0] = cls.calcEnthalpy2(T_initial, cls)

        for time_step in range(1, num_timesteps):
            T_old = T_arr[:, time_step - 1]
            H_old = H_arr[:, time_step - 1]

            # Calculate thermal properties for each segment
            k_vals = np.array([cls.calcThermalConductivity(T_old[i]) for i in range(num_segments)])
            c_vals = np.array([cls.calcSpecificHeat(T_old[i]) for i in range(num_segments)])
            rho = cls.rho
            alpha_vals = k_vals / (c_vals * rho)
            lmbda_vals = cls.dt / cls.dx ** 2 * alpha_vals

            # Construct the coefficient matrix A
            diagonals = [
                -lmbda_vals[1:],  # Sub-diagonal
                1 + 2 * lmbda_vals,  # Main diagonal
                -lmbda_vals[:-1]  # Super-diagonal
            ]
            A = diags(diagonals, [-1, 0, 1], shape=(num_segments, num_segments)).toarray()
            A = csc_matrix(A)  # Convert to CSC format for efficient solving
            b = H_old.copy()

            # Apply boundary conditions
            A[0, :] = 0
            A[0, 0] = 1
            b[0] = cls.calcEnthalpy2(cls.T_m, cls)  # Set enthalpy corresponding to T_m at the first cell
            A[-1, :] = 0
            A[-1, -1] = 1
            b[-1] = cls.calcEnthalpy2(cls.T_a, cls)  # Set enthalpy corresponding to T_a at the last cell

            try:
                # Solve the linear system A * H_new = b
                lu = splu(A)
                H_new = lu.solve(b)

                # Update temperature based on the new enthalpy
                T_new = self.update_temperature(H_new, cls)

                # Store the new temperature and enthalpy values
                T_arr[:, time_step] = T_new
                H_arr[:, time_step] = H_new

                # Update phase and boundary masks
                phase_mask_array[:, time_step], boundary_mask_array[:, time_step] = compute_mask_arrays(T_new, cls)

                # Calculate moving boundary indices
                moving_boundary_indices[time_step] = self.calculate_boundary_indices(
                    x_arr, self.x_max, cls.dt, T=T_arr, T_m=cls.T_m, mode='moving_boundary', tolerance=100
                )[time_step]

                # Debug prints for monitoring
                print(f"Debug: Time step {time_step} - T_new: {T_new}")
                print(f"Debug: Time step {time_step} - H_new: {H_new}")
                print(f"Debug: Time step {time_step} - phase_mask: {phase_mask_array[:, time_step]}")
                print(f"Debug: Time step {time_step} - boundary_mask: {boundary_mask_array[:, time_step]}")

            except Exception as e:
                print(f"An error occurred: {e}")
                break

        print(f"Debug: Final T_arr (Implicit): {T_arr}")
        print(f"Debug: Final H_arr (Implicit): {H_arr}")
        print(f"Debug: Final phase_mask_array: {phase_mask_array}")
        print(f"Debug: Final boundary_mask_array: {boundary_mask_array}")
        print(f"Debug: Final moving_boundary_indices: {moving_boundary_indices}")

        return T_arr, H_arr, phase_mask_array, boundary_mask_array, moving_boundary_indices

    def calcThermalConductivity(self, temp):
        # Thermal conductivity for granular regolith (W/m·K)
        C1 = 1.281e-2
        C2 = 4.431e-2

        if isinstance(temp, (list, np.ndarray)):
            # If temp is a list or array, compute k_granular as an array
            k_granular = C1 + C2 * np.array(temp, dtype=np.float64) ** (-3)
        else:
            # If temp is a scalar, compute k_granular as a scalar
            k_granular = C1 + C2 * float(temp) ** (-3)

        # For molten regolith, thermal conductivity transitions to a fixed value
        k_molten_end = 2.5  # Increased value for molten phase conductivity

        if isinstance(temp, (list, np.ndarray)):
            # For arrays, ensure proper interpolation
            k_final = np.where(
                temp < Regolith.T_m,
                k_granular,
                np.linspace(k_granular[-1], k_molten_end, len(temp))
            )
        else:
            # For scalars, handle the molten phase directly
            k_final = k_granular if temp < Regolith.T_m else k_molten_end

        return 1000.0 * k_final

    def calcSpecificHeat(self, temp):
        # Specific heat for lunar regolith (J/kg·K)
        specific_heat = -1848.5 + 1047.41 * np.log(temp)
        # Ensure the specific heat is realistic and not negative or excessively high
        # specific_heat = np.clip(specific_heat, 0, 100)
        return specific_heat

    def heat_source_function(self, x, t):
        # Parameters for the heat source
        tube_radius = 0.1  # Example radius of the tube in meters
        max_heat_flux = 10000  # Example maximum heat flux in W/m^2

        # Calculate the distance from the tube axis (assumed at x=0)
        distance_from_tube = np.abs(x)

        # Spatial decay: Stronger heat near the tube, diminishing further away
        # Using an exponential decay model as an example
        heat_flux = max_heat_flux * np.exp(-distance_from_tube / tube_radius)

        return heat_flux


class Iron(PCM):
    T_m, LH, rho, T_a = 1810, 247000, 7870, 293  # Units should be consistent
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
        moving_boundary_indices = None

        for timestep in range(1, num_timesteps):
            T_old = T_arr[:, timestep - 1]
            diffusive_term = (np.roll(T_old, -1) - 2 * T_old + np.roll(T_old, 1))
            T_new = T_old + (alpha * cls.dt / cls.dx ** 2) * diffusive_term

            T_new[0] = cls.T_m + 10.0
            T_new[-1] = T_old[-1]

            T_arr[:, timestep] = T_new
            phase_mask_array[:, timestep], boundary_mask_array[:, timestep] = compute_mask_arrays(T_new, cls)

        if moving_boundary_indices is None:
            x_features = np.column_stack([np.tile(x_arr, len(t_arr)), np.repeat(t_arr, len(x_arr))])
            moving_boundary_indices = self.calculate_boundary_indices(x_features, x_arr[-1], cls.dt, T=T_arr,
                                                                      T_m=cls.T_m, mode='moving_boundary')

        return T_arr, phase_mask_array, boundary_mask_array, moving_boundary_indices

    def implicitSol(self, x_arr, t_arr, T_arr, H_arr, cls, phase_mask_array=None, boundary_mask_array=None):
        num_segments = len(x_arr)
        num_timesteps = len(t_arr)

        if phase_mask_array is None:
            phase_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)
        if boundary_mask_array is None:
            boundary_mask_array = np.zeros((num_segments, num_timesteps), dtype=int)

        moving_boundary_indices = np.full(num_timesteps, -1, dtype=int)
        for time_step in range(1, num_timesteps):
            T_old = T_arr[:, time_step - 1]
            H_old = H_arr[:, time_step - 1]

            k_vals = np.array([cls.calcThermalConductivity(T_old[i]) for i in range(num_segments)])
            c_vals = np.array([cls.calcSpecificHeat(T_old[i]) for i in range(num_segments)])
            rho = cls.rho
            alpha_vals = k_vals / (c_vals * rho)
            lmbda_vals = cls.dt / cls.dx ** 2 * alpha_vals

            diagonals = [
                -lmbda_vals[1:],
                1 + 2 * lmbda_vals,
                -lmbda_vals[:-1]
            ]
            A = diags(diagonals, [-1, 0, 1], shape=(num_segments, num_segments)).toarray()
            b = T_old.copy()

            # Apply boundary conditions
            A[0, :] = 0
            A[0, 0] = 1
            b[0] = cls.T_m + 10  # Temperature slightly above T_m at the first cell
            A[-1, :] = 0
            A[-1, -1] = 1
            b[-1] = cls.T_a  # Ambient temperature T_a everywhere else

            try:
                lu = splu(A)
                T_new = lu.solve(b)

                print(f"Debug: Time step {time_step} - T_new: {T_new}")

                H_new = cls.calcEnthalpy2(T_new, cls)
                T_arr[:, time_step] = T_new
                H_arr[:, time_step] = H_new

                # Use compute_mask_arrays to compute masks
                phase_mask, boundary_mask = compute_mask_arrays(T_new, cls)
                phase_mask_array[:, time_step] = phase_mask
                boundary_mask_array[:, time_step] = boundary_mask

                print(f"Debug: Time step {time_step} - mask_new: {phase_mask}")
                print(f"Debug: Time step {time_step} - boundary_mask_new: {boundary_mask}")

                moving_boundary_indices[time_step] = self.calculate_boundary_indices(
                    x_arr, self.x_max, cls.dt, T=T_arr, T_m=cls.T_m, mode='moving_boundary', tolerance=100
                )[time_step]
            except Exception as e:
                print(f"An error occurred: {e}")
                break

        print(f"Debug: Final phase_mask_array: {phase_mask_array}")
        print(f"Debug: Final boundary_mask_array: {boundary_mask_array}")
        print(f"Debug: Final moving_boundary_indices: {moving_boundary_indices}")

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
