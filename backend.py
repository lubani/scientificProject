import math
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import erf


def compute_mask_arrays(x_temp, x_boundary, L, t_steps):
    nx = len(x_temp) // t_steps  # Assuming x_temp is now 1D
    print("compute mask arrays: x_temp shape:", x_temp.shape)
    print("compute mask arrays: x_boundary shape:", x_boundary.shape)

    # Initialize mask arrays with the expected shape
    mask_temp = np.zeros(nx * t_steps)
    mask_boundary = np.zeros(nx * t_steps)

    # Populate mask arrays
    for t in range(t_steps):
        # Identify temperature boundaries (at x=0 and x=L)
        mask_temp[t::t_steps] = np.isclose(x_temp[t::t_steps], 0.0) | np.isclose(x_temp[t::t_steps], L)

        # Assuming x_boundary is spatially ordered, mark the first and last points as boundary points
        mask_boundary[t * nx] = 1  # First point in each time step
        mask_boundary[(t + 1) * nx - 1] = 1  # Last point in each time step

    print("compute mask arrays: boundary mask array values: ", mask_boundary)
    print("compute mask arrays: temp mask array values: ", mask_temp)
    return mask_temp, mask_boundary




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

    def alpha(self, k, c, rho):
        # Check for division by zero or small number
        if c * rho == 0 or k == 0:
            raise ValueError("c * rho should not be zero")
        return k / (c * rho)

    import numpy as np

    def solve_stefan_problem_enthalpy(self, cls, L, t_max, phase_mask_array, boundary_mask_array):
        gamma = cls.alpha2 * cls.dt / cls.dx ** 2
        print("gamma: ", gamma)
        t_arr = np.arange(cls.dt, t_max, cls.dt)
        x_arr = np.arange(0, L, cls.dx)
        Nx, Nt = len(x_arr), len(t_arr)

        T = np.ones((Nx, Nt)) * cls.T_a  # Initial temperature array
        # Inside solve_stefan_problem_enthalpy
        H = self.initial_enthalpy(x_arr, cls, Nt)  # Nt is the number of time steps
        print("H_arr_enthalpy initial: ", H)
        T[0, :] = cls.T_m + 100  # Setting boundary condition at x=0

        for n in range(0, Nt - 1):
            for i in range(1, Nx - 1):
                # print(f"stefan enthalpy: boundary at index of t={n}, index of x={i}: {boundary_mask_array[i, n]}")
                if boundary_mask_array[i, n] == 0:
                    # Calculate the change in enthalpy based on the thermal diffusivity and neighboring points
                    dH = gamma * (H[i + 1, n] - 2 * H[i, n] + H[i - 1, n])
                    print(f"Change in enthalpy at this step: {dH}")
                    H[i, n + 1] = H[i, n] + dH
                    print(f"Updated enthalpy at this step: {H[i, n + 1]}")
                    # Update temperature based on the new enthalpy
                    T[i, n + 1] = self.update_temperature(H[i, n + 1], cls)

            # After updating all temperatures at this time step, update the phase mask.
            phase_mask_array[:, n + 1] = self.update_phase_mask(T[:, n + 1], cls)
        return T, H, phase_mask_array, boundary_mask_array

    def update_temperature(self, enthalpy, cls):
        if enthalpy <= 0:
            return cls.T_a + (enthalpy / (cls.rho * cls.c))
        elif 0 < enthalpy < cls.LH:
            return cls.T_m
        else:
            return cls.T_m + ((enthalpy - cls.LH) / (cls.rho * cls.c))

    def update_phase_mask(self, temperature_array, cls):
        tolerance = 50
        return np.select(
            [temperature_array < cls.T_m - tolerance,
             temperature_array > cls.T_m + tolerance],
            [0, 1],
            default=2  # Default to phase transition if within tolerance
        )

    # def calculate_boundary_indices(self, x_features, x_max, dt):
    #     boundary_indices = {
    #         'condition1': [],
    #         'condition2': []
    #     }
    #
    #     for i, (x, t) in enumerate(x_features):
    #         # print(f"Debug: i = {i}, x = {x}, t = {t}")  # Debugging line
    #         if np.isclose(t, dt):
    #             if np.isclose(x, 0):
    #                 boundary_indices['condition1'].append(i)
    #             elif not np.isclose(x, x_max):
    #                 boundary_indices['condition2'].append(i)
    #
    #     # print(f"Debug: boundary_indices = {boundary_indices}")  # Debugging line
    #     return boundary_indices

    def calculate_boundary_indices(self, x, x_max, dt, T=None, T_m=None, tolerance=50, mode='initial', atol=1e-8,
                                   rtol=1e-5):
        if mode == 'initial':
            boundary_indices = {
                'condition1': [],
                'condition2': []
            }
            dt_indices = np.isclose(x[:, 1], dt, atol=atol, rtol=rtol)
            boundary_indices['condition1'] = np.where(dt_indices & np.isclose(x[:, 0], 0, atol=atol, rtol=rtol))[0]
            boundary_indices['condition2'] = np.where(dt_indices & ~np.isclose(x[:, 0], x_max, atol=atol, rtol=rtol))[0]
            return boundary_indices

        elif mode == 'moving_boundary':
            if T is None or T_m is None:
                raise ValueError(
                    "Temperature array T and melting point T_m must be provided for 'moving_boundary' mode.")
            moving_boundary_indices = np.full(T.shape[1], -1,
                                              dtype=int)  # Initialize with -1 to indicate no boundary found
            for n in range(T.shape[1]):
                phase_change_indices = np.where(np.abs(T[:, n] - T_m) <= tolerance)[0]
                if phase_change_indices.size > 0:
                    moving_boundary_indices[n] = phase_change_indices[0]
            return moving_boundary_indices
        else:
            raise ValueError("Invalid mode. Choose between 'initial' and 'moving_boundary'")

    def calculate_moving_boundary_indices(self, T, T_m, tolerance=50):
        moving_boundary_indices = np.full(T.shape[1], -1, dtype=int)  # Initialize with -1 to indicate no boundary found
        abs_diff = np.abs(T - T_m)
        for n in range(T.shape[1]):
            phase_change_indices = np.where(abs_diff[:, n] <= tolerance)[0]
            if phase_change_indices.size > 0:
                moving_boundary_indices[n] = phase_change_indices[0]
        return moving_boundary_indices

    def calcTemperature(self, x_arr, t_arr, T_arr, cls):
        for j in range(len(t_arr) - 1):
            for i in range(1, len(x_arr) - 1):
                if x_arr[i] != 0 and t_arr[j] != 0:
                    cls.k = cls.calcThermalConductivity(T_arr[i, j])
                    cls.c = cls.calcSpecificHeat(T_arr[i, j])
                    # cls.alpha2 = self.alpha(cls.k, cls.c, cls.rho)
                    # cls.lmbda *= cls.alpha2
                    T_arr[i, j + 1] = T_arr[i, j] * (1 - 2 * cls.lmbda) + cls.lmbda * \
                                      (T_arr[i + 1, j] + T_arr[i - 1, j])
        return T_arr

    def heat_source_function(self, t, cycle_duration, heat_source_max):
        half_cycle = cycle_duration / 2
        if 0 <= t % cycle_duration < half_cycle:
            return heat_source_max
        else:
            return 0

    @abstractmethod
    def implicitSol(self, x_arr, tmax, cls, phase_mask_array=None, boundary_mask_array=None):
        pass

    @abstractmethod
    def calcTemperature3(self, x_arr, tmax, cls, phase_mask_array=None, boundary_mask_array=None):
        pass

    def inverseEnthalpy2(self, H, cls):
        T = np.full_like(H, cls.T_m)
        T[H < 0] = cls.T_a + H[H < 0] / (cls.rho * cls.c)
        T[H > cls.LH] = cls.T_a + (H[H > cls.LH] - cls.LH) / (cls.rho * cls.c)
        return T

    def initial_enthalpy(self, x_arr, cls, t_steps):
        # Initialize the enthalpy array with zeros for both space and time dimensions
        H_arr = np.zeros((len(x_arr), t_steps), dtype=np.float32)

        # Loop over the spatial domain to set initial enthalpy based on the initial temperature
        for i, x in enumerate(x_arr):
            T_initial = cls.T_a if i != 0 else cls.T_m + 100  # Adjust as necessary

            # Calculate the enthalpy for the first time step based on the initial temperature
            if T_initial < cls.T_m:
                initial_H = cls.c * (T_initial - cls.T_a)
            elif T_initial == cls.T_m:
                initial_H = cls.LH
            else:
                initial_H = cls.LH + cls.c * (T_initial - cls.T_m)

            # Set the initial enthalpy for all time steps to the value calculated from the initial temperature
            H_arr[i, :] = initial_H

        return H_arr

    def initial_temperature(self, x_arr, cls):
        return np.full_like(x_arr, cls.T_m)

    # def calcEnthalpy2(self, temp, cls):
    #     if temp >= cls.T_m:
    #         E = (temp - cls.T_m) * cls.rho * cls.calcSpecificHeat(temp)
    #     else:
    #         E = cls.calcSpecificHeat(temp) * (cls.T_m - temp)
    #     return E

    def calcEnthalpy2(self, T, cls):
        conditions = [
            T < cls.T_m,
            T > cls.T_m
        ]

        choices = [
            cls.rho * cls.c * (T - cls.T_a),
            cls.LH + cls.rho * cls.c * (T - cls.T_m)
        ]

        H = np.select(conditions, choices, default=cls.LH)

        return H

    def calcEnthalpy(self, x_arr, t_max, cls):
        num_points = len(x_arr)
        T = np.full(num_points, cls.T_a)  # temperature array
        c = np.array([self.calcSpecificHeat(temp) for temp in T])
        H = c * T  # enthalpy array
        H = np.array(H, dtype=np.float32)
        c = np.array(c, dtype=np.float32)

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

    def explicitSol(self, x_val, t_arr, cls, phase_mask_array=None, bound_mask_array=None):
        T_initial = cls.T_a
        T_final = cls.T_m
        T = np.full((len(x_val), len(t_arr)), T_initial, dtype=np.float64)  # Using float64 for higher precision
        # T[0, 0] = T_final

        # Compute the mask arrays
        if bound_mask_array is None:
            phase_mask_array, bound_mask_array = compute_mask_arrays(x_val, x_val, max(x_val), len(t_arr))

        for t_idx, t_val in enumerate(t_arr):
            alpha2 = self.alpha(cls.k, cls.c, cls.rho)
            x_term = np.outer(x_val, 1 / (2 * alpha2 * np.sqrt(t_val)))
            x_term = np.squeeze(x_term)  # Reshaping to ensure correct broadcasting

            T[:, t_idx] = T_initial + (T_final - T_initial) * (1 - erf(x_term))

            # Apply boundary condition
            T[:, t_idx] *= (1 - bound_mask_array[:, t_idx])

            # Update phase mask array based on temperature
            phase_mask_array[:, t_idx] = np.where(T[:, t_idx] < cls.T_m, 0, np.where(T[:, t_idx] > cls.T_m, 1, 2))

        return T, phase_mask_array, bound_mask_array

    def calc_k(self, e_val, cls):
        if e_val <= cls.LH:
            return 1
        elif cls.LH < e_val < 2 * cls.LH:
            return 0
        else:
            return 1


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
    T_m, LH, rho, T_a = 1373, 470, 1.7, 273

    def __init__(self):
        self.dx = 0.25
        self.k = self.calcThermalConductivity(Regolith.T_m)
        self.c = self.calcSpecificHeat(Regolith.T_m)
        self.alpha2 = self.alpha(self.k, self.c, Regolith.rho)
        self.dt = (0.4 * self.dx ** 2) * self.alpha2
        print("dt = ", self.dt)
        print(f'alpha = {self.alpha2}')
        self.lmbda = self.dt / (self.dx ** 2)

    def generate_data(self, x_max, t_max):
        x_grid = np.arange(0, x_max, self.dx)
        t_grid = np.arange(self.dt, t_max, self.dt)

        X, T = np.meshgrid(x_grid, t_grid, indexing='ij')
        x_features = np.hstack((X.ravel().reshape(-1, 1), T.ravel().reshape(-1, 1)))

        # Initialize x_boundary with zeros and then set values based on condition
        x_boundary = np.zeros_like(x_features, dtype=np.float32)
        for i in range(len(x_features)):
            x_val, t_val = x_features[i]
            x_boundary[i] = 0 if x_val < self.T_m else 1

        y_T = np.full(X.size, Regolith.T_a, dtype=np.float32)
        y_B = np.full(X.size, Regolith.T_a, dtype=np.float32)  # Match the size of y_T

        # Reshape if needed to match the model's expected input shape
        expected_shape = (len(x_grid) * len(t_grid), 2)
        x_features = x_features.reshape(expected_shape)
        x_boundary = x_boundary.reshape(expected_shape)
        y_T = y_T.reshape(len(x_grid) * len(t_grid))
        y_B = y_B.reshape(len(x_grid) * len(t_grid))

        return x_features, y_T, y_B, x_boundary, x_grid, t_grid

    def calcTemperature3(self, x_arr, tmax, cls, phase_mask_array=None, boundary_mask_array=None):
        nx = len(x_arr)
        cls.dt = max(min(cls.dt, 0.5 * cls.dx ** 2 * cls.alpha2), 0.001)
        T_arr = np.full((nx, 1), cls.T_a, dtype=np.float32)
        T_arr[0, :] = Regolith.T_m + 100
        H_arr = self.initial_enthalpy(x_arr, cls).reshape(-1, 1)
        current_time = cls.dt
        t_arr = np.array([current_time], dtype=np.float32)

        heat_source_max = 1  # W/m^2, solar constant
        day_duration = 14 * 24 * 60  # in minutes

        if boundary_mask_array is None:
            boundary_mask_array = np.zeros((nx, 1))
        if phase_mask_array is None:
            phase_mask_array = np.zeros((nx, 1))

        while current_time < tmax:
            H_old = H_arr[:, -1]
            H_new = H_old.copy()
            mask_new = phase_mask_array[:, -1].copy()

            # Heat source effect at x=0
            heat_source = heat_source_max if current_time % (2 * day_duration) < day_duration else 0
            T_left = Regolith.T_m + heat_source / (cls.rho * cls.c)
            H_new[0] = T_left * cls.rho * cls.c if T_left < cls.T_m else T_left * cls.rho * cls.c + cls.LH

            # Neumann boundary condition at x=nx-1 (zero flux)
            H_new[-1] = H_old[-1]

            for i in range(1, nx - 1):
                if phase_mask_array[i, -1] == 0:  # Modify only if it's an internal point
                    H_new[i] = H_old[i] + cls.dt * cls.k * (H_old[i - 1] - 2 * H_old[i] + H_old[i + 1]) / cls.dx ** 2

            # Compute temperature based on enthalpy
            T_new = np.full_like(H_new, cls.T_a, dtype=np.float32)
            mask_new = np.zeros_like(phase_mask_array[:, -1])  # Initialize a new mask array

            for i, h in enumerate(H_new):
                if h < 0:
                    T_new[i] = max(cls.T_a, cls.T_m + h / (cls.rho * cls.c))
                elif 0 <= h <= cls.LH:
                    T_new[i] = cls.T_m
                elif h > cls.LH:
                    T_new[i] = cls.T_m + (h - cls.LH) / (cls.rho * cls.c)

                # Update mask based on the new temperature
                mask_new[i] = 0 if T_new[i] < cls.T_m else (1 if T_new[i] > cls.T_m else 2)

            # Dynamically add new cells to T_arr, H_arr, phase_mask_array, and t_arr
            T_arr = np.hstack((T_arr, T_new[:, np.newaxis]))
            H_arr = np.hstack((H_arr, H_new[:, np.newaxis]))
            phase_mask_array = np.hstack((phase_mask_array, mask_new[:, np.newaxis]))
            t_arr = np.hstack((t_arr, current_time))

            current_time += cls.dt

        return T_arr, phase_mask_array, boundary_mask_array

    def implicitSol(self, x_arr, tmax, cls, phase_mask_array=None, boundary_mask_array=None):
        # Initialize 2D boundary_mask_array and phase_mask_array
        if boundary_mask_array is None:
            phase_mask_array, boundary_mask_array = compute_mask_arrays(x_arr, x_arr, max(x_arr), int(tmax / self.dt))

        heat_source_max = 256  # Maximum heat source
        T_ambient = cls.T_a  # Ambient temperature

        num_segments = len(x_arr)
        t_arr = [self.dt]  # Time array
        T_arr = np.full((num_segments, len(t_arr)), cls.T_a)
        H_arr = np.ones((num_segments, 1)) * cls.calcEnthalpy2(T_arr, cls)  # Initial enthalpy

        t = self.dt
        while t < tmax:
            T_old = T_arr[:, -1]
            H_old = H_arr[:, -1]
            T_new = np.copy(T_old)
            H_new = np.copy(H_old)

            # Create the coefficient matrix A and the right-hand side vector b
            A = np.zeros((num_segments, num_segments))
            b = np.zeros(num_segments)

            for i in range(1, num_segments - 1):
                if boundary_mask_array[i, -1] == 0:  # Modify only if it's not a boundary point
                    # Modify only if it's an internal point or mask_array is None
                    k = cls.calcThermalConductivity(T_old[i])
                    c = cls.calcSpecificHeat(T_old[i])
                    alpha = self.alpha(k, c, cls.rho)

                    # Check if alpha is too small
                    if abs(alpha) < 1e-10:
                        raise ValueError("alpha is too small")

                    lmbda = cls.dt / cls.dx ** 2

                    # Check if lmbda is too large
                    if abs(lmbda) > 1e10:
                        raise ValueError("lmbda is too large")

                    day = t / (24 * 60 * 60)  # Convert time t from seconds to days
                    if day % 29.5 < 14:  # Heating for 14 days
                        heat_source = heat_source_max
                    else:  # Cooling for 14 days
                        heat_source = 0

                    # Fill the coefficient matrix A and the right-hand side vector b
                    A[i, i - 1] = -lmbda
                    A[i, i] = 1 + 2 * lmbda
                    A[i, i + 1] = -lmbda
                    b[i] = T_old[i]  # Right-hand side is just the old temperature

                    phase_mask_array[:, -1] = np.where(T_new < cls.T_m, 0, np.where(T_new > cls.T_m, 1, 2))
            # Apply Neumann boundary conditions
            if boundary_mask_array[0, -1] == 1:  # Check if it's a boundary point
                A[0, 0] = 1
                A[0, 1] = -1
                b[0] = heat_source * cls.dt / cls.dx  # Heat source applied at the left boundary

            if boundary_mask_array[-1, -1] == 1:  # Check if it's a boundary point
                A[-1, -1] = 1
                A[-1, -2] = -1
                b[-1] = 0  # Right boundary is insulated

            # Perform LU decomposition
            lu, piv = lu_factor(A)

            # Solve the system of equations
            T_new = lu_solve((lu, piv), b)

            # Calculate the new enthalpy
            H_new = cls.calcEnthalpy2(T_new, cls)

            # Append the new temperature and enthalpy to the arrays
            T_arr = np.hstack((T_arr, T_new[:, np.newaxis]))
            H_arr = np.hstack((H_arr, H_new[:, np.newaxis]))

            # Update the time
            t += cls.dt
            t_arr.append(t)

        return T_arr, H_arr, np.array(t_arr), phase_mask_array, boundary_mask_array

    def calcThermalConductivity(self, temp):
        # Thermal conductivity for granular regolith
        C1 = 1.281e-2
        C2 = 4.431e-2
        k_granular = C1 + C2 * np.float32(temp) ** (-3)

        # Thermal conductivity for molten regolith
        k_molten = 1.53  # This is a placeholder, replace with actual value if available
        k_final = np.where(temp < Regolith.T_m, k_granular, k_molten)
        # print("k = ", k_final)
        # Return the appropriate thermal conductivity based on the phase of the regolith
        return k_final

    def calcSpecificHeat(self, temp):
        # Specific heat for granular regolith
        self.c_solid = 1.512

        # Specific heat for molten regolith
        self.c_liquid = 1.512

        # Return the appropriate specific heat based on the phase of the regolith
        return np.where(temp < Regolith.T_m, self.c_solid, self.c_liquid)


class Iron(PCM):
    T_m, LH, rho, T_a = 1810, 247, 7.657, 300
    c_solid = 0.449  # specific heat of solid iron in J/g°C
    c_liquid = 0.82  # specific heat of liquid iron in J/g°C

    def __init__(self):
        self.dx = 0.1
        self.k = self.calcThermalConductivity(Iron.T_a)
        self.c = self.calcSpecificHeat(Iron.T_a)
        self.alpha2 = self.alpha(self.k, self.c, Iron.rho)
        self.dt = (0.4 * self.dx ** 2) * self.alpha2
        print("dt = ", self.dt)
        print(f'alpha = {self.alpha2}')
        self.lmbda = self.dt / (self.dx ** 2)

    # def generate_data(self, x_max, t_max):
    #     x_grid = np.arange(0, x_max, self.dx)
    #     t_grid = np.arange(self.dt, t_max, self.dt)
    #
    #     X, T = np.meshgrid(x_grid, t_grid)
    #     x_features = np.hstack(
    #         (X.reshape(-1, 1), T.reshape(-1, 1)))
    #
    #     x_boundary = np.vstack((np.hstack((np.zeros_like(t_grid).reshape(-1, 1), t_grid.reshape(-1, 1))),
    #                             np.hstack((np.full_like(t_grid, x_max).reshape(-1, 1), t_grid.reshape(-1, 1)))))
    #
    #     y_T = np.full(x_features.shape[0], Iron.T_a, dtype=np.float32)
    #     y_B = np.full(x_boundary.shape[0], Iron.T_a, dtype=np.float32)
    #
    #     return x_features, y_T, y_B, x_boundary, x_grid, t_grid

    def generate_data(self, x_max, t_max):
        x_grid = np.arange(0, x_max, self.dx)
        t_grid = np.arange(self.dt, t_max, self.dt)

        X, T = np.meshgrid(x_grid, t_grid, indexing='ij')
        x_features = np.hstack((X.ravel().reshape(-1, 1), T.ravel().reshape(-1, 1)))

        # Initialize x_boundary with zeros and then set values based on condition
        x_boundary = np.zeros_like(x_features, dtype=np.float32)
        for i in range(len(x_features)):
            x_val, t_val = x_features[i]
            x_boundary[i] = 0 if x_val < self.T_m else 1

        y_T = np.full(X.size, Iron.T_a, dtype=np.float32)
        y_B = np.full(X.size, Iron.T_a, dtype=np.float32)  # Match the size of y_T

        # Reshape if needed to match the model's expected input shape
        expected_shape = (len(x_grid) * len(t_grid), 2)
        x_features = x_features.reshape(expected_shape)
        x_boundary = x_boundary.reshape(expected_shape)
        y_T = y_T.reshape(len(x_grid) * len(t_grid))
        y_B = y_B.reshape(len(x_grid) * len(t_grid))

        return x_features, y_T, y_B, x_boundary, x_grid, t_grid

    def calcTemperature3(self, x_arr, tmax, cls, phase_mask_array=None, boundary_mask_array=None):
        nx = len(x_arr)  # Number of spatial points
        t_steps = int(np.ceil((tmax - cls.dt) / cls.dt))  # Total time steps

        # Initialize the temperature array
        T_arr = np.full((nx, t_steps), cls.T_a, dtype=np.float32)
        T_arr[0, :] = cls.T_m  # Apply boundary condition at the left

        # Initialize mask arrays with correct dimensions
        if phase_mask_array is None:
            phase_mask_array = np.zeros((nx, t_steps))
        if boundary_mask_array is None:
            boundary_mask_array = np.zeros((nx, t_steps))

        dx_squared = cls.dx ** 2

        for t_idx in range(1, t_steps):
            T_old = T_arr[:, t_idx - 1]
            T_new = T_old.copy()

            # Calculate k and k_dx_squared dynamically for each spatial point
            k_values = np.array([self.calcThermalConductivity(T) for T in T_old])
            k_dx_squared_values = k_values / dx_squared

            # Vectorized update for temperature values
            for i in range(1, nx - 1):
                if boundary_mask_array[i, t_idx - 1] == 0:  # Check if it's an interior point
                    T_new[i] = T_old[i] + cls.dt * k_dx_squared_values[i] * (T_old[i - 1] - 2 * T_old[i] + T_old[i + 1])

            # Update phase mask array
            updated_mask = np.where(T_new < cls.T_m, 0, np.where(T_new > cls.T_m, 1, 2))
            phase_mask_array[:, t_idx] = updated_mask

            T_arr[:, t_idx] = T_new

            # Debug: Print max and min temperatures for the current time step
            print(f"Time Step {t_idx}, Max T_new: {np.max(T_new)}, Min T_new: {np.min(T_new)}")

        return T_arr, phase_mask_array, boundary_mask_array

    def implicitSol(self, x_arr, tmax, cls, phase_mask_array=None, boundary_mask_array=None):
        if boundary_mask_array is None:
            phase_mask_array, boundary_mask_array = compute_mask_arrays(x_arr, x_arr, max(x_arr), int(tmax / self.dt))

        num_segments = len(x_arr)
        t_arr = [self.dt]  # Time array
        T_arr = np.full((num_segments, len(t_arr)), cls.T_a)  # Temperature array
        H_arr = np.ones((num_segments, 1)) * cls.calcEnthalpy2(T_arr, cls)  # Enthalpy array

        t = self.dt
        while t < tmax:
            T_old = T_arr[:, -1]
            H_old = H_arr[:, -1]
            A = np.zeros((num_segments, num_segments))  # Coefficient matrix
            b = np.zeros(num_segments)  # Right-hand side vector

            # Debug: Print the current time step and the maximum and minimum temperatures
            print(f"Time step: {t}, Max T_old: {np.max(T_old)}, Min T_old: {np.min(T_old)}")

            for i in range(1, num_segments - 1):
                if boundary_mask_array[i, -1] == 0:  # Interior points
                    k = cls.calcThermalConductivity(T_old[i])
                    c = cls.calcSpecificHeat(T_old[i])
                    alpha = self.alpha(k, c, cls.rho)
                    lmbda = cls.dt / cls.dx ** 2 * alpha

                    # Debug: Print alpha and lambda values
                    print(f"alpha: {alpha}, lambda: {lmbda}")

                    A[i, i - 1] = -lmbda
                    A[i, i] = 1 + 2 * lmbda
                    A[i, i + 1] = -lmbda
                    b[i] = T_old[i]

            # Boundary Conditions
            # Left boundary
            if boundary_mask_array[0, -1] == 1:
                A[0, 0] = 1
                b[0] = cls.T_m  # Replace with actual boundary value

            # Right boundary
            if boundary_mask_array[-1, -1] == 1:
                A[-1, -1] = 1
                b[-1] = cls.T_a  # Replace with actual boundary value

            # Debug: Print the coefficient matrix and vector before LU decomposition
            print(f"Matrix A before LU decomposition:\n{A}")
            print(f"Vector b before LU decomposition: {b}")

            try:
                # Perform LU decomposition
                lu, piv = lu_factor(A)
                # Solve the system of equations
                T_new = lu_solve((lu, piv), b)

                # Debug: Print the updated temperature array
                print(f"T_new: {T_new}")

                # Calculate new enthalpy based on updated temperature
                H_new = cls.calcEnthalpy2(T_new, cls)

                # Update arrays with new values
                T_arr = np.hstack((T_arr, T_new[:, np.newaxis]))
                H_arr = np.hstack((H_arr, H_new[:, np.newaxis]))
                mask_new = np.where(T_new < cls.T_m, 0, np.where(T_new > cls.T_m, 1, 2))
                phase_mask_array = np.hstack((phase_mask_array, mask_new[:, np.newaxis]))

                t += cls.dt
                t_arr.append(t)
            except Exception as e:
                # Debug: Print the exception message and break from the loop
                print(f"An error occurred: {e}")
                break

        return T_arr, H_arr, np.array(t_arr), phase_mask_array, boundary_mask_array

    def calcThermalConductivity(self, temp):
        k = 0.95 - 0.64 * 10 ** -3 * temp + 0.67 * 1e-6 * temp ** 2
        print(f"Calculating k for temp={temp}: k = {k}")
        return k

    def calcSpecificHeat(self, temp):
        return np.where(temp < Iron.T_m, Iron.c_solid, Iron.c_liquid)
