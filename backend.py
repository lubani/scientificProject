import math
from abc import ABC, abstractmethod
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.special import erf


class PCM(ABC):


    @abstractmethod
    def calcThermalConductivity(self, temp):
        pass

    @abstractmethod
    def calcSpecificHeat(self, temp):
        pass

    def alpha(self, k, c, rho):
        # Check for division by zero or small number
        if c * rho == 0 or k == 0:
            raise ValueError("c * rho should not be zero")
        return k / (c * rho)

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
    def implicitSol(self, x_arr, tmax, cls):
        pass

    @abstractmethod
    def calcTemperature3(self, x_arr, tmax, cls):
        pass
        # nx = len(x_arr)
        # cls.dt = max(min(cls.dt, 0.5 * cls.dx ** 2 * cls.alpha2), 0.001)
        #
        # T_arr = np.full((nx, 1), cls.T_a, dtype=np.float64)
        # T_arr[0, 0] = cls.T_m + 100
        # H_arr = self.calcEnthalpy2(T_arr[:, 0], cls).reshape(-1, 1)
        # current_time = cls.dt
        # t_arr = np.array([current_time], dtype=np.float64)
        #
        # heat_source_max = 5  # W/m^2, solar constant
        # dx = cls.dx
        # dx_squared = dx ** 2
        # day_duration = 14 * 24 * 60  # in minutes
        # k_dx_squared = cls.k / dx_squared
        #
        # while current_time < tmax:
        #     dt = cls.dt
        #     dt = max(min(dt, 0.5 * dx_squared * cls.alpha2), 0.001)
        #     H_old = H_arr[:, -1]
        #     H_new = H_old.copy()
        #
        #     heat_source = heat_source_max if current_time % (2 * day_duration) < day_duration else 0
        #     flux = heat_source / cls.k
        #
        #     # Neumann boundary condition at x=0 and x=nx-1
        #     common_term = dt * k_dx_squared
        #     H_new[0] = H_old[0] + common_term * (2 * H_old[1] - 2 * H_old[0] + 2 * flux * dx)
        #     H_new[-1] = H_old[-1] + common_term * (2 * H_old[-2] - 2 * H_old[-1])
        #
        #     # Finite difference equation for enthalpy
        #     H_new[1:nx - 1] = H_old[1:nx - 1] + common_term * (H_old[:-2] - 2 * H_old[1:-1] + H_old[2:])
        #
        #     T_new = self.inverseEnthalpy2(H_new, cls)
        #
        #     # Dynamically add new cells to T_arr, H_arr, and t_arr
        #     T_arr = np.hstack((T_arr, T_new.reshape(-1, 1)))
        #     H_arr = np.hstack((H_arr, H_new.reshape(-1, 1)))
        #     t_arr = np.hstack((t_arr, (current_time,)))
        #
        #     current_time += dt
        #
        # return T_arr, t_arr

    def inverseEnthalpy2(self, H, cls):
        T = np.full_like(H, cls.T_m)
        T[H < 0] = cls.T_a + H[H < 0] / (cls.rho * cls.c)
        T[H > cls.LH] = cls.T_a + (H[H > cls.LH] - cls.LH) / (cls.rho * cls.c)
        return T

    def initial_enthalpy(self, x_arr, cls):
        # Initialize the enthalpy array
        H_arr = np.ones_like(x_arr)

        # Set the reference temperature
        T_ref = cls.T_m  # Adjust this as needed

        # Calculate the enthalpy
        for i, x in enumerate(x_arr):
            if cls.T_a > cls.T_m:
                H_arr[i] = cls.c * (cls.T_m - T_ref) + cls.LH + cls.c * (cls.T_a - cls.T_m)
            else:
                H_arr[i] = cls.c * (cls.T_a - T_ref)

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
        H = np.ones_like(T)

        mask_below_melting = T < cls.T_m
        print("Below Melting:", T[mask_below_melting])
        H[mask_below_melting] = cls.rho * cls.c * (T[mask_below_melting] - cls.T_a)

        mask_at_melting = T == cls.T_m
        print("At Melting:", T[mask_at_melting])
        H[mask_at_melting] = cls.LH

        mask_above_melting = T > cls.T_m
        print("Above Melting:", T[mask_above_melting])
        H[mask_above_melting] = cls.LH + cls.rho * cls.c * (T[mask_above_melting] - cls.T_m)

        print("Enthalpy:", H)
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

    def explicitSol(self, x_val, t_arr, cls):
        T_initial = cls.T_a
        T_final = cls.T_m

        T = np.full((len(x_val), len(t_arr)), T_initial)
        for t_idx, t_val in enumerate(t_arr):
            for x_idx, x in enumerate(x_val):
                alpha2 = self.alpha(cls.k, cls.c, cls.rho)
                T[x_idx, t_idx] = T_initial + (T_final - T_initial) * (
                        1 - erf(x / (2 * alpha2 * np.sqrt(t_val))))

        return T

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
        self.dx = 0.1
        self.k = self.calcThermalConductivity(Regolith.T_m)
        self.c = self.calcSpecificHeat(Regolith.T_m)
        self.alpha2 = self.alpha(self.k, self.c, Regolith.rho)
        self.dt = (0.4 * self.dx ** 2) / self.alpha2
        print("dt = ", self.dt)
        print(f'alpha = {self.alpha2}')
        self.lmbda = self.dt / (self.dx ** 2)

    def calcTemperature3(self, x_arr, tmax, cls):
        nx = len(x_arr)
        cls.dt = max(min(cls.dt, 0.5 * cls.dx ** 2 * cls.alpha2), 0.001)
        print("dt = ", cls.dt)
        T_arr = np.full((nx, 1), cls.T_a, dtype=np.float64)
        T_arr[0, :] = 1800
        H_arr = self.initial_enthalpy(x_arr, cls).reshape(-1, 1)  # Initialize enthalpy
        print("initial enthalpy = ", H_arr)
        current_time = cls.dt
        t_arr = np.array([current_time], dtype=np.float64)

        heat_source_max = 1  # W/m^2, solar constant
        day_duration = 14 * 24 * 60  # in minutes

        while current_time < tmax:
            print("current time = ", current_time)
            H_old = H_arr[:, -1]
            print("H_old = ", H_old)
            H_new = H_old.copy()

            # Heat source effect at x=0
            heat_source = heat_source_max if current_time % (2 * day_duration) < day_duration else 0
            T_left = self.T_m + heat_source / (cls.rho * cls.c)  # Modify this equation as needed
            print("heat_source = ", heat_source)
            flux = heat_source / cls.k
            print("flux = ", flux)
            H_new[0] = T_left * cls.rho * cls.c if T_left < cls.T_m else T_left * cls.rho * cls.c + cls.LH

            # Neumann boundary condition at x=nx-1 (zero flux)
            H_new[-1] = H_old[-1]

            # Finite difference equation for enthalpy in the interior
            for i in range(1, nx - 1):
                H_new[i] = H_old[i] + cls.dt * cls.k * (H_old[i - 1] - 2 * H_old[i] + H_old[i + 1]) / cls.dx ** 2

            # Compute temperature based on enthalpy
            T_new = np.full_like(H_new, cls.T_a, dtype=np.float64)

            for i, h in enumerate(H_new):
                if h < 0:
                    T_new[i] = max(cls.T_a, cls.T_m + h / (cls.rho * cls.c))
                elif 0 <= h <= cls.LH:
                    T_new[i] = cls.T_m
                elif h > cls.LH:
                    T_new[i] = cls.T_m + (h - cls.LH) / (cls.rho * cls.c)
                    print("T_new[i] = ", T_new[i])

            # Dynamically add new cells to T_arr, H_arr, and t_arr
            T_arr = np.hstack((T_arr, T_new.reshape(-1, 1)))
            print("T_arr = ", T_arr)
            H_arr = np.hstack((H_arr, H_new.reshape(-1, 1)))
            print("H_arr = ", H_arr)
            t_arr = np.hstack((t_arr, (current_time,)))

            current_time += cls.dt

        return T_arr, t_arr

    def implicitSol(self, x_arr, tmax, cls):
        heat_source_max = 200  # Maximum heat source
        T_ambient = cls.T_a  # Ambient temperature

        num_segments = len(x_arr)
        t_arr = [self.dt]  # Time array
        # Initialize the temperature array
        T_arr = np.full((num_segments, len(t_arr)), cls.T_a)
        # T_arr[0, 0] = self.T_m + 100
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

            for i in range(1, num_segments - 1):  # Loop over spatial segments
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

            # Apply Neumann boundary conditions
            A[0, 0] = 1
            A[0, 1] = -1
            b[0] = heat_source * cls.dt / cls.dx  # Heat source applied at the left boundary

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

        return T_arr, np.array(t_arr)
    def calcThermalConductivity(self, temp):
        # Thermal conductivity for granular regolith
        C1 = 1.281e-2
        C2 = 4.431e-2
        k_granular = C1 + C2 * np.float64(temp) ** (-3)

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
        self.k = self.calcThermalConductivity(Iron.T_m)
        self.c = self.calcSpecificHeat(Iron.T_m)
        self.alpha2 = self.alpha(self.k, self.c, Iron.rho)
        self.dt = (0.4 * self.dx ** 2) / self.alpha2
        print("dt = ", self.dt)
        print(f'alpha = {self.alpha2}')
        self.lmbda = self.dt / (self.dx ** 2)

    def calcTemperature3(self, x_arr, tmax, cls):
        nx = len(x_arr)
        T_arr = np.full((nx, 1), cls.T_a, dtype=np.float64)
        T_arr[0, :] = 1800  # Initial temperature condition

        # Parameters
        dx = cls.dx
        dx_squared = dx ** 2
        k_dx_squared = cls.k / dx_squared

        current_time = cls.dt
        t_arr = np.array([current_time], dtype=np.float64)

        while current_time < tmax:
            dt = cls.dt
            dt = max(min(dt, 0.5 * dx_squared * cls.alpha2), 0.001)
            T_old = T_arr[:, -1]
            T_new = T_old.copy()

            # Finite difference equation for temperature (no phase change)
            T_new[1:nx - 1] = T_old[1:nx - 1] + dt * k_dx_squared * (T_old[:-2] - 2 * T_old[1:-1] + T_old[2:])

            # Boundaries (modify as needed for Iron)
            T_new[0] = 1800
            T_new[-1] = cls.T_a

            # Add new temperature to array
            T_arr = np.hstack((T_arr, T_new.reshape(-1, 1)))
            t_arr = np.hstack((t_arr, (current_time,)))

            current_time += dt

        return T_arr, t_arr

    def implicitSol(self, x_arr, tmax, cls):
        heat_source_value = 10

        num_segments = len(x_arr)
        t_arr = [self.dt]  # Time array
        # Initialize the temperature array
        T_arr = np.full((num_segments, len(t_arr)), cls.T_a)
        H_arr = np.ones((num_segments, 1)) * cls.calcEnthalpy2(T_arr, cls)  # Initial enthalpy

        t = self.dt
        while t < tmax:
            T_old = T_arr[:, -1]
            H_old = H_arr[:, -1]

            # Create the coefficient matrix A and the right-hand side vector b
            A = np.zeros((num_segments, num_segments))
            b = np.zeros(num_segments)

            for i in range(1, num_segments - 1):  # Loop over spatial segments
                k = cls.calcThermalConductivity(T_old[i])
                c = cls.calcSpecificHeat(T_old[i])
                alpha = self.alpha(k, c, cls.rho)

                lmbda = cls.dt / cls.dx ** 2 * alpha

                # Fill the coefficient matrix A and the right-hand side vector b
                A[i, i - 1] = -lmbda
                A[i, i] = 1 + 2 * lmbda
                A[i, i + 1] = -lmbda
                b[i] = T_old[i]

            A[0, 0] = 1
            A[0, 1] = -1
            b[0] = heat_source_value * cls.dt / cls.dx ** 2  # Heat source applied at the left boundary

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

        return T_arr, np.array(t_arr)

    def calcThermalConductivity(self, temp):
        k = 0.95 - 0.64 * 10 ** -3 * temp + 0.67 * 1e-6 * temp ** 2
        # print("k = ", k)
        return k

    def calcSpecificHeat(self, temp):
        return np.where(temp < Iron.T_m, Iron.c_solid, Iron.c_liquid)
