import math
from abc import ABC, abstractmethod
import numpy as np


class PCM(ABC):
    T = 273

    @abstractmethod
    def calcThermalConductivity(self, temp):
        pass

    @abstractmethod
    def calcSpecificHeat(self, temp):
        pass

    def alpha(self, k, c, rho):
        return k / (c * rho)

    def calcTemperature(self, x_arr, t_arr, T_arr, cls):
        for j in range(len(t_arr) - 1):
            for i in range(1, len(x_arr) - 1):
                if x_arr[i] != 0 and t_arr[j] != 0:
                    cls.k = cls.calcThermalConductivity(T_arr[i, j])
                    cls.c = cls.calcSpecificHeat(T_arr[i, j])
                    cls.alpha2 = self.alpha(cls.k, cls.c, cls.rho)
                    cls.lmbda *= cls.alpha2
                    T_arr[i, j + 1] = T_arr[i, j] * (1 - 2 * cls.lmbda) + cls.lmbda * \
                                      (T_arr[i + 1, j] + T_arr[i - 1, j])
        return T_arr

    def calcTemperature3(self, x_arr, tmax, cls):
        lmbda = cls.dt / cls.dx ** 2
        t_arr = np.zeros(1)
        T_vec = np.ones_like(x_arr) * cls.T_m
        T_vec[0] = PCM.T  # set the left boundary to 0
        # T_vec[-1] = PCM.T_m  # set the right boundary to 0
        T_arr = np.ones((len(x_arr), 1)) * cls.T_m
        with open("matrix.txt", "a+") as f:
            j = 0
            while t_arr[j] <= tmax:
                T_new = T_vec.copy()
                for i in range(1, len(T_vec) - 1):
                    k = cls.calcThermalConductivity(T_vec[i])
                    c = cls.calcSpecificHeat(T_vec[i])
                    alpha = self.alpha(k, c, cls.rho)
                    # lmbda *= alpha
                    # add if statement with stability condition requirement
                    while (1 - 2 * lmbda) < 0:
                        print(f'halved time step from {cls.dt} to {cls.dt / 2}')
                        cls.dt /= 2
                        lmbda = (alpha * cls.dt) / cls.dx ** 2
                    T_new[i] = T_vec[i] * (1 - 2 * lmbda) + lmbda * (T_vec[i + 1] + T_vec[i - 1])

                T_vec = T_new.copy()
                T_arr = np.column_stack((T_arr, T_vec))
                f.write(f'T[t={t_arr[j]:.1f}]:\n{T_vec}\n')
                print(f'T[t={t_arr[j]:.1f}]:\n{T_vec}')
                t_arr = np.append(t_arr, t_arr[j] + cls.dt)
                j += 1
        print(f't_arr = {t_arr}')
        print(f'T_arr = {T_arr}')
        t_arr2 = np.linspace(0, tmax, len(T_arr[0]))
        # self.alpha2 = alpha
        return T_vec, T_arr, t_arr2

    def calcEnthalpy2(self, temp, cls):
        if temp >= cls.T_m:
            E = (temp - cls.T_m) * cls.rho * cls.calcSpecificHeat(temp)
        else:
            E = cls.calcSpecificHeat(temp) * (cls.T_m - temp)
        return E

    def explicitSol(self, x_val, t_val, cls):
        cls.k = cls.calcThermalConductivity(cls.T_m)
        cls.c = cls.calcSpecificHeat(cls.T_m)
        cls.alpha2 = self.alpha(cls.k, cls.c, cls.rho)
        return cls.T_m - (cls.T_m - PCM.T) * math.erf(x_val / (2 * math.sqrt(cls.alpha2 * t_val)))


class customPCM(PCM):
    T_m, LH, k, c, rho = None, None, None, None, None

    def __init__(self, k, c, rho, T_m, LH):
        customPCM.k, customPCM.c, customPCM.rho, customPCM.T_m, customPCM.LH = k, c, rho, T_m, LH
        self.dx = 1
        self.alpha2 = self.alpha(customPCM.k, customPCM.c, customPCM.rho)
        self.dt = (0.4 * self.dx ** 2)
        print(f'alpha = {self.alpha2}')
        self.lmbda = self.dt / self.dx ** 2

    def calcThermalConductivity(self, temp):
        return customPCM.k

    def calcSpecificHeat(self, temp):
        return customPCM.c


class Regolith(PCM):
    T_m, LH, rho = 1373, 1429, 1.7

    def __init__(self):
        self.dx = 1
        self.k = self.calcThermalConductivity(Regolith.T_m)
        self.c = self.calcSpecificHeat(Regolith.T_m)
        self.alpha2 = self.alpha(self.k, self.c, Regolith.rho)
        # self.dt = (0.4 * self.dx ** 2) * self.alpha2
        self.dt = (0.4 * self.dx ** 2)
        print(f'alpha = {self.alpha2}')
        self.lmbda = self.dt / self.dx ** 2

    def calcThermalConductivity(self, temp):
        return 0.001561 + 5.426 * 1e-11 * temp ** 3

    def calcSpecificHeat(self, temp):
        ca = -2.32 * 1e-2
        cb = 2.13 * 1e-3
        cc = 1.5 * 1e-5
        cd = -7.37 * 1e-8
        ce = 9.66 * 1e-11
        return ca + cb * temp + cc * temp ** 2 + cd * temp ** 3 + ce * temp ** 4


class Iron(PCM):
    T_m, LH, rho = 1810, 13800, 7.67

    def __init__(self):
        self.dx = 1
        self.k = self.calcThermalConductivity(Iron.T_m)
        self.c = self.calcSpecificHeat(Iron.T_m)
        self.alpha2 = self.alpha(self.k, self.c, Iron.rho)
        self.dt = (0.4 * self.dx ** 2) * self.alpha2
        # self.dt = (0.4 * self.dx ** 2)
        print(f'alpha = {self.alpha2}')
        self.lmbda = self.dt / self.dx ** 2

    def calcThermalConductivity(self, temp):
        return 80.2

    def calcSpecificHeat(self, temp):
        return 0.45