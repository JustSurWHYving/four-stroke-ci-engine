"""
This module contains functions and classes for modelling combustion processes.

It includes a CombustionModel class that simulates the combustion process in a cylinder
using the Wiebe function and differential equations.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class CombustionModel:
    def __init__(self):
        # Engine parameters
        self.bore = 0.08  # m
        self.stroke = 0.1  # m
        self.con_rod_length = 0.15  # m
        self.compression_ratio = 18.0
        self.engine_speed = 2000  # rpm
        
        # Combustion parameters
        self.wiebe_a = 5.0  # Efficiency parameter
        self.wiebe_m = 2.0  # Form factor
        self.start_of_combustion = -10.0  # degrees before TDC
        self.combustion_duration = 60.0  # degrees
        
        # Fuel properties
        self.fuel_lower_heating_value = 42.5e6  # J/kg
        self.fuel_mass_per_cycle = 2.0e-5  # kg/cycle
        
        # Thermodynamic properties
        self.gamma = 1.35  # Specific heat ratio
        self.R = 287.0  # Gas constant J/(kg·K)
        
        # Initial conditions
        self.p_initial = 101325.0  # Pa
        self.T_initial = 350.0  # K
        
        # Calculation range
        self.theta_start = -180.0  # degrees
        self.theta_end = 180.0  # degrees
        
    def wiebe_function(self, theta):
        """
        Calculate the mass fraction burned using the Wiebe function
        
        Parameters:
        theta (float): Crank angle in degrees
        
        Returns:
        float: Mass fraction burned (0 to 1)
        """
        if theta < self.start_of_combustion:
            return 0.0
        
        x = (theta - self.start_of_combustion) / self.combustion_duration
        
        if x <= 0:
            return 0.0
        elif x >= 1:
            return 1.0
        else:
            return 1.0 - np.exp(-self.wiebe_a * (x ** self.wiebe_m))
    
    def heat_release_rate(self, theta):
        """
        Calculate the heat release rate at a given crank angle
        
        Parameters:
        theta (float): Crank angle in degrees
        
        Returns:
        float: Heat release rate in J/degree
        """
        # Numerical differentiation of Wiebe function
        delta = 0.1
        dxdt = (self.wiebe_function(theta + delta) - self.wiebe_function(theta - delta)) / (2 * delta)
        
        # Total heat release from fuel
        Q_total = self.fuel_mass_per_cycle * self.fuel_lower_heating_value
        
        # Heat release rate
        return Q_total * dxdt
    
    def cylinder_volume(self, theta):
        """
        Calculate the cylinder volume at a given crank angle
        
        Parameters:
        theta (float): Crank angle in degrees
        
        Returns:
        float: Cylinder volume in m^3
        """
        theta_rad = np.radians(theta)
        
        # Clearance volume
        V_clearance = np.pi * (self.bore ** 2) * self.stroke / (4 * (self.compression_ratio - 1))
        
        # Displacement volume
        V_displacement = np.pi * (self.bore ** 2) * self.stroke / 4
        
        # Piston position
        s = self.con_rod_length + self.stroke/2 - (self.stroke/2 * np.cos(theta_rad) + 
                                                 np.sqrt(self.con_rod_length**2 - (self.stroke/2 * np.sin(theta_rad))**2))
        
        # Current volume
        return V_clearance + (np.pi * (self.bore ** 2) * s / 4)
    
    def dVdtheta(self, theta):
        """
        Calculate the rate of change of cylinder volume with respect to crank angle
        
        Parameters:
        theta (float): Crank angle in degrees
        
        Returns:
        float: dV/dθ in m^3/degree
        """
        # Numerical differentiation
        delta = 0.1
        return (self.cylinder_volume(theta + delta) - self.cylinder_volume(theta - delta)) / (2 * delta)
    
    def combustion_model_derivatives(self, theta, y):
        """
        Define the system of differential equations for the combustion model
        
        Parameters:
        theta (float): Crank angle in degrees
        y (array): State variables [pressure, temperature]
        
        Returns:
        array: Derivatives [dp/dθ, dT/dθ]
        """
        p, T = y
        
        # Current volume and rate of change
        V = self.cylinder_volume(theta)
        dVdt = self.dVdtheta(theta)
        
        # Mass in cylinder (assumed constant)
        m = self.p_initial * self.cylinder_volume(self.theta_start) / (self.R * self.T_initial)
        
        # Heat release
        dQdt = self.heat_release_rate(theta)
        
        # Heat transfer to walls (simplified model)
        Q_wall = 0  # Simplified, could be enhanced with heat transfer model
        
        # Specific heat at constant volume
        cv = self.R / (self.gamma - 1)
        
        # Temperature derivative
        dTdt = (1 / (m * cv)) * (dQdt - Q_wall - p * dVdt)
        
        # Pressure derivative (from ideal gas law)
        dpdt = (self.gamma - 1) * dQdt / V - self.gamma * p * dVdt / V
        
        return [dpdt, dTdt]
    
    def solve_combustion_cycle(self):
        """
        Solve the combustion model over a complete cycle
        
        Returns:
        tuple: (theta, pressure, temperature, volume, heat_release, mass_fraction_burned)
        """
        # Initial conditions
        y0 = [self.p_initial, self.T_initial]
        
        # Solve the system of differential equations
        theta_range = np.linspace(self.theta_start, self.theta_end, 1000)
        
        solution = solve_ivp(
            self.combustion_model_derivatives,
            [self.theta_start, self.theta_end],
            y0,
            t_eval=theta_range,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        theta = solution.t
        pressure = solution.y[0]
        temperature = solution.y[1]
        
        # Calculate additional results
        volume = np.array([self.cylinder_volume(t) for t in theta])
        heat_release = np.array([self.heat_release_rate(t) for t in theta])
        mass_fraction_burned = np.array([self.wiebe_function(t) for t in theta])
        
        return theta, pressure, temperature, volume, heat_release, mass_fraction_burned

    def plot_results(self):
        """
        Plot the results of the combustion model
        """
        theta, pressure, temperature, volume, heat_release, mass_fraction_burned = self.solve_combustion_cycle()
        
        # Pressure plot
        plt.figure(figsize=(8, 6))
        plt.plot(theta, pressure / 1e6)
        plt.xlabel('Crank Angle (degrees)')
        plt.ylabel('Pressure (MPa)')
        plt.title('Cylinder Pressure')
        plt.grid(True)
        plt.show()
        
        # Temperature plot
        plt.figure(figsize=(8, 6))
        plt.plot(theta, temperature)
        plt.xlabel('Crank Angle (degrees)')
        plt.ylabel('Temperature (K)')
        plt.title('Cylinder Temperature')
        plt.grid(True)
        plt.show()
        
        # Heat release rate plot
        plt.figure(figsize=(8, 6))
        plt.plot(theta, heat_release)
        plt.xlabel('Crank Angle (degrees)')
        plt.ylabel('Heat Release Rate (J/deg)')
        plt.title('Heat Release Rate')
        plt.grid(True)
        plt.show()
        
        # Mass fraction burned plot
        plt.figure(figsize=(8, 6))
        plt.plot(theta, mass_fraction_burned)
        plt.xlabel('Crank Angle (degrees)')
        plt.ylabel('Mass Fraction Burned')
        plt.title('Combustion Progress')
        plt.grid(True)
        plt.show()
        
        # P-V diagram (logarithmic scale)
        plt.figure(figsize=(8, 6))
        plt.plot(volume, pressure)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Volume (m³)')
        plt.ylabel('Pressure (Pa)')
        plt.title('Actual P-V Diagram (Log Scale)')
        plt.grid(True)
        plt.show()
