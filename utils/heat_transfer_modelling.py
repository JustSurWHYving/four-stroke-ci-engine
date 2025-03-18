
"""
This module provides a class for modelling heat transfer in an internal combustion engine.

The model includes:
- Calculation of cylinder volume based on engine geometry and crank angle.
- Calculation of convective heat transfer coefficient using the Woschni correlation.
- Calculation of radiative heat transfer.
- Calculation of in-cylinder pressure and temperature using the first law of thermodynamics.
- Plotting of results.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class HeatTransferModel:
    def __init__(self):
        # Engine parameters
        self.bore = 0.08  # m
        self.stroke = 0.1  # m
        self.con_rod_length = 0.15  # m
        self.compression_ratio = 18.0
        self.engine_speed = 2000  # rpm
        
        # Wall temperature parameters
        self.T_wall_cylinder = 450  # K
        self.T_wall_piston = 550  # K
        self.T_wall_head = 500  # K
        
        # Woschni correlation parameters
        self.C1 = 2.28  # Gas exchange period
        self.C2 = 0.00324  # Combustion period
        self.p_motored_ref = 1e5  # Pa, reference motored pressure
        
        # Radiative heat transfer parameters
        self.soot_emissivity = 0.9
        self.wall_emissivity = 0.2
        self.stefan_boltzmann = 5.67e-8  # W/(m²·K⁴)
        
        # Thermodynamic properties
        self.gamma = 1.35  # Specific heat ratio
        self.R = 287.0  # Gas constant J/(kg·K)
        
        # Initial conditions
        self.p_initial = 101325.0  # Pa
        self.T_initial = 350.0  # K
        
        # Combustion parameters for Wiebe function
        self.wiebe_a = 5.0  # Efficiency parameter
        self.wiebe_m = 2.0  # Form factor
        self.start_of_combustion = -10.0  # degrees before TDC
        self.combustion_duration = 60.0  # degrees
        self.fuel_lower_heating_value = 42.5e6  # J/kg
        self.fuel_mass_per_cycle = 2.0e-5  # kg/cycle
        
        # Calculation range
        self.theta_start = -180.0  # degrees
        self.theta_end = 180.0  # degrees
        
    def cylinder_volume(self, theta):
        """
        Calculate the cylinder volume at a given crank angle
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
        """
        # Numerical differentiation
        delta = 0.1
        return (self.cylinder_volume(theta + delta) - self.cylinder_volume(theta - delta)) / (2 * delta)
    
    def wiebe_function(self, theta):
        """
        Calculate the mass fraction burned using the Wiebe function
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
        """
        # Numerical differentiation of Wiebe function
        delta = 0.1
        dxdt = (self.wiebe_function(theta + delta) - self.wiebe_function(theta - delta)) / (2 * delta)
        
        # Total heat release from fuel
        Q_total = self.fuel_mass_per_cycle * self.fuel_lower_heating_value
        
        # Heat release rate
        return Q_total * dxdt
    
    def cylinder_surface_area(self, theta):
        """
        Calculate the total surface area of the cylinder (including piston and head)
        """
        # Current volume
        V = self.cylinder_volume(theta)
        
        # Clearance volume
        V_clearance = np.pi * (self.bore ** 2) * self.stroke / (4 * (self.compression_ratio - 1))
        
        # Piston position
        theta_rad = np.radians(theta)
        s = self.con_rod_length + self.stroke/2 - (self.stroke/2 * np.cos(theta_rad) + 
                                                 np.sqrt(self.con_rod_length**2 - (self.stroke/2 * np.sin(theta_rad))**2))
        
        # Bore area
        A_bore = np.pi * self.bore * s
        
        # Piston area
        A_piston = np.pi * (self.bore ** 2) / 4
        
        # Head area (approximated as flat)
        A_head = np.pi * (self.bore ** 2) / 4
        
        return A_bore + A_piston + A_head
    
    def woschni_heat_transfer_coefficient(self, theta, p, T, p_motored):
        """
        Calculate the convective heat transfer coefficient using the Woschni correlation
        
        Parameters:
        theta (float): Crank angle in degrees
        p (float): Cylinder pressure in Pa
        T (float): Gas temperature in K
        p_motored (float): Motored pressure at the same crank angle in Pa
        
        Returns:
        float: Heat transfer coefficient in W/(m²·K)
        """
        # Convert rpm to m/s
        mean_piston_speed = 2 * self.stroke * self.engine_speed / 60  # m/s
        
        # Reference conditions
        V_displacement = np.pi * (self.bore ** 2) * self.stroke / 4
        T_ref = self.T_initial
        p_ref = self.p_initial
        V_ref = self.cylinder_volume(self.theta_start)
        
        # Determine coefficients based on engine cycle
        if self.start_of_combustion <= theta <= (self.start_of_combustion + self.combustion_duration):
            # Combustion period
            C1 = 2.28
            C2 = 0.00324
        else:
            # Gas exchange period
            C1 = 2.28
            C2 = 0.0
        
        # Calculate characteristic velocity
        gas_velocity = C1 * mean_piston_speed + C2 * (V_displacement * T_ref) / (p_ref * V_ref) * (p - p_motored)
        
        # Woschni correlation
        return 3.26 * (self.bore ** -0.2) * (p ** 0.8) * (T ** -0.55) * (gas_velocity ** 0.8)
    
    def radiative_heat_transfer(self, T_gas, T_wall):
        """
        Calculate radiative heat transfer rate from gas to wall
        
        Parameters:
        T_gas (float): Gas temperature in K
        T_wall (float): Wall temperature in K
        
        Returns:
        float: Radiative heat transfer rate in W/m²
        """
        # Effective emissivity
        e_eff = self.soot_emissivity * self.wall_emissivity
        
        # Stefan-Boltzmann law
        q_rad = e_eff * self.stefan_boltzmann * (T_gas**4 - T_wall**4)
        
        return q_rad
    
    def total_heat_transfer_rate(self, theta, p, T, p_motored):
        """
        Calculate the total heat transfer rate from gas to walls
        
        Parameters:
        theta (float): Crank angle in degrees
        p (float): Cylinder pressure in Pa
        T (float): Gas temperature in K
        p_motored (float): Motored pressure at the same crank angle in Pa
        
        Returns:
        float: Total heat transfer rate in W
        """
        # Convective heat transfer coefficient
        h_conv = self.woschni_heat_transfer_coefficient(theta, p, T, p_motored)
        
        # Surface area
        A = self.cylinder_surface_area(theta)
        
        # Average wall temperature
        T_wall_avg = (self.T_wall_cylinder + self.T_wall_piston + self.T_wall_head) / 3
        
        # Convective heat transfer
        q_conv = h_conv * A * (T - T_wall_avg)
        
        # Radiative heat transfer
        q_rad = self.radiative_heat_transfer(T, T_wall_avg) * A
        
        # Total heat transfer
        return q_conv + q_rad
    
    def generate_motored_pressure_curve(self):
        """
        Generate a motored pressure curve (no combustion)
        """
        theta_range = np.linspace(self.theta_start, self.theta_end, 1000)
        p_motored = np.zeros_like(theta_range)
        
        # Initial conditions
        p0 = self.p_initial
        V0 = self.cylinder_volume(self.theta_start)
        
        # Polytropic compression/expansion
        for i, theta in enumerate(theta_range):
            V = self.cylinder_volume(theta)
            p_motored[i] = p0 * (V0 / V) ** self.gamma
            
        return theta_range, p_motored
    
    def combustion_model_with_heat_transfer(self):
        """
        Solve the combustion model with heat transfer
        """
        # Generate motored pressure curve
        theta_motored, p_motored = self.generate_motored_pressure_curve()
        
        # Initial conditions
        y0 = [self.p_initial, self.T_initial]
        
        # Function to interpolate motored pressure
        def get_motored_pressure(theta):
            return np.interp(theta, theta_motored, p_motored)
        
        # Define ODE system
        def derivatives(theta, y):
            p, T = y
            
            # Current volume and rate of change
            V = self.cylinder_volume(theta)
            dVdt = self.dVdtheta(theta)
            
            # Mass in cylinder (assumed constant)
            m = self.p_initial * self.cylinder_volume(self.theta_start) / (self.R * self.T_initial)
            
            # Heat release from combustion
            dQdt_combustion = self.heat_release_rate(theta)
            
            # Heat transfer to walls
            p_motored = get_motored_pressure(theta)
            dQdt_heat_transfer = self.total_heat_transfer_rate(theta, p, T, p_motored)
            
            # Net heat release
            dQdt_net = dQdt_combustion - dQdt_heat_transfer
            
            # Specific heat at constant volume
            cv = self.R / (self.gamma - 1)
            
            # Temperature derivative
            dTdt = (1 / (m * cv)) * (dQdt_net - p * dVdt)
            
            # Pressure derivative (from ideal gas law)
            dpdt = (self.gamma - 1) * dQdt_net / V - self.gamma * p * dVdt / V
            
            return [dpdt, dTdt]
        
        # Solve ODE system
        theta_range = np.linspace(self.theta_start, self.theta_end, 1000)
        solution = solve_ivp(
            derivatives,
            [self.theta_start, self.theta_end],
            y0,
            t_eval=theta_range,
            method='RK45',
            rtol=1e-6,
            atol=1e-9
        )
        
        # Extract results
        theta = solution.t
        pressure = solution.y[0]
        temperature = solution.y[1]
        
        # Calculate heat transfer
        heat_transfer = np.zeros_like(theta)
        for i, (t, p, T) in enumerate(zip(theta, pressure, temperature)):
            p_motored = get_motored_pressure(t)
            heat_transfer[i] = self.total_heat_transfer_rate(t, p, T, p_motored)
        
        return theta, pressure, temperature, heat_transfer
    
    def wall_temperature_distribution(self, q_heat_transfer, wall_thickness=0.01, nodes=20):
        """
        Calculate the temperature distribution in the cylinder wall
        
        Parameters:
        q_heat_transfer (float): Heat flux through the wall (W/m²)
        wall_thickness (float): Wall thickness in meters
        nodes (int): Number of nodes for finite difference calculation
        
        Returns:
        array: Temperature distribution across the wall thickness
        """
        # Thermal conductivity of cast iron (W/(m·K))
        k_wall = 50
        
        # Coolant temperature (K)
        T_coolant = 360
        
        # Heat transfer coefficient to coolant (W/(m²·K))
        h_coolant = 5000
        
        # Node spacing
        dx = wall_thickness / (nodes - 1)
        
        # Temperature array
        T_wall = np.zeros(nodes)
        
        # Boundary conditions
        T_wall[0] = self.T_wall_cylinder  # Inner wall
        
        # Heat conduction equation
        for i in range(1, nodes):
            T_wall[i] = T_wall[i-1] - (q_heat_transfer * dx) / k_wall
        
        # Boundary condition at coolant interface
        T_wall[-1] = (T_wall[-2] + (h_coolant * dx / k_wall) * T_coolant) / (1 + h_coolant * dx / k_wall)
        
        return np.linspace(0, wall_thickness, nodes), T_wall
    
    def plot_results(self):
        """
        Plot the results of the heat transfer analysis
        """
        # Run combustion model with heat transfer
        theta, pressure, temperature, heat_transfer = self.combustion_model_with_heat_transfer()
        
        # Create plots
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
        plt.title('Gas Temperature')
        plt.grid(True)
        plt.show()
        
        # Heat transfer rate plot
        plt.figure(figsize=(8, 6))
        plt.plot(theta, heat_transfer / 1e3)
        plt.xlabel('Crank Angle (degrees)')
        plt.ylabel('Heat Transfer Rate (kW)')
        plt.title('Heat Transfer Rate')
        plt.grid(True)
        plt.show()
        
        # Wall temperature distribution at peak heat transfer
        max_heat_idx = np.argmax(heat_transfer)
        max_heat_flux = heat_transfer[max_heat_idx] / self.cylinder_surface_area(theta[max_heat_idx])
        
        wall_pos, wall_temp = self.wall_temperature_distribution(max_heat_flux)
        
        plt.figure(figsize=(8, 6))
        plt.plot(wall_pos * 1000, wall_temp)
        plt.xlabel('Wall Position (mm)')
        plt.ylabel('Temperature (K)')
        plt.title(f'Wall Temperature Distribution at θ = {theta[max_heat_idx]:.1f}°')
        plt.grid(True)
        plt.show()
        
        # Cumulative heat transfer
        cumulative_heat = np.cumsum(heat_transfer) * (theta[1] - theta[0]) * np.pi / 180
        
        plt.figure(figsize=(8, 6))
        plt.plot(theta, cumulative_heat / 1e3)
        plt.xlabel('Crank Angle (degrees)')
        plt.ylabel('Cumulative Heat Transfer (kJ)')
        plt.title('Cumulative Heat Transfer')
        plt.grid(True)
        plt.show()

        # Heat transfer coefficient
        h_conv = np.zeros_like(theta)
        theta_motored, p_motored = self.generate_motored_pressure_curve()
        
        for i, (t, p, T) in enumerate(zip(theta, pressure, temperature)):
            p_mot = np.interp(t, theta_motored, p_motored)
            h_conv[i] = self.woschni_heat_transfer_coefficient(t, p, T, p_mot)
            
        plt.figure(figsize=(8, 6))
        plt.plot(theta, h_conv)
        plt.xlabel('Crank Angle (degrees)')
        plt.ylabel('Heat Transfer Coefficient (W/m²·K)')
        plt.title('Woschni Heat Transfer Coefficient')
        plt.grid(True)
        plt.show()
