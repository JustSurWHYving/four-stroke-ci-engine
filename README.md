# Code Components

## Combustion Modeling

This section describes the combustion modeling implemented in the code.

**Objective:** Model the combustion process inside the engine cylinder.

**Features:**

*   **Wiebe Function Implementation:**  The code utilizes the Wiebe function to model the heat release rate during combustion. This allows for flexible control over combustion parameters such as duration and shape. The Wiebe function is defined as:

    $$x = 1 - exp[-a(\frac{\theta - \theta_0}{\Delta \theta})^m]$$

    where:
    *   $x$ is the mass fraction burned
    *   $a$ is the efficiency parameter
    *   $\theta_0$ is the start of combustion
    *   $\Delta \theta$ is the combustion duration
    *   $m$ is the form factor

*   **Zero-Dimensional Combustion Model:** A zero-dimensional model is employed, assuming uniform conditions within the cylinder.  The code solves for temperature and pressure changes based on the calculated heat release.

*   **Fuel Injection Modeling (Diesel):** For diesel engine simulations, the code includes a fuel injection model.  This allows the user to specify injection parameters such as start time, duration, and injection rate, and to analyze their impact on the combustion process.

## Heat Transfer Analysis

This section describes the heat transfer analysis implemented in the code.

**Objective:** Model heat transfer from the combustion gases to the cylinder walls.

**Features:**

*   **Convective Heat Transfer:**  The code calculates convective heat transfer using empirical correlations. Specifically, the Woschni correlation is implemented to determine the convective heat transfer coefficient.

*   **Radiative Heat Transfer:** Radiative heat transfer from the combustion gases (including soot particles) to the cylinder walls is modeled. This accounts for the energy radiated from the hot gases to the cooler cylinder walls.

*   **Cylinder Wall Temperature Prediction:** The code solves for the temperature distribution within the cylinder wall. Depending on the chosen model complexity, this can be a one-dimensional (1D) or two-dimensional (2D) solution. Finite difference or finite volume methods are used to solve the heat conduction equation within the wall.

