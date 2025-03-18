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

