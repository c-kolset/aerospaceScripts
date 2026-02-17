# aerospace

This repository is a fork from dreasttom/aerospace repository. This repository contains various Python scripts for performing aerospace calculations. This is meant for students so some are simple, while others are more complex.




## Eulersequations.py
---

Simple 1D Euler Equations Demo for Hypersonics (Teaching Script)

This script:
- Solves the 1D compressible Euler equations (inviscid flow, ideal gas).
- Uses a finite-volume scheme with Rusanov (local Lax–Friedrichs) flux.
- Models a "shock tube" style problem (left/right initial states).
- Provides a Tkinter GUI so students can choose:
    * Left/right density, velocity, pressure
    * Ratio of specific heats gamma
    * Number of grid cells
    * Final simulation time

It is intentionally:
- Not production quality (but numerically reasonable).
- Highly commented for teaching.
- Includes basic error handling and validation.

Dependencies:
    pip install numpy matplotlib

Tkinter is part of the standard Python library on most systems.


### Numerical Core: 1D Euler Equations

`primative_to_concerved()`
    Convert primitive variables (rho, u, p) to conserved variables (U).

    Euler equations (1D):
        U = [rho, rho*u, E]^T

    where total energy per unit volume:
        E = p/(gamma - 1) + 0.5 * rho * u^2


`conserved_to_primitive()`
    Convert conserved variables U = [rho, rho*u, E] to primitive (rho, u, p).

    Includes basic sanity checks to avoid division by zero or negative pressure.

`flux()`
    Compute the physical flux F(U) for the 1D Euler equations.

    U = [rho, rho*u, E]
    F = [rho*u, rho*u^2 + p, u*(E + p)]


`sound_speed()`
    Speed of sound for an ideal gas: a = sqrt(gamma * p / rho)

`rusanov_flux()`
    Rusanov (local Lax–Friedrichs) numerical flux at an interface.

    Given left state UL and right state UR (conserved variables),
    we compute:

        F_hat = 0.5*(F(UL) + F(UR)) - 0.5*lambda_max*(UR - UL)

    where lambda_max is the maximum wave speed at the interface.

    This is simple and quite diffusive, but very robust for teaching.

`run_euler_1d()`
    Run a simple 1D Euler simulation of a shock-tube style problem.

    Parameters
    ----------
    rhoL, uL, pL : float
        Left state primitive variables.
    rhoR, uR, pR : float
        Right state primitive variables.
    gamma : float
        Ratio of specific heats (1.4 for air).
    nx : int
        Number of grid cells.
    x_domain : (float, float)
        Spatial domain [x_start, x_end].
    final_time : float
        Final simulation time.
    cfl : float
        Courant number for time-step stability.

    Returns
    -------
    x_centers : np.ndarray
        Cell center locations.
    rho, u, p : np.ndarray
        Arrays of density, velocity, pressure at final time.
