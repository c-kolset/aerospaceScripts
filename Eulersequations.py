"""
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
"""

import sys
import traceback

# --- Third-party imports with error handling ---------------------------------
try:
    import numpy as np
except ImportError:
    print("ERROR: NumPy is required. Install it with 'pip install numpy'.")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: Matplotlib is required. Install it with 'pip install matplotlib'.")
    sys.exit(1)

# Tkinter (GUI)
try:
    import tkinter as tk
    from tkinter import messagebox
except ImportError:
    print("ERROR: Tkinter is not available. On some systems you may need to install "
          "it separately (e.g., 'sudo apt-get install python3-tk').")
    sys.exit(1)


# =============================================================================
#  NUMERICAL CORE: 1D EULER EQUATIONS
# =============================================================================

def primitive_to_conserved(rho, u, p, gamma):
    """
    Convert primitive variables (rho, u, p) to conserved variables (U).

    Euler equations (1D):
        U = [rho, rho*u, E]^T

    where total energy per unit volume:
        E = p/(gamma - 1) + 0.5 * rho * u^2
    """
    E = p / (gamma - 1.0) + 0.5 * rho * u**2
    return np.array([rho, rho*u, E])


def conserved_to_primitive(U, gamma):
    """
    Convert conserved variables U = [rho, rho*u, E] to primitive (rho, u, p).

    Includes basic sanity checks to avoid division by zero or negative pressure.
    """
    rho = U[0]
    mom = U[1]
    E = U[2]

    # Avoid division by zero in velocity:
    if rho <= 0:
        raise ValueError(f"Non-positive density encountered: rho = {rho}")

    u = mom / rho
    # Internal energy per volume
    e_int = E - 0.5 * rho * u**2
    p = (gamma - 1.0) * e_int

    if p <= 0:
        raise ValueError(f"Non-positive pressure computed: p = {p}")

    return rho, u, p


def flux(U, gamma):
    """
    Compute the physical flux F(U) for the 1D Euler equations.

    U = [rho, rho*u, E]
    F = [rho*u, rho*u^2 + p, u*(E + p)]
    """
    rho, u, p = conserved_to_primitive(U, gamma)
    F1 = rho * u
    F2 = rho * u**2 + p
    F3 = u * (E := (p / (gamma - 1.0) + 0.5 * rho * u**2)) + u * p  # recompute E
    return np.array([F1, F2, F3])


def sound_speed(rho, p, gamma):
    """
    Speed of sound for an ideal gas: a = sqrt(gamma * p / rho)
    """
    return np.sqrt(gamma * p / rho)


def rusanov_flux(UL, UR, gamma):
    """
    Rusanov (local Lax–Friedrichs) numerical flux at an interface.

    Given left state UL and right state UR (conserved variables),
    we compute:

        F_hat = 0.5*(F(UL) + F(UR)) - 0.5*lambda_max*(UR - UL)

    where lambda_max is the maximum wave speed at the interface.

    This is simple and quite diffusive, but very robust for teaching.
    """
    # Physical fluxes
    FL = flux(UL, gamma)
    FR = flux(UR, gamma)

    # Primitive variables on both sides
    rhoL, uL, pL = conserved_to_primitive(UL, gamma)
    rhoR, uR, pR = conserved_to_primitive(UR, gamma)

    # Local max wave speeds (|u| + a)
    aL = sound_speed(rhoL, pL, gamma)
    aR = sound_speed(rhoR, pR, gamma)
    smax = max(abs(uL) + aL, abs(uR) + aR)

    return 0.5 * (FL + FR) - 0.5 * smax * (UR - UL)


def run_euler_1d(
    rhoL, uL, pL,
    rhoR, uR, pR,
    gamma=1.4,
    nx=200,
    x_domain=(0.0, 1.0),
    final_time=0.2,
    cfl=0.5,
):
    """
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
    """

    if nx < 10:
        raise ValueError("nx (number of cells) should be at least 10 for a meaningful demo.")
    if final_time <= 0:
        raise ValueError("final_time must be positive.")
    if gamma <= 1.0:
        raise ValueError("gamma must be > 1 (ideal gas).")
    if cfl <= 0 or cfl >= 1:
        raise ValueError("CFL should be in (0, 1) for stability (e.g., 0.5).")

    x_start, x_end = x_domain
    if x_end <= x_start:
        raise ValueError("x_domain end must be greater than start.")

    # Uniform grid
    dx = (x_end - x_start) / nx
    x_centers = x_start + (np.arange(nx) + 0.5) * dx

    # Allocate conserved variables U[i, :] for each cell i
    U = np.zeros((nx, 3))

    # Initialize: left state on left half, right state on right half
    midpoint = 0.5 * (x_start + x_end)
    for i, x in enumerate(x_centers):
        if x < midpoint:
            U[i, :] = primitive_to_conserved(rhoL, uL, pL, gamma)
        else:
            U[i, :] = primitive_to_conserved(rhoR, uR, pR, gamma)

    t = 0.0
    # Time stepping loop
    while t < final_time:
        # Compute max wave speed for CFL condition
        max_speed = 0.0
        for i in range(nx):
            rho_i, u_i, p_i = conserved_to_primitive(U[i, :], gamma)
            a_i = sound_speed(rho_i, p_i, gamma)
            max_speed = max(max_speed, abs(u_i) + a_i)

        dt = cfl * dx / max_speed
        if t + dt > final_time:
            dt = final_time - t

        # Compute numerical fluxes at interfaces
        F_num = np.zeros((nx + 1, 3))
        for i in range(1, nx):
            # Interface between cell i-1 and i
            UL = U[i - 1, :]
            UR = U[i, :]
            F_num[i, :] = rusanov_flux(UL, UR, gamma)

        # Simple transmissive boundary conditions:
        #   F_0 = F_1, F_{nx} = F_{nx-1}
        F_num[0, :] = F_num[1, :]
        F_num[nx, :] = F_num[nx - 1, :]

        # Update U using finite volume update:
        #   U_i^{n+1} = U_i^n - (dt/dx)*(F_{i+1/2} - F_{i-1/2})
        U_new = np.empty_like(U)
        for i in range(nx):
            U_new[i, :] = U[i, :] - (dt / dx) * (F_num[i + 1, :] - F_num[i, :])

        U = U_new
        t += dt

    # Convert final U back to primitive variables (rho, u, p)
    rho = np.zeros(nx)
    u = np.zeros(nx)
    p = np.zeros(nx)
    for i in range(nx):
        rho[i], u[i], p[i] = conserved_to_primitive(U[i, :], gamma)

    return x_centers, rho, u, p


# =============================================================================
#  GUI SETUP (Tkinter)
# =============================================================================

class EulerGUI:
    """
    Tkinter-based GUI wrapper for the 1D Euler solver.
    """

    def __init__(self, master):
        self.master = master
        master.title("1D Euler Equations (Hypersonics) Demo")

        # Use a simple grid of labels + entries
        # Left state inputs
        tk.Label(master, text="Left State (x < mid)").grid(row=0, column=0, columnspan=2, pady=(5, 0))
        self.rhoL_var = self._add_labeled_entry("rho_L (density)", 1, 0, default="1.0")
        self.uL_var   = self._add_labeled_entry("u_L (velocity, m/s)", 2, 0, default="0.0")
        self.pL_var   = self._add_labeled_entry("p_L (pressure, Pa)", 3, 0, default="100000.0")

        # Right state inputs
        tk.Label(master, text="Right State (x >= mid)").grid(row=0, column=2, columnspan=2, pady=(5, 0))
        self.rhoR_var = self._add_labeled_entry("rho_R (density)", 1, 2, default="0.125")
        self.uR_var   = self._add_labeled_entry("u_R (velocity, m/s)", 2, 2, default="0.0")
        self.pR_var   = self._add_labeled_entry("p_R (pressure, Pa)", 3, 2, default="10000.0")

        # Gas & numerical parameters
        row = 4
        tk.Label(master, text="Gas & Numerical Parameters").grid(row=row, column=0, columnspan=4, pady=(10, 0))

        self.gamma_var      = self._add_labeled_entry("gamma (ratio of specific heats)", row+1, 0, default="1.4")
        self.nx_var         = self._add_labeled_entry("Number of cells (nx)",           row+2, 0, default="200")
        self.final_time_var = self._add_labeled_entry("Final time (seconds)",           row+1, 2, default="0.02")
        self.cfl_var        = self._add_labeled_entry("CFL number",                    row+2, 2, default="0.5")

        # Run button
        tk.Button(master, text="Run Simulation", command=self.run_simulation).grid(
            row=row+3, column=0, columnspan=4, pady=10
        )

        # Small note for students
        tk.Label(
            master,
            text="Note: This is a simple 1D shock-tube solver for the Euler equations.\n"
                 "Large jumps & long times may require more cells or can cause instabilities.",
            fg="gray"
        ).grid(row=row+4, column=0, columnspan=4, pady=(0, 5))

    def _add_labeled_entry(self, label_text, row, col, default=""):
        """
        Helper to add "Label: [Entry]" to the Tk grid.
        Returns a tk.StringVar bound to the entry.
        """
        label = tk.Label(self.master, text=label_text)
        label.grid(row=row, column=col, sticky="e", padx=5, pady=2)

        var = tk.StringVar(value=default)
        entry = tk.Entry(self.master, textvariable=var, width=15)
        entry.grid(row=row, column=col+1, sticky="w", padx=5, pady=2)
        return var

    def _parse_float(self, var, name):
        """
        Safely parse a float from a tk.StringVar.
        Show a messagebox on error.
        """
        try:
            value = float(var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"{name} must be a number. Got: {var.get()}")
            raise
        return value

    def _parse_int(self, var, name):
        """
        Safely parse an int from a tk.StringVar.
        """
        try:
            value = int(var.get())
        except ValueError:
            messagebox.showerror("Invalid Input", f"{name} must be an integer. Got: {var.get()}")
            raise
        return value

    def run_simulation(self):
        """
        Callback for the "Run Simulation" button.
        Parses inputs, runs the solver, and plots results.
        """
        try:
            # Parse all inputs
            rhoL = self._parse_float(self.rhoL_var, "rho_L")
            uL   = self._parse_float(self.uL_var, "u_L")
            pL   = self._parse_float(self.pL_var, "p_L")

            rhoR = self._parse_float(self.rhoR_var, "rho_R")
            uR   = self._parse_float(self.uR_var, "u_R")
            pR   = self._parse_float(self.pR_var, "p_R")

            gamma      = self._parse_float(self.gamma_var, "gamma")
            nx         = self._parse_int(self.nx_var, "Number of cells")
            final_time = self._parse_float(self.final_time_var, "Final time")
            cfl        = self._parse_float(self.cfl_var, "CFL number")

            # Basic physical checks
            for name, val in [("rho_L", rhoL), ("rho_R", rhoR)]:
                if val <= 0:
                    messagebox.showerror("Invalid Input", f"{name} must be > 0.")
                    return
            for name, val in [("p_L", pL), ("p_R", pR)]:
                if val <= 0:
                    messagebox.showerror("Invalid Input", f"{name} must be > 0.")
                    return

            # Run the solver
            x, rho, u, p = run_euler_1d(
                rhoL, uL, pL,
                rhoR, uR, pR,
                gamma=gamma,
                nx=nx,
                x_domain=(0.0, 1.0),
                final_time=final_time,
                cfl=cfl,
            )

            # Plot results in a separate Matplotlib window
            self.plot_results(x, rho, u, p)

        except Exception as e:
            # Catch any unexpected error and show a messagebox
            # Also print traceback to console for debugging.
            traceback.print_exc()
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")

    def plot_results(self, x, rho, u, p):
        """
        Simple Matplotlib plot: rho, u, p vs. x in three subplots.
        """
        fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

        axes[0].plot(x, rho)
        axes[0].set_ylabel("Density ρ")
        axes[0].grid(True)

        axes[1].plot(x, u)
        axes[1].set_ylabel("Velocity u (m/s)")
        axes[1].grid(True)

        axes[2].plot(x, p)
        axes[2].set_ylabel("Pressure p (Pa)")
        axes[2].set_xlabel("x")
        axes[2].grid(True)

        fig.suptitle("1D Euler Shock Tube Solution (Final Time)")
        plt.tight_layout()
        plt.show()


# =============================================================================
#  MAIN
# =============================================================================

def main():
    root = tk.Tk()
    app = EulerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
