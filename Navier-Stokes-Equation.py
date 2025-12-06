"""
navier_stokes_cavity_menu_animation.py

Educational 2D incompressible Navier–Stokes solver
for a lid-driven cavity using finite differences.

Now with:
    - Simple text-based menu for choosing simulation parameters
    - Optional animation of the velocity field evolution

------------------------------------------------------------
PROBLEM SETUP
------------------------------------------------------------
We solve the 2D incompressible Navier–Stokes equations in a
square cavity [0,1] x [0,1] with:

    - No-slip walls everywhere
    - Top lid moving to the right with constant speed U_lid

Equations (constant density rho and kinematic viscosity nu):

    Continuity (incompressible):
        ∂u/∂x + ∂v/∂y = 0

    x-momentum:
        ∂u/∂t + u ∂u/∂x + v ∂u/∂y
            = - (1/rho) ∂p/∂x + nu (∂²u/∂x² + ∂²u/∂y²)

    y-momentum:
        ∂v/∂t + u ∂v/∂x + v ∂v/∂y
            = - (1/rho) ∂p/∂y + nu (∂²v/∂x² + ∂²v/∂y²)

We use:
    - Finite differences on a uniform grid
    - Simple explicit time stepping for advection and diffusion
    - A projection method to enforce incompressibility (∇·u=0)
      by solving a Poisson equation for pressure at each time step.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# ============================================================
# HELPER: SIMPLE INPUT WITH DEFAULT
# ============================================================

def input_with_default(prompt, default, cast_type=float):
    """
    Ask the user for a value, but allow ENTER to keep a default.

    Parameters
    ----------
    prompt : str
        Text shown to user.
    default : any
        Default value if user just presses ENTER.
    cast_type : callable
        Type to cast the input to (float, int, etc.).

    Returns
    -------
    value : same type as 'default'
    """
    full_prompt = f"{prompt} [default = {default}]: "
    s = input(full_prompt)

    if s.strip() == "":
        return default

    try:
        return cast_type(s)
    except ValueError:
        print("Invalid input. Using default value.")
        return default


# ============================================================
# NUMERICAL HELPER FUNCTIONS
# ============================================================

def build_rhs(u, v, dx, dy, dt, rho):
    """
    Build the right-hand side of the pressure Poisson equation.

    Projection step idea:
        ∇² p = (rho / dt) * (∂u*/∂x + ∂v*/∂y)

    where (u*, v*) are tentative velocities that do NOT yet satisfy
    incompressibility. We approximate ∂u*/∂x and ∂v*/∂y using central
    differences at internal points.

    Parameters
    ----------
    u, v : 2D arrays
        Velocity fields (x- and y-components).
    dx, dy : float
        Grid spacings.
    dt : float
        Time step size.
    rho : float
        Density.

    Returns
    -------
    b : 2D array
        RHS of the Poisson equation.
    """
    ny, nx = u.shape
    b = np.zeros_like(u)

    for i in range(1, ny - 1):
        for j in range(1, nx - 1):
            dudx = (u[i, j+1] - u[i, j-1]) / (2.0 * dx)
            dvdy = (v[i+1, j] - v[i-1, j]) / (2.0 * dy)
            b[i, j] = rho / dt * (dudx + dvdy)

    return b


def pressure_poisson(p, b, dx, dy, nit=50):
    """
    Solve the Poisson equation ∇²p = b with Gauss–Seidel iterations.

    Discrete Poisson in 2D:
        (p[i,j+1] + p[i,j-1]) / dx^2
      + (p[i+1,j] + p[i-1,j]) / dy^2
      - 2*(1/dx^2 + 1/dy^2)*p[i,j] = b[i,j]

    We implement a simple iterative scheme and apply Neumann
    boundary conditions (zero normal gradient) for pressure.

    Parameters
    ----------
    p : 2D array
        Initial guess for pressure (updated in-place).
    b : 2D array
        RHS of Poisson equation.
    dx, dy : float
        Grid spacings.
    nit : int
        Number of Gauss–Seidel iterations.

    Returns
    -------
    p : 2D array
        Updated pressure field.
    """
    ny, nx = p.shape

    for _ in range(nit):
        p_old = p.copy()

        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                p[i, j] = (
                    (p_old[i, j+1] + p_old[i, j-1]) * dy**2
                    + (p_old[i+1, j] + p_old[i-1, j]) * dx**2
                    - b[i, j] * dx**2 * dy**2
                ) / (2.0 * (dx**2 + dy**2))

        # Pressure boundary conditions (Neumann: dp/dn = 0)
        p[:, 0] = p[:, 1]       # left
        p[:, -1] = p[:, -2]     # right
        p[0, :] = p[1, :]       # bottom
        p[-1, :] = p[-2, :]     # top

    return p


def apply_velocity_boundary_conditions(u, v, U_lid=1.0):
    """
    Apply no-slip boundary conditions and lid movement.

    Boundaries (indices):
        i = 0..ny-1 (y-direction)
        j = 0..nx-1 (x-direction)

    - Bottom (i=0):       u=0, v=0
    - Top (i=ny-1):       u=U_lid, v=0   (moving lid)
    - Left (j=0):         u=0, v=0
    - Right (j=nx-1):     u=0, v=0
    """
    ny, nx = u.shape

    # Bottom wall
    u[0, :] = 0.0
    v[0, :] = 0.0

    # Top wall (moving lid)
    u[-1, :] = U_lid
    v[-1, :] = 0.0

    # Left wall
    u[:, 0] = 0.0
    v[:, 0] = 0.0

    # Right wall
    u[:, -1] = 0.0
    v[:, -1] = 0.0

    return u, v


# ============================================================
# CORE SOLVER
# ============================================================

def cavity_flow(nx=41, ny=41, Lx=1.0, Ly=1.0,
                rho=1.0, nu=0.1, U_lid=1.0,
                dt=0.001, nt=500, nit=50,
                animate=False, save_every=10):
    """
    Main function to run 2D lid-driven cavity flow.

    Parameters
    ----------
    nx, ny : int
        Number of grid points in x and y.
    Lx, Ly : float
        Domain size in x and y (0..Lx, 0..Ly).
    rho : float
        Density.
    nu : float
        Kinematic viscosity.
    U_lid : float
        Velocity of the top lid.
    dt : float
        Time step size.
    nt : int
        Number of time steps.
    nit : int
        Iterations for pressure Poisson solver per time step.
    animate : bool
        If True, store snapshots and create an animation.
    save_every : int
        Store a snapshot every 'save_every' time steps for animation.

    Returns
    -------
    (X, Y, u, v, snapshots) where:
        X, Y : 2D meshgrids for plotting.
        u, v : final velocity fields.
        snapshots : list of (u_snap, v_snap) or empty list if animate=False.
    """
    dx = Lx / (nx - 1)
    dy = Ly / (ny - 1)

    x = np.linspace(0, Lx, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Initialize fields
    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))

    # Apply initial BCs (lid)
    u, v = apply_velocity_boundary_conditions(u, v, U_lid=U_lid)

    snapshots = []  # for animation: list of (u, v)

    # Time stepping
    for n in range(nt):
        u_old = u.copy()
        v_old = v.copy()

        # 1) Build RHS for pressure Poisson
        b = build_rhs(u_old, v_old, dx, dy, dt, rho)

        # 2) Solve pressure Poisson equation
        p = pressure_poisson(p, b, dx, dy, nit=nit)

        # 3) Update velocities
        for i in range(1, ny - 1):
            for j in range(1, nx - 1):
                # Advection (upwind / semi-upwind style)
                u_adv = (
                    u_old[i, j] * (u_old[i, j] - u_old[i, j-1]) / dx +
                    v_old[i, j] * (u_old[i, j] - u_old[i-1, j]) / dy
                )
                v_adv = (
                    u_old[i, j] * (v_old[i, j] - v_old[i, j-1]) / dx +
                    v_old[i, j] * (v_old[i, j] - v_old[i-1, j]) / dy
                )

                # Diffusion
                u_diff = nu * (
                    (u_old[i, j+1] - 2.0 * u_old[i, j] + u_old[i, j-1]) / dx**2 +
                    (u_old[i+1, j] - 2.0 * u_old[i, j] + u_old[i-1, j]) / dy**2
                )
                v_diff = nu * (
                    (v_old[i, j+1] - 2.0 * v_old[i, j] + v_old[i, j-1]) / dx**2 +
                    (v_old[i+1, j] - 2.0 * v_old[i, j] + v_old[i-1, j]) / dy**2
                )

                # Pressure gradients
                dpdx = (p[i, j+1] - p[i, j-1]) / (2.0 * dx)
                dpdy = (p[i+1, j] - p[i-1, j]) / (2.0 * dy)

                # Update
                u[i, j] = u_old[i, j] + dt * (-u_adv - dpdx / rho + u_diff)
                v[i, j] = v_old[i, j] + dt * (-v_adv - dpdy / rho + v_diff)

        # 4) Reapply boundary conditions
        u, v = apply_velocity_boundary_conditions(u, v, U_lid=U_lid)

        # Save snapshots for animation (if requested)
        if animate and (n % save_every == 0):
            snapshots.append((u.copy(), v.copy()))

        # Optional progress print
        if (n + 1) % max(1, nt // 10) == 0:
            print(f"Time step {n + 1}/{nt}")

    return X, Y, u, v, snapshots


# ============================================================
# POST-PROCESSING: STATIC PLOT
# ============================================================

def plot_final_field(X, Y, u, v):
    """
    Plot the final velocity field:
        - Contours of velocity magnitude
        - Quiver plot of velocity vectors (subsampled)
    """
    vel_mag = np.sqrt(u**2 + v**2)

    plt.figure(figsize=(7, 6))
    cf = plt.contourf(X, Y, vel_mag, 20, cmap="viridis")
    plt.colorbar(cf, label="Velocity magnitude")
    plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2],
               color="white", scale=10)
    plt.title("2D Lid-Driven Cavity Flow\n(Incompressible Navier–Stokes)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


# ============================================================
# POST-PROCESSING: ANIMATION
# ============================================================

def animate_cavity(X, Y, snapshots, Lx=1.0, Ly=1.0, interval=50):
    """
    Create an animation of the velocity magnitude field
    using stored snapshots.

    We use imshow with a color map of |u|.

    Parameters
    ----------
    X, Y : 2D arrays
        Meshgrid (only used for extent).
    snapshots : list of (u_snap, v_snap)
        Saved velocity fields.
    Lx, Ly : float
        Domain size (0..Lx, 0..Ly) for axis extents.
    interval : int
        Delay between frames in milliseconds.
    """
    if len(snapshots) == 0:
        print("No snapshots to animate (animation disabled or save_every too large).")
        return

    # Precompute velocity magnitudes
    vel_frames = []
    for (u_snap, v_snap) in snapshots:
        vel_frames.append(np.sqrt(u_snap**2 + v_snap**2))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.set_title("Velocity Magnitude Evolution")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Initial frame
    im = ax.imshow(
        vel_frames[0],
        origin="lower",
        extent=[0, Lx, 0, Ly],
        cmap="viridis",
        aspect="equal"
    )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("|u|")

    # Animation update function
    def update(frame_index):
        im.set_data(vel_frames[frame_index])
        ax.set_title(f"Velocity Magnitude Evolution\nFrame {frame_index+1}/{len(vel_frames)}")
        return [im]

    ani = FuncAnimation(fig, update, frames=len(vel_frames),
                        interval=interval, blit=True)

    plt.tight_layout()
    plt.show()
    # Note: for saving as GIF or MP4, you'd use ani.save(...),
    # but that requires extra dependencies (ffmpeg, imagemagick).


# ============================================================
# SIMPLE TEXT MENU
# ============================================================

def run_with_menu():
    """
    Provide a simple text menu so students can:
      - Use default params
      - Enter custom params
      - Choose to see animation or just final field
    """
    while True:
        print("\n==============================================")
        print("  2D Navier–Stokes Lid-Driven Cavity Solver   ")
        print("==============================================")
        print("1) Run with DEFAULT parameters (no animation)")
        print("2) Run with CUSTOM parameters (no animation)")
        print("3) Run with CUSTOM parameters + ANIMATION")
        print("4) Quit")
        choice = input("Enter your choice (1, 2, 3, or 4): ").strip()

        if choice == "1":
            # Default parameters
            print("\nRunning with default parameters...")
            X, Y, u, v, snapshots = cavity_flow(
                nx=41, ny=41,
                Lx=1.0, Ly=1.0,
                rho=1.0, nu=0.1, U_lid=1.0,
                dt=0.001, nt=500, nit=50,
                animate=False
            )
            plot_final_field(X, Y, u, v)

        elif choice == "2" or choice == "3":
            # Custom parameters
            print("\nEnter custom parameters (press ENTER for defaults):")

            nx = input_with_default("Number of grid points in x (nx)", 41, int)
            ny = input_with_default("Number of grid points in y (ny)", 41, int)
            nu = input_with_default("Kinematic viscosity nu", 0.1, float)
            U_lid = input_with_default("Lid velocity U_lid", 1.0, float)
            dt = input_with_default("Time step dt", 0.001, float)
            nt = input_with_default("Number of time steps nt", 500, int)
            nit = input_with_default("Pressure iterations per step (nit)", 50, int)

            if choice == "3":
                animate_flag = True
                save_every = input_with_default("Save snapshot every N steps", 10, int)
                print("\nRunning simulation WITH animation...")
            else:
                animate_flag = False
                save_every = 10  # unused
                print("\nRunning simulation WITHOUT animation...")

            X, Y, u, v, snapshots = cavity_flow(
                nx=nx, ny=ny,
                Lx=1.0, Ly=1.0,
                rho=1.0, nu=nu, U_lid=U_lid,
                dt=dt, nt=nt, nit=nit,
                animate=animate_flag, save_every=save_every
            )

            # Always show final field
            plot_final_field(X, Y, u, v)

            # Optionally animate
            if animate_flag:
                animate_cavity(X, Y, snapshots, Lx=1.0, Ly=1.0, interval=80)

        elif choice == "4":
            print("\nExiting Navier–Stokes solver. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please enter 1, 2, 3, or 4.")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Entry point of the script.
    """
    run_with_menu()


if __name__ == "__main__":
    main()
