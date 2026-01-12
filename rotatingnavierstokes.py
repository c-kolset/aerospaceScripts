"""
Rotating Navier–Stokes (with Coriolis force) — 2D incompressible solver
======================================================================

This script solves the *rotating* incompressible Navier–Stokes equations
on a 2D periodic box using a classic **projection method**:

    ∂u/∂t + (u·∇)u + f k×u = -(1/ρ)∇p + ν∇²u + F
    ∇·u = 0

where:
    u = (u, v) is velocity in the x-y plane
    f = 2Ω is the Coriolis parameter (constant "f-plane")
    k×u = (-v, u) rotates velocity by 90 degrees
    ν is kinematic viscosity
    p is pressure that enforces incompressibility
    F is an optional forcing term (here set to zero by default)

Numerics (student-friendly summary)
-----------------------------------
1) Compute a "tentative" velocity u* ignoring pressure.
2) Solve a Poisson equation for pressure so that the corrected velocity is divergence-free.
3) Correct velocities: u^(n+1) = u* - (dt/ρ) ∇p

We use:
- 2D finite differences (central) for derivatives
- FFT-based Poisson solver (fast and simple for periodic boundaries)
- Matplotlib animation to visualize the flow

Requirements:
    pip install numpy matplotlib

Tips for students:
- Try increasing/decreasing f (rotation) and see inertial oscillations.
- Try changing ν (viscosity) and see smoothing/damping.
- Try changing the initial vortex size/strength.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# -----------------------------
# 1) Physical & numerical parameters
# -----------------------------
Lx, Ly = 2.0 * np.pi, 2.0 * np.pi   # domain size (periodic in both directions)
Nx, Ny = 128, 128                   # grid resolution (try 64 for faster, 256 for sharper)
nu = 1e-3                           # kinematic viscosity ν
rho = 1.0                           # density ρ (set to 1 for convenience)

Omega = 2.0                         # rotation rate Ω
f = 2.0 * Omega                     # Coriolis parameter f = 2Ω (constant f-plane)

t_end = 20.0                        # final simulation time
cfl = 0.5                           # CFL number for choosing dt based on max speed

# Visualization settings
plot_every = 2                      # update plot every N time steps (faster animation)
quiver_stride = 6                   # larger = fewer arrows
show_quiver = True                  # toggle velocity arrows


# -----------------------------
# 2) Grid setup
# -----------------------------
dx = Lx / Nx
dy = Ly / Ny

x = np.linspace(0, Lx, Nx, endpoint=False)
y = np.linspace(0, Ly, Ny, endpoint=False)
X, Y = np.meshgrid(x, y, indexing="xy")


# -----------------------------
# 3) Helper functions: periodic finite differences
# -----------------------------
def ddx(q):
    """∂q/∂x using central differences with periodic boundary conditions."""
    return (np.roll(q, -1, axis=1) - np.roll(q, 1, axis=1)) / (2.0 * dx)

def ddy(q):
    """∂q/∂y using central differences with periodic boundary conditions."""
    return (np.roll(q, -1, axis=0) - np.roll(q, 1, axis=0)) / (2.0 * dy)

def laplacian(q):
    """∇²q using second-order central differences with periodic boundaries."""
    return (
        (np.roll(q, -1, axis=1) - 2.0 * q + np.roll(q, 1, axis=1)) / dx**2
        + (np.roll(q, -1, axis=0) - 2.0 * q + np.roll(q, 1, axis=0)) / dy**2
    )


# -----------------------------
# 4) FFT-based Poisson solver (periodic)
# -----------------------------
# We solve: ∇²p = rhs  on a periodic domain
# In Fourier space: -(kx^2 + ky^2) p_hat = rhs_hat
# so: p_hat = - rhs_hat / (kx^2 + ky^2)
kx = 2.0 * np.pi * np.fft.fftfreq(Nx, d=dx)  # shape (Nx,)
ky = 2.0 * np.pi * np.fft.fftfreq(Ny, d=dy)  # shape (Ny,)
KX, KY = np.meshgrid(kx, ky, indexing="xy")
K2 = KX**2 + KY**2

# Avoid division by zero at the zero wavenumber (mean mode).
# Pressure is determined up to an additive constant anyway, so we set p_hat(0,0)=0.
K2[0, 0] = 1.0


def solve_poisson_periodic(rhs):
    """Return p solving ∇²p = rhs with periodic BCs using FFT."""
    rhs_hat = np.fft.fft2(rhs)
    p_hat = -rhs_hat / K2
    p_hat[0, 0] = 0.0
    p = np.fft.ifft2(p_hat).real
    return p


# -----------------------------
# 5) Initial condition: a vortex (student-tweakable)
# -----------------------------
# We'll start with a smooth "Gaussian vortex" in the velocity field.
# This makes a nice swirling flow that rotation will influence.
x0, y0 = Lx / 2.0, Ly / 2.0
sigma = 0.4          # vortex size (try 0.2, 0.6)
strength = 1.5       # vortex strength (try 0.5, 3.0)

r2 = (X - x0)**2 + (Y - y0)**2
gauss = np.exp(-r2 / (2.0 * sigma**2))

# A simple swirling pattern (like a small eddy):
u = -strength * (Y - y0) * gauss
v =  strength * (X - x0) * gauss

# Optional: subtract mean flow so the domain-average velocity is ~0
u -= u.mean()
v -= v.mean()


# -----------------------------
# 6) Time stepping utilities
# -----------------------------
def compute_dt(u, v):
    """
    Choose dt based on a CFL condition:
        dt <= cfl * min(dx, dy) / max_speed
    plus a viscous stability-ish guard.
    """
    speed = np.sqrt(u**2 + v**2)
    umax = max(1e-12, speed.max())  # avoid division by zero
    dt_adv = cfl * min(dx, dy) / umax

    # A simple diffusion restriction (not strict for all schemes, but helpful):
    dt_diff = 0.25 * min(dx, dy)**2 / max(nu, 1e-12)

    return min(dt_adv, dt_diff)


def step_projection(u, v, dt):
    """
    One time step of rotating incompressible Navier–Stokes using projection.

    1) Compute nonlinear advection: (u·∇)u, (u·∇)v
    2) Add Coriolis:  f k×u = (-f v, +f u)
    3) Add viscosity: ν∇²u, ν∇²v
    4) Tentative velocity u* (no pressure)
    5) Solve Poisson for pressure from divergence of u*
    6) Correct velocity to enforce ∇·u = 0
    """
    # --- 1) Advection terms (u·∇)u and (u·∇)v ---
    ux = ddx(u)
    uy = ddy(u)
    vx = ddx(v)
    vy = ddy(v)

    adv_u = u * ux + v * uy
    adv_v = u * vx + v * vy

    # --- 2) Coriolis term: f k×u = (-f v, +f u) ---
    cor_u = -f * v
    cor_v =  f * u

    # --- 3) Viscous diffusion ---
    diff_u = nu * laplacian(u)
    diff_v = nu * laplacian(v)

    # --- Optional forcing (set to zero by default) ---
    Fx = 0.0
    Fy = 0.0

    # --- 4) Tentative velocity (explicit Euler) ---
    # u* = u + dt * [ -advection - (1/ρ)∂p/∂x + diffusion + coriolis + forcing ]
    # but here we delay pressure to the projection step
    u_star = u + dt * (-adv_u + diff_u + cor_u + Fx)
    v_star = v + dt * (-adv_v + diff_v + cor_v + Fy)

    # --- 5) Pressure Poisson equation from incompressibility ---
    # Enforce: ∇·u^(n+1) = 0
    # with: u^(n+1) = u* - (dt/ρ) ∇p
    # => ∇·u* - (dt/ρ) ∇²p = 0
    # => ∇²p = (ρ/dt) ∇·u*
    div_u_star = ddx(u_star) + ddy(v_star)
    rhs = (rho / dt) * div_u_star
    p = solve_poisson_periodic(rhs)

    # --- 6) Velocity correction ---
    px = ddx(p)
    py = ddy(p)

    u_new = u_star - (dt / rho) * px
    v_new = v_star - (dt / rho) * py

    return u_new, v_new, p


# -----------------------------
# 7) Set up plotting / animation
# -----------------------------
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 5))

# Left panel: speed magnitude
speed = np.sqrt(u**2 + v**2)
im = ax0.imshow(
    speed,
    origin="lower",
    extent=[0, Lx, 0, Ly],
    aspect="auto"
)
ax0.set_title("Speed |u|")
ax0.set_xlabel("x")
ax0.set_ylabel("y")
cb = plt.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
cb.set_label("|u|")

# Optional quiver plot on top of the speed
quiv = None
if show_quiver:
    xs = X[::quiver_stride, ::quiver_stride]
    ys = Y[::quiver_stride, ::quiver_stride]
    quiv = ax0.quiver(
        xs, ys,
        u[::quiver_stride, ::quiver_stride],
        v[::quiver_stride, ::quiver_stride],
        scale=30
    )

# Right panel: kinetic energy vs time
ax1.set_title("Total kinetic energy vs time")
ax1.set_xlabel("time")
ax1.set_ylabel("K = 0.5 ∫|u|^2 dA")
line_ke, = ax1.plot([], [], lw=2)
ax1.grid(True)

times = [0.0]
KEs = [0.5 * np.mean(u**2 + v**2) * (Lx * Ly)]  # mean*area gives integral estimate

# A text box with current parameters
param_text = ax0.text(
    0.02, 0.98,
    f"f = {f:.2f}\nν = {nu:.2e}",
    transform=ax0.transAxes,
    va="top",
    ha="left",
    bbox=dict(facecolor="white", alpha=0.7, edgecolor="none")
)

t = 0.0
step_count = 0


def update(frame):
    """
    Update function called by Matplotlib's animation.
    We advance several simulation steps between frames for smooth animation.
    """
    global u, v, t, step_count

    # Choose dt dynamically each update (helps stability if speeds change)
    dt = compute_dt(u, v)

    # Advance multiple steps per frame (controlled by plot_every)
    for _ in range(plot_every):
        u, v, p = step_projection(u, v, dt)
        t += dt
        step_count += 1

        # Record kinetic energy for the time series plot
        times.append(t)
        KEs.append(0.5 * np.mean(u**2 + v**2) * (Lx * Ly))

        if t >= t_end:
            break

    # Update the speed image
    speed = np.sqrt(u**2 + v**2)
    im.set_data(speed)
    im.set_clim(vmin=speed.min(), vmax=speed.max())

    # Update quiver arrows
    if show_quiver and quiv is not None:
        quiv.set_UVC(
            u[::quiver_stride, ::quiver_stride],
            v[::quiver_stride, ::quiver_stride]
        )

    # Update KE plot
    line_ke.set_data(times, KEs)
    ax1.set_xlim(0, max(times))
    ax1.set_ylim(0, max(KEs) * 1.05)

    # Update parameter/time text
    param_text.set_text(f"t = {t:.2f}\nf = {f:.2f}\nν = {nu:.2e}")

    return (im, line_ke) if quiv is None else (im, quiv, line_ke, param_text)


ani = FuncAnimation(fig, update, interval=30, blit=False)

plt.tight_layout()
plt.show()


"""
Student experiments (easy edits)
--------------------------------
1) Rotation strength:
   - Try Omega = 0.0 (no rotation) vs Omega = 5.0 (strong rotation)

2) Viscosity:
   - Try nu = 1e-4 (less damping) or nu = 5e-3 (more damping)

3) Initial condition:
   - Change sigma and strength
   - Or add a second vortex to see interactions

4) Forcing:
   - Inside step_projection(), set Fx, Fy to something nonzero
     to keep turbulence going (advanced).

Notes / limitations (honest):
-----------------------------
- This is a teaching-quality solver, not a production CFD code.
- Uses explicit Euler time stepping (simple but not super accurate).
  A great next step is to implement RK2/RK4 for the advection+Coriolis part.
- Periodic boundaries simplify the Poisson solve and are common in theory/learning.
"""
