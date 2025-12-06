"""
hypersonic_wedge_visualizer_gui.py

Educational script to visualize hypersonic airflow over a 2D wedge.

FEATURES
--------
1) Single-case visualization (menu option 1):
   - User inputs:
       * Freestream Mach number M_inf
       * Wedge half-angle theta (deg)
       * Freestream static temperature T_inf (K)
   - Script:
       * Solves oblique shock θ–β–M relation for shock angle β
       * Computes oblique shock property ratios
       * Produces:
           (a) Geometry: wedge + attached oblique shock + flow arrows
           (b) Qualitative temperature map above the wedge
           (c) θ–β curve for that Mach, marking the operating point

2) Mach sweep visualization (menu option 2):
   - User inputs a list of Mach numbers.
   - Script plots β vs θ curves for each Mach on one figure.

3) Slider-based GUI (menu option 4):
   - Sliders for:
       * M_inf
       * theta
       * T_inf
   - Live-updates three panels:
       (a) Geometry + shock + annotation
       (b) Temperature map
       (c) θ–β curve for that Mach with highlighted point

ASSUMPTIONS
-----------
- Perfect gas, gamma = 1.4
- 2D, inviscid, adiabatic flow
- Sharp wedge with attached oblique shock when possible
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# ============================================================
# PHYSICAL CONSTANTS
# ============================================================

GAMMA = 1.4  # Ratio of specific heats for air (assumed constant)


# ============================================================
# CORE OBLIQUE SHOCK RELATIONS
# ============================================================

def theta_beta_m_relation(beta, M, theta, gamma=GAMMA):
    """
    Oblique shock θ–β–M relation written as f(beta) = 0.

    Classical relation:
        tan(theta) = 2 * cot(beta) * (M^2 * sin^2(beta) - 1)
                     / [ M^2 (gamma + cos(2beta)) + 2 ]

    We define:
        f(beta) = tan(theta) - RHS

    So, f(beta) = 0 should give the physical shock angle.

    Parameters
    ----------
    beta : float
        Shock angle in radians.
    M : float
        Upstream Mach number (dimensionless).
    theta : float
        Flow deflection angle (radians) = wedge half-angle (for a wedge).
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    float
        f(beta) value (we want f(beta) = 0).
    """
    # Avoid singular or invalid beta values.
    if beta <= 0.0 or beta >= math.pi / 2:
        return 1e9

    tan_theta = math.tan(theta)
    sin_beta = math.sin(beta)
    cos_beta = math.cos(beta)
    cos_2beta = math.cos(2.0 * beta)

    # cot(beta) = cos(beta) / sin(beta)
    cot_beta = cos_beta / sin_beta

    M2 = M * M
    sin2_beta = sin_beta * sin_beta

    numerator = 2.0 * cot_beta * (M2 * sin2_beta - 1.0)
    denominator = M2 * (gamma + cos_2beta) + 2.0

    rhs = numerator / denominator

    return tan_theta - rhs


def solve_shock_angle(M, theta_deg, gamma=GAMMA):
    """
    Solve for the *weak* oblique shock angle β (deg) for a given M and θ (deg).

    We use a simple bisection method between:
      beta_low  = theta
      beta_high = ~90° (π/2 rad)

    If there is no sign change in f(beta), we assume there is
    no attached oblique shock solution (shock becomes detached)
    for that combination of M and θ.

    Parameters
    ----------
    M : float
        Freestream Mach number.
    theta_deg : float
        Flow deflection / wedge half-angle (degrees).
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    beta_deg : float or None
        Weak solution shock angle in degrees, or None if no solution.
    """
    theta = math.radians(theta_deg)

    beta_low = theta
    beta_high = math.pi / 2

    f_low = theta_beta_m_relation(beta_low, M, theta, gamma)
    f_high = theta_beta_m_relation(beta_high * 0.999, M, theta, gamma)

    # If no sign change, bisection won't work -> probably no attached shock
    if f_low * f_high > 0:
        return None

    for _ in range(100):
        beta_mid = 0.5 * (beta_low + beta_high)
        f_mid = theta_beta_m_relation(beta_mid, M, theta, gamma)

        if abs(f_mid) < 1e-7:
            break

        if f_low * f_mid < 0:
            beta_high = beta_mid
            f_high = f_mid
        else:
            beta_low = beta_mid
            f_low = f_mid

    beta_deg = math.degrees(beta_mid)
    return beta_deg


def oblique_shock_relations(M1, beta_deg, theta_deg, gamma=GAMMA):
    """
    Compute basic oblique-shock relations for a given M1, beta, and theta.

    We treat the oblique shock as a normal shock in the normal direction.

    Steps:
      - M_n1 = M1 * sin(beta)
      - Use normal shock relations to get p2/p1, rho2/rho1, T2/T1, M_n2
      - Convert M_n2 back to M2 using M2 = M_n2 / sin(beta - theta)

    Parameters
    ----------
    M1 : float
        Upstream Mach number.
    beta_deg : float
        Shock angle (degrees).
    theta_deg : float
        Flow deflection angle = wedge half-angle (degrees).
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    dict
        Keys: "M_n1", "M_n2", "M2", "p2_p1", "rho2_rho1", "T2_T1"
    """
    beta = math.radians(beta_deg)
    theta = math.radians(theta_deg)

    M_n1 = M1 * math.sin(beta)
    M_n1_sq = M_n1 * M_n1

    # Normal shock pressure ratio
    p2_p1 = 1.0 + 2.0 * gamma / (gamma + 1.0) * (M_n1_sq - 1.0)

    # Density ratio
    rho2_rho1 = ((gamma + 1.0) * M_n1_sq) / (2.0 + (gamma - 1.0) * M_n1_sq)

    # Temperature ratio
    T2_T1 = p2_p1 / rho2_rho1

    # Downstream normal Mach
    numerator = 1.0 + 0.5 * (gamma - 1.0) * M_n1_sq
    denominator = gamma * M_n1_sq - 0.5 * (gamma - 1.0)
    M_n2_sq = numerator / denominator
    M_n2 = math.sqrt(M_n2_sq)

    # Convert back to flow direction
    M2 = M_n2 / math.sin(beta - theta)

    return {
        "M_n1": M_n1,
        "M_n2": M_n2,
        "M2": M2,
        "p2_p1": p2_p1,
        "rho2_rho1": rho2_rho1,
        "T2_T1": T2_T1
    }


def stagnation_temperature(T, M, gamma=GAMMA):
    """
    Stagnation temperature relation:

        T0 = T * [1 + (gamma - 1)/2 * M^2]

    Parameters
    ----------
    T : float
        Static temperature (K).
    M : float
        Mach number.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    float
        Stagnation temperature T0 (K).
    """
    return T * (1.0 + 0.5 * (gamma - 1.0) * M * M)


# ============================================================
# INPUT HELPER
# ============================================================

def input_with_default(prompt, default, cast_type=float):
    """
    Prompt the user for input with a default value.

    If the user just presses ENTER, the default is used.

    Parameters
    ----------
    prompt : str
        Prompt string.
    default : any
        Default value if user presses ENTER.
    cast_type : callable
        Function used to convert the string to desired type (e.g. float, int).

    Returns
    -------
    value : same type as default (after casting).
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
# GEOMETRY + SHOCK PLOT (STATIC VERSION, FOR MENU OPTION 1)
# ============================================================

def plot_wedge_and_shock(M_inf, theta_deg, beta_deg, shock_info, T_inf):
    """
    2D geometric plot showing:
      - Wedge surface
      - Attached oblique shock
      - Incoming flow arrows
      - Text box with key numbers (M_inf, θ, β, p2/p1, T2/T1, etc.)

    Used in the single-case menu option.
    """
    theta = math.radians(theta_deg)
    beta = math.radians(beta_deg)

    x_max = 1.0
    wedge_x = [0.0, x_max]
    wedge_y = [0.0, x_max * math.tan(theta)]

    shock_x = [0.0, x_max]
    shock_y = [0.0, x_max * math.tan(beta)]

    plt.figure(figsize=(8, 4))
    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    plt.plot(wedge_x, wedge_y, linewidth=3, label="Wedge surface")
    plt.axhline(0.0, linestyle="--", linewidth=1, label="Flow direction")
    plt.plot(shock_x, shock_y, linestyle="--", linewidth=2, label="Oblique shock")

    for y_arrow in [0.0, 0.1, -0.1]:
        plt.arrow(-0.3, y_arrow, 0.25, 0.0,
                  head_width=0.02, head_length=0.05,
                  length_includes_head=True)

    plt.text(0.6, wedge_y[1] + 0.03, f"Wedge, θ = {theta_deg:.1f}°", fontsize=10)
    plt.text(0.6, shock_y[1] + 0.03, f"Shock, β ≈ {beta_deg:.1f}°", fontsize=10)

    T0_inf = stagnation_temperature(T_inf, M_inf)
    T2_T1 = shock_info["T2_T1"]
    T2 = T2_T1 * T_inf
    M2 = shock_info["M2"]
    T0_2 = stagnation_temperature(T2, M2)

    text_x = -0.95
    text_y = 0.45
    props = dict(boxstyle="round", facecolor="white", alpha=0.9)

    info_text = (
        f"M_inf = {M_inf:.2f}\n"
        f"θ = {theta_deg:.1f}°   β ≈ {beta_deg:.1f}°\n"
        f"p2/p1 ≈ {shock_info['p2_p1']:.2f}\n"
        f"T2/T1 ≈ {shock_info['T2_T1']:.2f}\n"
        f"M2 ≈ {shock_info['M2']:.2f}\n"
        f"T_inf = {T_inf:.1f} K   T0_inf ≈ {T0_inf:.1f} K\n"
        f"T2 ≈ {T2:.1f} K   T0_2 ≈ {T0_2:.1f} K"
    )

    plt.text(text_x, text_y, info_text, fontsize=9, bbox=props)

    plt.xlabel("x (arbitrary units)")
    plt.ylabel("y (arbitrary units)")
    plt.title("Hypersonic Flow over a 2D Wedge\n(Geometry + Shock)")
    plt.xlim(-1.0, 1.2)
    plt.ylim(-0.5, 0.8)
    plt.grid(True, linestyle=":")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    return T2


# ============================================================
# TEMPERATURE MAP (STATIC VERSION, FOR MENU OPTION 1)
# ============================================================

def build_temperature_array(theta_deg, T_inf, T2, nx=150, ny=80):
    """
    Build a simple 2D temperature array (qualitative) for plotting.

    We assume:
      - At the wedge surface, T ~ T2 (post-shock, hot).
      - Far from the surface, T -> T_inf.
      - We linearly blend between T2 and T_inf over a vertical distance.

    This returns:
      - T_array: 2D numpy array of temperatures
      - x_min, x_max, y_min, y_max: for use in imshow extent.
    """
    theta = math.radians(theta_deg)

    x_min, x_max = 0.0, 1.0
    y_min, y_max = 0.0, 0.3

    xs = np.linspace(x_min, x_max, nx)
    ys = np.linspace(y_min, y_max, ny)

    T_array = np.zeros((ny, nx))

    d_max = y_max - y_min

    for j, y in enumerate(ys):
        for i, x in enumerate(xs):
            y_wedge = x * math.tan(theta)
            if y >= y_wedge:
                dy = y - y_wedge
                frac = min(dy / d_max, 1.0)
                T_here = T2 - (T2 - T_inf) * frac
                T_array[j, i] = T_here
            else:
                T_array[j, i] = T_inf

    return T_array, x_min, x_max, y_min, y_max


def plot_temperature_map(theta_deg, T_inf, T2):
    """
    Static temperature map plot for menu option 1.
    """
    T_array, x_min, x_max, y_min, y_max = build_temperature_array(
        theta_deg, T_inf, T2
    )

    plt.figure(figsize=(6, 4))
    plt.imshow(
        T_array,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto"
    )
    plt.colorbar(label="Temperature (K)")

    wedge_x = [0.0, x_max]
    wedge_y = [0.0, x_max * math.tan(math.radians(theta_deg))]
    plt.plot(wedge_x, wedge_y, linewidth=2)

    plt.xlabel("x (arbitrary units)")
    plt.ylabel("y (arbitrary units)")
    plt.title("Qualitative Temperature Map above Wedge Surface")
    plt.tight_layout()
    plt.show()


# ============================================================
# θ–β CURVES
# ============================================================

def compute_theta_beta_curve_for_M(M, theta_max_deg=40.0, dtheta_deg=0.5):
    """
    Compute arrays theta_list, beta_list for a given Mach M by
    scanning wedge angles from small values up to theta_max_deg.

    For each theta, we solve for beta (weak shock). If there is
    no attached solution (solve_shock_angle returns None),
    we stop the scan.
    """
    theta_list = []
    beta_list = []

    theta_deg = dtheta_deg
    while theta_deg <= theta_max_deg:
        beta_deg = solve_shock_angle(M, theta_deg, gamma=GAMMA)
        if beta_deg is None:
            break
        theta_list.append(theta_deg)
        beta_list.append(beta_deg)
        theta_deg += dtheta_deg

    return theta_list, beta_list


def plot_theta_beta_for_M(M, theta_current_deg=None, beta_current_deg=None):
    """
    Static θ–β curve for a single Mach number (used in menu option 1).
    """
    theta_list, beta_list = compute_theta_beta_curve_for_M(M)

    if len(theta_list) == 0:
        print(f"\nNo attached-shock θ–β curve could be computed for M = {M:.2f}.")
        return

    plt.figure(figsize=(6, 4))
    plt.plot(theta_list, beta_list, label=f"M = {M:.2f}")

    if theta_current_deg is not None and beta_current_deg is not None:
        plt.scatter([theta_current_deg], [beta_current_deg],
                    zorder=5,
                    label="Current operating point")
        plt.text(theta_current_deg + 0.5, beta_current_deg,
                 f"({theta_current_deg:.1f}°, {beta_current_deg:.1f}°)",
                 fontsize=9)

    plt.xlabel("Wedge angle θ (deg)")
    plt.ylabel("Shock angle β (deg)")
    plt.title(f"Shock Angle vs Wedge Angle for M = {M:.2f}")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_theta_beta_mach_sweep(M_list, theta_max_deg=40.0, dtheta_deg=0.5):
    """
    Plot θ–β curves for a list of Mach numbers on the same figure.
    """
    plt.figure(figsize=(7, 5))

    for M in M_list:
        theta_list, beta_list = compute_theta_beta_curve_for_M(
            M, theta_max_deg=theta_max_deg, dtheta_deg=dtheta_deg
        )

        if len(theta_list) == 0:
            print(f"  (No attached-shock θ–β curve for M = {M:.2f})")
            continue

        plt.plot(theta_list, beta_list, label=f"M = {M:.2f}")

    plt.xlabel("Wedge angle θ (deg)")
    plt.ylabel("Shock angle β (deg)")
    plt.title("Shock Angle β vs Wedge Angle θ\nfor Multiple Mach Numbers")
    plt.grid(True, linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# MENU OPTION 1: SINGLE CASE (STATIC PLOTS)
# ============================================================

def run_single_case():
    """
    Menu option 1:
      - Ask user for M_inf, θ, T_inf.
      - Solve for β and shock properties.
      - Plot:
          (1) Wedge + shock geometry + text
          (2) Temperature map above wedge
          (3) θ–β curve for that Mach with operating point marked
    """
    print("\n=== Hypersonic Wedge Flow Case ===\n")
    print("Assumptions:")
    print("  - Perfect gas, gamma = 1.4")
    print("  - 2D, inviscid, adiabatic flow")
    print("  - Sharp 2D wedge with attached oblique shock (when possible)\n")

    M_inf = input_with_default("Enter freestream Mach number M_inf", 7.0, float)
    theta_deg = input_with_default("Enter wedge half-angle θ (deg)", 10.0, float)
    T_inf = input_with_default("Enter freestream static temperature T_inf (K)", 220.0, float)

    if M_inf <= 1.0:
        print("\nWARNING: M_inf ≤ 1 is not really supersonic; shocks may not form as expected.\n")

    if theta_deg <= 0.0:
        print("\nERROR: Wedge angle must be > 0.\n")
        return

    beta_deg = solve_shock_angle(M_inf, theta_deg, gamma=GAMMA)
    if beta_deg is None:
        print("\nNo attached oblique shock solution found (shock likely detached).")
        print("Try a smaller wedge angle or higher Mach.\n")
        return

    shock_info = oblique_shock_relations(M_inf, beta_deg, theta_deg, gamma=GAMMA)

    print("\n--- Flow / Shock Info ---")
    print(f"M_inf              = {M_inf:.3f}")
    print(f"θ (wedge)          = {theta_deg:.3f} deg")
    print(f"β (shock)          ≈ {beta_deg:.3f} deg")
    print(f"p2/p1              ≈ {shock_info['p2_p1']:.3f}")
    print(f"T2/T1              ≈ {shock_info['T2_T1']:.3f}")
    print(f"M2 (behind shock)  ≈ {shock_info['M2']:.3f}")
    print("--------------------------\n")

    T2 = plot_wedge_and_shock(M_inf, theta_deg, beta_deg, shock_info, T_inf)
    plot_temperature_map(theta_deg, T_inf, T2)
    plot_theta_beta_for_M(M_inf, theta_current_deg=theta_deg, beta_current_deg=beta_deg)


# ============================================================
# MENU OPTION 2: MACH SWEEP
# ============================================================

def run_mach_sweep():
    """
    Menu option 2:
      - Ask user for a comma-separated list of Mach numbers.
      - Plot θ–β curves for each Mach on same figure.
    """
    print("\n=== Mach Number Sweep: θ–β Curves ===\n")
    default_M_list_str = "3,5,7,10"
    M_list_str = input_with_default(
        "Enter Mach numbers as comma-separated list (e.g. 3,5,7,10)",
        default_M_list_str,
        str
    )

    M_list = []
    for token in M_list_str.split(","):
        token = token.strip()
        if token == "":
            continue
        try:
            M_val = float(token)
            if M_val > 1.0:
                M_list.append(M_val)
            else:
                print(f"  Skipping M = {M_val} (must be > 1)")
        except ValueError:
            print(f"  Could not parse '{token}' as a Mach number; skipping.")

    if len(M_list) == 0:
        print("\nNo valid Mach numbers. Aborting Mach sweep.\n")
        return

    theta_max_deg = input_with_default("Max wedge angle θ_max to plot (deg)", 40.0, float)
    dtheta_deg = input_with_default("Angle step Δθ for curves (deg)", 0.5, float)

    plot_theta_beta_mach_sweep(M_list, theta_max_deg=theta_max_deg, dtheta_deg=dtheta_deg)


# ============================================================
# MENU OPTION 4: SLIDER-BASED GUI
# ============================================================

def launch_slider_gui():
    """
    Launch a slider-based GUI using matplotlib widgets.

    Sliders:
      - M_inf (freestream Mach number)
      - θ (wedge angle in degrees)
      - T_inf (freestream static temperature in K)

    Three panels:
      (1) Geometry + shock + annotation
      (2) Temperature map (qualitative)
      (3) θ–β curve for current Mach with current point marked
    """
    # Initial values for sliders
    M0 = 7.0
    theta0 = 10.0
    Tinf0 = 220.0

    # Create figure with a 2x2 layout:
    #   [0,0]: geometry
    #   [0,1]: temperature map
    #   [1,0]: θ–β curve
    # Bottom row of the figure reserved for sliders.
    fig = plt.figure(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.25, wspace=0.4, hspace=0.4)

    ax_geom = fig.add_subplot(2, 2, 1)
    ax_temp = fig.add_subplot(2, 2, 2)
    ax_theta_beta = fig.add_subplot(2, 2, 3)

    # -------------------------
    # 1) Geometry axis setup
    # -------------------------
    ax_geom.set_aspect("equal", adjustable="box")
    ax_geom.set_xlim(-1.0, 1.2)
    ax_geom.set_ylim(-0.5, 0.8)
    ax_geom.set_xlabel("x (arb.)")
    ax_geom.set_ylabel("y (arb.)")
    ax_geom.set_title("Wedge + Oblique Shock")

    # Static flow direction line and arrows
    ax_geom.axhline(0.0, linestyle="--", linewidth=1)

    for y_arrow in [0.0, 0.1, -0.1]:
        ax_geom.arrow(-0.3, y_arrow, 0.25, 0.0,
                      head_width=0.02, head_length=0.05,
                      length_includes_head=True)

    # Placeholders for wedge and shock lines
    wedge_line, = ax_geom.plot([], [], linewidth=3, label="Wedge")
    shock_line, = ax_geom.plot([], [], linestyle="--", linewidth=2, label="Shock")

    # Text box for flow properties
    props = dict(boxstyle="round", facecolor="white", alpha=0.9)
    info_text_obj = ax_geom.text(-0.95, 0.45, "", fontsize=9, bbox=props)

    ax_geom.legend(loc="lower left")

    # -------------------------
    # 2) Temperature map axis
    # -------------------------
    ax_temp.set_xlabel("x (arb.)")
    ax_temp.set_ylabel("y (arb.)")
    ax_temp.set_title("Temperature Map")

    # Create an initial temperature field
    T_array_init, x_min, x_max, y_min, y_max = build_temperature_array(
        theta0, Tinf0, 2 * Tinf0  # rough initial guess
    )
    im_temp = ax_temp.imshow(
        T_array_init,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        aspect="auto"
    )
    cbar = fig.colorbar(im_temp, ax=ax_temp)
    cbar.set_label("Temperature (K)")

    # Wedge line overlay on temp plot
    wedge_temp_line, = ax_temp.plot([], [], linewidth=2)

    # -------------------------
    # 3) θ–β curve axis
    # -------------------------
    ax_theta_beta.set_xlabel("Wedge angle θ (deg)")
    ax_theta_beta.set_ylabel("Shock angle β (deg)")
    ax_theta_beta.set_title("Shock Angle vs Wedge Angle")
    ax_theta_beta.grid(True, linestyle=":")

    theta_curve_line, = ax_theta_beta.plot([], [], label="θ–β curve")
    current_point_scatter = ax_theta_beta.scatter([], [], zorder=5, label="Current point")
    ax_theta_beta.legend()

    # -------------------------
    # Slider axes (below plots)
    # -------------------------
    axcolor = "lightgoldenrodyellow"
    ax_M = plt.axes([0.15, 0.10, 0.70, 0.03], facecolor=axcolor)
    ax_theta = plt.axes([0.15, 0.06, 0.70, 0.03], facecolor=axcolor)
    ax_Tinf = plt.axes([0.15, 0.02, 0.70, 0.03], facecolor=axcolor)

    s_M = Slider(ax_M, "M_inf", 2.0, 15.0, valinit=M0, valstep=0.1)
    s_theta = Slider(ax_theta, "θ (deg)", 2.0, 30.0, valinit=theta0, valstep=0.5)
    s_Tinf = Slider(ax_Tinf, "T_inf (K)", 150.0, 500.0, valinit=Tinf0, valstep=5.0)

    # -------------------------
    # Update function for sliders
    # -------------------------

    def update(val):
        """
        Called whenever any slider is moved.

        Steps:
          1) Read slider values.
          2) Solve for shock angle β (if possible).
          3) Update geometry plot.
          4) Update temperature map.
          5) Update θ–β curve.
        """
        M = s_M.val
        theta_deg = s_theta.val
        T_inf = s_Tinf.val

        # Solve for shock angle
        beta_deg = solve_shock_angle(M, theta_deg, gamma=GAMMA)

        if beta_deg is None:
            # No attached shock: clear shock line and indicate in text
            wedge_x = np.array([0.0, 1.0])
            wedge_y = wedge_x * math.tan(math.radians(theta_deg))
            wedge_line.set_data(wedge_x, wedge_y)
            shock_line.set_data([], [])

            info_text_obj.set_text(
                f"M_inf = {M:.2f}\n"
                f"θ = {theta_deg:.1f}°\n"
                f"No attached shock solution.\n"
                f"Try smaller θ or larger M."
            )

            # Clear θ–β curve and temperature map
            theta_curve_line.set_data([], [])
            current_point_scatter.set_offsets(np.array([[np.nan, np.nan]]))
            im_temp.set_data(np.zeros_like(T_array_init))
            im_temp.set_clim(T_inf, T_inf)

            wedge_temp_x = np.array([0.0, 1.0])
            wedge_temp_y = wedge_temp_x * math.tan(math.radians(theta_deg))
            wedge_temp_line.set_data(wedge_temp_x, wedge_temp_y)

            fig.canvas.draw_idle()
            return

        # If we have an attached shock, compute shock info
        shock_info = oblique_shock_relations(M, beta_deg, theta_deg, gamma=GAMMA)

        # --- Update geometry plot ---
        theta_rad = math.radians(theta_deg)
        beta_rad = math.radians(beta_deg)
        x_max_geom = 1.0
        wedge_x = np.array([0.0, x_max_geom])
        wedge_y = wedge_x * math.tan(theta_rad)
        shock_x = np.array([0.0, x_max_geom])
        shock_y = shock_x * math.tan(beta_rad)

        wedge_line.set_data(wedge_x, wedge_y)
        shock_line.set_data(shock_x, shock_y)

        # Update info text
        T0_inf = stagnation_temperature(T_inf, M)
        T2_T1 = shock_info["T2_T1"]
        T2 = T2_T1 * T_inf
        M2 = shock_info["M2"]
        T0_2 = stagnation_temperature(T2, M2)

        info_text = (
            f"M_inf = {M:.2f}\n"
            f"θ = {theta_deg:.1f}°   β ≈ {beta_deg:.1f}°\n"
            f"p2/p1 ≈ {shock_info['p2_p1']:.2f}\n"
            f"T2/T1 ≈ {shock_info['T2_T1']:.2f}\n"
            f"M2 ≈ {M2:.2f}\n"
            f"T_inf = {T_inf:.1f} K   T0_inf ≈ {T0_inf:.1f} K\n"
            f"T2 ≈ {T2:.1f} K   T0_2 ≈ {T0_2:.1f} K"
        )
        info_text_obj.set_text(info_text)

        # --- Update temperature map ---
        T_array, x_min2, x_max2, y_min2, y_max2 = build_temperature_array(
            theta_deg, T_inf, T2
        )
        im_temp.set_data(T_array)
        im_temp.set_extent([x_min2, x_max2, y_min2, y_max2])
        im_temp.set_clim(T_inf, T2)

        wedge_temp_x = np.array([0.0, x_max2])
        wedge_temp_y = wedge_temp_x * math.tan(theta_rad)
        wedge_temp_line.set_data(wedge_temp_x, wedge_temp_y)

        # --- Update θ–β curve ---
        theta_list, beta_list = compute_theta_beta_curve_for_M(M)
        theta_curve_line.set_data(theta_list, beta_list)
        ax_theta_beta.relim()
        ax_theta_beta.autoscale_view()

        # Mark the current operating point
        current_point_scatter.set_offsets(np.array([[theta_deg, beta_deg]]))

        fig.canvas.draw_idle()

    # Attach the update function to each slider
    s_M.on_changed(update)
    s_theta.on_changed(update)
    s_Tinf.on_changed(update)

    # Call update once at start to initialize everything
    update(None)

    plt.show()


# ============================================================
# MAIN MENU
# ============================================================

def main():
    """
    MAIN FUNCTION

    Menu:
      1) Single-case hypersonic wedge visualization
         -> geometry, shock, temperature map, θ–β curve
      2) Mach sweep
         -> multiple θ–β curves on one plot
      3) Quit
      4) Launch slider-based GUI
    """
    while True:
        print("===================================================")
        print("       Hypersonic Wedge Airflow Visualizer         ")
        print("===================================================")
        print("1) Single-case visualization (wedge, shock, T-map, θ–β)")
        print("2) Mach sweep (multiple θ–β curves)")
        print("3) Quit")
        print("4) Launch slider-based GUI (interactive)")
        choice = input("Enter your choice (1, 2, 3, or 4): ").strip()

        if choice == "1":
            run_single_case()
        elif choice == "2":
            run_mach_sweep()
        elif choice == "3":
            print("\nExiting hypersonic airflow visualizer. Goodbye!\n")
            break
        elif choice == "4":
            launch_slider_gui()
        else:
            print("\nInvalid choice. Please enter 1, 2, 3, or 4.\n")


if __name__ == "__main__":
    main()
