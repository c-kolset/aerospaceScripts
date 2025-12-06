"""
paris_law_menu.py

Educational script to demonstrate Paris' Law for fatigue crack growth:

    da/dN = C * (ΔK)^m

With:
    ΔK = Y(a) * Δσ * sqrt(pi * a)

Where:
    a   = crack length (m)
    N   = number of load cycles
    C,m = material Paris-law constants
    Δσ  = stress range (Pa)
    Y   = geometry factor (dimensionless, can depend on a)

This version adds:
    1. A simple TEXT-BASED MENU for student interaction.
    2. A more realistic geometry factor Y(a) for an EDGE CRACK in a plate.

Two modeling options:
    (1) SIMPLE MODEL (constant Y):
        - Assumes Y is constant (e.g. Y = 1.0).
        - We can derive an ANALYTIC expression for N (cycles).
        - Also perform numerical integration to compare.

    (2) EDGE-CRACK PLATE MODEL (Y(a)):
        - Uses a more realistic Y(a) for an edge crack in a finite-width plate.
        - Must be solved NUMERICALLY (no simple analytic formula for N).
"""

import numpy as np
import matplotlib.pyplot as plt


# ================================================================
# GEOMETRY FACTORS (Y FUNCTIONS)
# ================================================================

def Y_constant(a, Y0=1.0):
    """
    Geometry factor that is simply constant (no dependence on a).

    Parameters
    ----------
    a : float or np.ndarray
        Crack length (m). *Not actually used*, but kept for a uniform
        interface where Y is always a function of a.
    Y0 : float
        Constant geometry factor.

    Returns
    -------
    float or np.ndarray
        Geometry factor Y = Y0.
    """
    return Y0


def Y_edge_crack(a, W):
    """
    Geometry factor Y(a) for an edge crack in a finite-width plate.

    This is a commonly used empirical polynomial approximation:

        Y(a) ≈ 1.12 - 0.23*α + 10.55*α^2 - 21.72*α^3 + 30.39*α^4

    where α = a / W, and:
        a = crack length
        W = plate width

    Assumes a/W is not too large (typically a/W < ~0.6).

    Parameters
    ----------
    a : float or np.ndarray
        Crack length (m).
    W : float
        Plate width (m).

    Returns
    -------
    float or np.ndarray
        Geometry factor Y(a).
    """
    alpha = a / W
    return 1.12 - 0.23 * alpha + 10.55 * alpha**2 - 21.72 * alpha**3 + 30.39 * alpha**4


# ================================================================
# CORE PARIS LAW FUNCTIONS
# ================================================================

def delta_K(a, delta_sigma, Y_function):
    """
    Compute the stress intensity factor range ΔK for a given crack length
    and geometry.

    Parameters
    ----------
    a : float or np.ndarray
        Crack length (m).
    delta_sigma : float
        Stress range Δσ (Pa).
    Y_function : callable
        A function of a that returns Y(a). Examples:
            - lambda a: Y_constant(a, Y0=1.0)
            - lambda a: Y_edge_crack(a, W)

    Returns
    -------
    float or np.ndarray
        Stress intensity factor range ΔK (Pa * sqrt(m)).
    """
    Y_val = Y_function(a)        # Evaluate geometry factor at current a
    return Y_val * delta_sigma * np.sqrt(np.pi * a)


def paris_law_da_dN(a, C, m, delta_sigma, Y_function):
    """
    Compute da/dN from Paris' law:

        da/dN = C * (ΔK)^m

    Parameters
    ----------
    a : float or np.ndarray
        Crack length (m).
    C : float
        Paris law constant.
    m : float
        Paris law exponent.
    delta_sigma : float
        Stress range Δσ (Pa).
    Y_function : callable
        Function of a that returns Y(a).

    Returns
    -------
    float or np.ndarray
        Crack growth rate da/dN (m per cycle).
    """
    dK = delta_K(a, delta_sigma, Y_function)
    return C * (dK ** m)


def paris_analytic_cycles_constant_Y(a_initial, a_final, C, m, delta_sigma, Y0):
    """
    Analytical solution for the number of cycles required
    to grow a crack from a_initial to a_final using Paris' law,
    ASSUMING CONSTANT Y (no a-dependence).

    Starting from:
        da/dN = C * (ΔK)^m
        ΔK = Y0 * Δσ * sqrt(pi * a)

    => da/dN = C * (Y0 * Δσ * sqrt(pi * a))^m
             = C * (Y0 * Δσ * sqrt(pi))^m * a^(m/2)

    Rearranging:
        dN = da / [ C * (Y0 * Δσ * sqrt(pi))^m * a^(m/2) ]

    Integrate from a_initial to a_final:

    If m ≠ 2:
        N = [1 / ( C * (Y0 * Δσ * sqrt(pi))^m )] *
            [ 1 / (1 - m/2) ] *
            ( a_final^(1 - m/2) - a_initial^(1 - m/2) )

    If m = 2:
        N = [1 / ( C * (Y0 * Δσ * sqrt(pi))^2 )] * ln(a_final / a_initial)

    Parameters
    ----------
    a_initial : float
        Initial crack length (m).
    a_final : float
        Final crack length (m).
    C : float
        Paris law constant.
    m : float
        Paris law exponent.
    delta_sigma : float
        Stress range Δσ (Pa).
    Y0 : float
        Constant geometry factor.

    Returns
    -------
    float
        Number of cycles to grow from a_initial to a_final.
    """
    factor = (Y0 * delta_sigma * np.sqrt(np.pi)) ** m

    if np.isclose(m, 2.0):
        # Use ln form if m is exactly (or very close to) 2
        N = (1.0 / (C * factor)) * np.log(a_final / a_initial)
    else:
        exponent = 1.0 - m / 2.0
        N = (1.0 / (C * factor)) * (1.0 / exponent) * \
            (a_final ** exponent - a_initial ** exponent)

    return N


def paris_numerical_growth(a_initial, a_final, C, m, delta_sigma,
                           Y_function, num_steps=1000):
    """
    Numerical integration of Paris' law to obtain crack length vs. cycles.

    Concept (simple explicit Euler in a, not N):
        - We choose many small steps in crack length (a).
        - For each small increment da:
              dN = da / (da/dN)
          where (da/dN) is computed from Paris' law at the previous crack length.
        - Sum up all dN to get N.

    This is educational (simple) rather than highly accurate.

    Parameters
    ----------
    a_initial : float
        Initial crack length (m).
    a_final : float
        Final crack length (m).
    C : float
        Paris law constant.
    m : float
        Paris law exponent.
    delta_sigma : float
        Stress range Δσ (Pa).
    Y_function : callable
        Function of a that returns Y(a).
    num_steps : int
        Number of a-steps between a_initial and a_final.

    Returns
    -------
    N_values : np.ndarray
        Number of cycles at each crack length step.
    a_values : np.ndarray
        Crack length values from a_initial to a_final.
    """
    # Create array of crack lengths
    a_values = np.linspace(a_initial, a_final, num_steps)

    # Array for N values (cycles)
    N_values = np.zeros_like(a_values)

    for i in range(1, num_steps):
        a_prev = a_values[i - 1]
        a_curr = a_values[i]
        da = a_curr - a_prev

        # Compute da/dN at previous a
        da_dN_prev = paris_law_da_dN(a_prev, C, m, delta_sigma, Y_function)

        if da_dN_prev <= 0.0:
            dN = 0.0
        else:
            dN = da / da_dN_prev

        N_values[i] = N_values[i - 1] + dN

    return N_values, a_values


# ================================================================
# HELPER: INPUT WITH DEFAULT
# ================================================================

def input_with_default(prompt, default, cast_type=float):
    """
    Ask the user for input, but use a default value if they just press Enter.

    Parameters
    ----------
    prompt : str
        Text shown to the user.
    default : any
        Default value if user enters nothing.
    cast_type : callable
        Function to convert the string input to a desired type (e.g. float, int).

    Returns
    -------
    value : same type as default (after casting)
    """
    full_prompt = f"{prompt} [default = {default}]: "
    s = input(full_prompt)

    # If user just hits Enter, return the default.
    if s.strip() == "":
        return default

    # Try casting to the desired type.
    try:
        return cast_type(s)
    except ValueError:
        print("Invalid input. Using default value.")
        return default


# ================================================================
# MAIN MENU + DEMOS
# ================================================================

def run_simple_constant_Y_model():
    """
    Run the simple model:
        - Constant geometry factor Y0.
        - Both analytic solution and numerical integration.
    """
    print("\n=== SIMPLE MODEL: CONSTANT Y ===\n")
    print("Units:")
    print("  - Crack lengths in mm (converted to meters internally).")
    print("  - Stress range in MPa (converted to Pa internally).\n")

    # Get user inputs with reasonable defaults
    C = input_with_default("Enter Paris law constant C", 1e-12, float)
    m = input_with_default("Enter Paris law exponent m", 3.0, float)

    delta_sigma_MPa = input_with_default("Enter stress range Δσ (MPa)", 100.0, float)
    delta_sigma = delta_sigma_MPa * 1e6  # convert MPa -> Pa

    Y0 = input_with_default("Enter constant geometry factor Y0", 1.0, float)

    a_initial_mm = input_with_default("Enter initial crack size a_initial (mm)", 1.0, float)
    a_final_mm = input_with_default("Enter final crack size a_final (mm)", 10.0, float)

    a_initial = a_initial_mm / 1000.0  # mm -> m
    a_final = a_final_mm / 1000.0      # mm -> m

    num_steps = int(input_with_default("Number of steps for numerical integration",
                                       1000, int))

    # Analytic solution (only valid because Y is constant)
    N_analytic = paris_analytic_cycles_constant_Y(a_initial, a_final, C, m,
                                                  delta_sigma, Y0)

    print("\nParis Law Analytical Solution (Constant Y)")
    print("-----------------------------------------")
    print(f"Initial crack size a_i = {a_initial:.6e} m")
    print(f"Final crack size   a_f = {a_final:.6e} m")
    print(f"Stress range Δσ         = {delta_sigma_MPa:.2f} MPa")
    print(f"Y0 (constant)           = {Y0:.3f}")
    print(f"Material constants: C = {C:.3e}, m = {m:.2f}")
    print(f"Analytical cycles N     = {N_analytic:.3e} cycles\n")

    # Numerical integration
    Y_func = lambda a: Y_constant(a, Y0=Y0)
    N_values, a_values = paris_numerical_growth(a_initial, a_final, C, m,
                                                delta_sigma, Y_func,
                                                num_steps=num_steps)

    print("Paris Law Numerical Integration (Constant Y)")
    print("--------------------------------------------")
    print(f"Numerical estimate of cycles to reach a_f: "
          f"N = {N_values[-1]:.3e} cycles")
    print("(Compare this with the analytical solution above.)\n")

    # Plot
    plt.figure()
    plt.plot(N_values, a_values * 1000.0)  # convert a back to mm for plotting
    plt.xlabel("Number of Cycles, N")
    plt.ylabel("Crack Length, a (mm)")
    plt.title("Fatigue Crack Growth (Constant Y)\nCrack Length vs. Number of Cycles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def run_edge_crack_model():
    """
    Run the edge-crack model with Y(a) for a finite-width plate.

    Here:
        - Y(a) is a polynomial in a/W.
        - We must integrate numerically (no simple analytic N).
    """
    print("\n=== EDGE-CRACK MODEL: Y(a) DEPENDS ON a/W ===\n")
    print("Geometry: Edge crack in a finite-width plate.")
    print("  Approximate geometry factor:")
    print("    Y(a) ≈ 1.12 - 0.23*(a/W) + 10.55*(a/W)^2")
    print("             - 21.72*(a/W)^3 + 30.39*(a/W)^4\n")
    print("Units:")
    print("  - a_initial, a_final, W in mm (converted to meters internally).")
    print("  - Stress range in MPa (converted to Pa internally).\n")

    C = input_with_default("Enter Paris law constant C", 1e-12, float)
    m = input_with_default("Enter Paris law exponent m", 3.0, float)

    delta_sigma_MPa = input_with_default("Enter stress range Δσ (MPa)", 100.0, float)
    delta_sigma = delta_sigma_MPa * 1e6  # MPa -> Pa

    W_mm = input_with_default("Enter plate width W (mm)", 100.0, float)
    a_initial_mm = input_with_default("Enter initial crack size a_initial (mm)", 1.0, float)
    a_final_mm = input_with_default("Enter final crack size a_final (mm)", 10.0, float)

    W = W_mm / 1000.0          # mm -> m
    a_initial = a_initial_mm / 1000.0
    a_final = a_final_mm / 1000.0

    num_steps = int(input_with_default("Number of steps for numerical integration",
                                       1000, int))

    # Define Y_function(a) using the plate width W
    Y_func = lambda a: Y_edge_crack(a, W=W)

    # Numerical integration
    N_values, a_values = paris_numerical_growth(a_initial, a_final, C, m,
                                                delta_sigma, Y_func,
                                                num_steps=num_steps)

    print("\nParis Law Numerical Integration (Edge Crack)")
    print("--------------------------------------------")
    print(f"Plate width W           = {W_mm:.2f} mm")
    print(f"Initial crack size a_i  = {a_initial_mm:.2f} mm")
    print(f"Final crack size   a_f  = {a_final_mm:.2f} mm")
    print(f"Stress range Δσ         = {delta_sigma_MPa:.2f} MPa")
    print(f"Material constants: C = {C:.3e}, m = {m:.2f}")
    print(f"Numerical cycles N      = {N_values[-1]:.3e} cycles\n")

    # Plot
    plt.figure()
    plt.plot(N_values, a_values * 1000.0)  # convert a to mm
    plt.xlabel("Number of Cycles, N")
    plt.ylabel("Crack Length, a (mm)")
    plt.title("Fatigue Crack Growth (Edge Crack Model)\nCrack Length vs. Number of Cycles")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def main():
    """
    Main function that implements a simple text-based menu for students.
    """
    while True:
        print("\n==============================")
        print(" Paris Law Demonstration Menu")
        print("==============================")
        print("1) Simple model (constant geometry factor Y)")
        print("   - Analytic solution for N")
        print("   - Numerical integration")
        print("")
        print("2) Edge-crack plate model (Y(a) depends on a/W)")
        print("   - More realistic geometry")
        print("   - Numerical integration only")
        print("")
        print("3) Quit")
        choice = input("Enter your choice (1, 2, or 3): ").strip()

        if choice == "1":
            run_simple_constant_Y_model()
        elif choice == "2":
            run_edge_crack_model()
        elif choice == "3":
            print("Exiting. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


# Run the menu only when executed directly
if __name__ == "__main__":
    main()
