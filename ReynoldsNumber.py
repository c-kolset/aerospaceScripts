"""
reynolds_number.py

A simple, student-friendly script to calculate the Reynolds number.

Reynolds number (Re) is a dimensionless quantity used in fluid mechanics to
predict flow patterns in different fluid flow situations.

There are two common forms of the Reynolds number equation:

1) Using dynamic viscosity (μ):
   Re = (ρ * V * L) / μ

   where:
       ρ (rho) = fluid density          [kg/m^3]
       V       = characteristic velocity [m/s]
       L       = characteristic length   [m]  (e.g., pipe diameter)
       μ (mu)  = dynamic viscosity       [Pa·s or N·s/m^2]

2) Using kinematic viscosity (ν):
   Re = (V * L) / ν

   where:
       V       = characteristic velocity [m/s]
       L       = characteristic length   [m]
       ν (nu)  = kinematic viscosity     [m^2/s]

This script:
- Asks the user which form of the equation they want to use.
- Prompts for the appropriate inputs.
- Computes the Reynolds number.
- Interprets the result as laminar, transitional, or turbulent.
"""

# We will use the 'sys' module only to exit cleanly if needed
import sys


def classify_flow(re):
    """
    Classify the type of flow based on the Reynolds number.

    Common engineering guideline for flow in a pipe:
        Re < 2300        -> Laminar flow
        2300 <= Re < 4000 -> Transitional flow
        Re >= 4000        -> Turbulent flow

    Parameters
    ----------
    re : float
        Reynolds number value.

    Returns
    -------
    str
        Text description of the flow regime.
    """
    if re < 2300:
        return "Laminar flow"
    elif re < 4000:
        return "Transitional flow"
    else:
        return "Turbulent flow"


def get_positive_float(prompt):
    """
    Safely get a positive floating-point number from the user.

    This function:
    - Shows a prompt to the user.
    - Tries to convert the input to a float.
    - Checks that the value is positive.
    - Keeps asking until the user enters a valid positive number.

    Parameters
    ----------
    prompt : str
        The text shown to the user when asking for input.

    Returns
    -------
    float
        A valid positive floating-point number.
    """
    while True:
        user_input = input(prompt)

        try:
            value = float(user_input)
        except ValueError:
            print("Input must be a number. Please try again.\n")
            continue

        if value <= 0:
            print("Value must be positive. Please try again.\n")
            continue

        return value


def reynolds_dynamic_viscosity(density, velocity, length, dynamic_viscosity):
    """
    Compute Reynolds number using dynamic viscosity (μ).

    Uses the formula:
        Re = (ρ * V * L) / μ

    Parameters
    ----------
    density : float
        Fluid density, ρ [kg/m^3].
    velocity : float
        Characteristic velocity, V [m/s].
    length : float
        Characteristic length (e.g. pipe diameter), L [m].
    dynamic_viscosity : float
        Dynamic viscosity, μ [Pa·s].

    Returns
    -------
    float
        Reynolds number.
    """
    re = (density * velocity * length) / dynamic_viscosity
    return re


def reynolds_kinematic_viscosity(velocity, length, kinematic_viscosity):
    """
    Compute Reynolds number using kinematic viscosity (ν).

    Uses the formula:
        Re = (V * L) / ν

    Parameters
    ----------
    velocity : float
        Characteristic velocity, V [m/s].
    length : float
        Characteristic length (e.g. pipe diameter), L [m].
    kinematic_viscosity : float
        Kinematic viscosity, ν [m^2/s].

    Returns
    -------
    float
        Reynolds number.
    """
    re = (velocity * length) / kinematic_viscosity
    return re


def main():
    """
    Main function of the script.

    Steps:
    1. Explain what the script does.
    2. Ask the user whether they want to use dynamic or kinematic viscosity.
    3. Ask for the necessary input values (with units).
    4. Compute the Reynolds number.
    5. Print the result and flow classification.
    """
    print("======================================================")
    print("     Reynolds Number Calculator (Educational Demo)     ")
    print("======================================================\n")

    print("You can calculate the Reynolds number in two ways:")
    print("  1) Using dynamic viscosity (μ): Re = (ρ * V * L) / μ")
    print("  2) Using kinematic viscosity (ν): Re = (V * L) / ν\n")

    print("Please choose which form you want to use:")
    print("  1 - Dynamic viscosity (μ)")
    print("  2 - Kinematic viscosity (ν)\n")

    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        # --- Use dynamic viscosity form ---
        print("\nYou chose: Dynamic viscosity (μ)")
        print("Please enter values in the SI units shown.\n")

        # Get user inputs with error checking
        density = get_positive_float("Fluid density ρ [kg/m^3]: ")
        velocity = get_positive_float("Velocity V [m/s]: ")
        length = get_positive_float("Characteristic length L [m] (e.g. pipe diameter): ")
        dynamic_viscosity = get_positive_float("Dynamic viscosity μ [Pa·s]: ")

        # Compute Reynolds number
        re = reynolds_dynamic_viscosity(density, velocity, length, dynamic_viscosity)

    elif choice == "2":
        # --- Use kinematic viscosity form ---
        print("\nYou chose: Kinematic viscosity (ν)")
        print("Please enter values in the SI units shown.\n")

        velocity = get_positive_float("Velocity V [m/s]: ")
        length = get_positive_float("Characteristic length L [m] (e.g. pipe diameter): ")
        kinematic_viscosity = get_positive_float("Kinematic viscosity ν [m^2/s]: ")

        # Compute Reynolds number
        re = reynolds_kinematic_viscosity(velocity, length, kinematic_viscosity)

    else:
        # If the user didn't enter 1 or 2, we exit the program.
        print("\nInvalid choice. Please run the script again and choose 1 or 2.")
        sys.exit(1)

    # Classify the flow regime
    flow_type = classify_flow(re)

    # Print the results in a friendly format
    print("\n================= RESULTS =================")
    print(f"Reynolds number (Re) = {re:.2f}")
    print(f"Flow regime          = {flow_type}")
    print("===========================================")


# This ensures that main() only runs when the script is executed directly,
# not when it is imported as a module into another script.
if __name__ == "__main__":
    main()
