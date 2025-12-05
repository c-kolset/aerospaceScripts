"""
reynolds_gui.py

Educational Reynolds Number Calculator with a simple Tkinter GUI.

Features:
- Lets the user choose between:
    1) Dynamic viscosity form: Re = (ρ * V * L) / μ
    2) Kinematic viscosity form: Re = (V * L) / ν
- Accepts user input via text boxes.
- Shows the Reynolds number and flow classification.
- Provides TWO example problems that can auto-fill the input fields.

This script is heavily commented for students learning:
- Basic fluid mechanics (Reynolds number),
- Python functions,
- and simple GUI programming with Tkinter.
"""

import tkinter as tk
from tkinter import ttk, messagebox


# -----------------------------
# 1. Core Reynolds Number Logic
# -----------------------------
def classify_flow(re):
    """
    Classify flow based on Reynolds number.

    Common guideline for flow in a pipe:
        Re < 2300        -> Laminar flow
        2300 <= Re < 4000 -> Transitional flow
        Re >= 4000        -> Turbulent flow

    Parameters
    ----------
    re : float
        Reynolds number.

    Returns
    -------
    str
        Description of the flow regime.
    """
    if re < 2300:
        return "Laminar flow"
    elif re < 4000:
        return "Transitional flow"
    else:
        return "Turbulent flow"


def reynolds_dynamic_viscosity(density, velocity, length, dynamic_viscosity):
    """
    Compute Reynolds number using dynamic viscosity (μ).

    Formula:
        Re = (ρ * V * L) / μ

    Parameters
    ----------
    density : float
        Fluid density, ρ [kg/m^3].
    velocity : float
        Velocity, V [m/s].
    length : float
        Characteristic length (e.g., pipe diameter), L [m].
    dynamic_viscosity : float
        Dynamic viscosity, μ [Pa·s].

    Returns
    -------
    float
        Reynolds number.
    """
    return (density * velocity * length) / dynamic_viscosity


def reynolds_kinematic_viscosity(velocity, length, kinematic_viscosity):
    """
    Compute Reynolds number using kinematic viscosity (ν).

    Formula:
        Re = (V * L) / ν

    Parameters
    ----------
    velocity : float
        Velocity, V [m/s].
    length : float
        Characteristic length, L [m].
    kinematic_viscosity : float
        Kinematic viscosity, ν [m^2/s].

    Returns
    -------
    float
        Reynolds number.
    """
    return (velocity * length) / kinematic_viscosity


# -------------------------------------------------
# 2. Example Problems (for students to experiment)
# -------------------------------------------------
EXAMPLES = {
    "Example 1: Water in small pipe (μ-form)": {
        "mode": "dynamic",
        "description": (
            "Example 1: Water flowing in a small circular pipe.\n\n"
            "Given:\n"
            "  Fluid: Water at ~20°C\n"
            "  Density ρ ≈ 998 kg/m³\n"
            "  Velocity V = 1.5 m/s\n"
            "  Pipe diameter L = 0.05 m\n"
            "  Dynamic viscosity μ ≈ 0.001 Pa·s\n\n"
            "Task: Compute the Reynolds number and classify the flow."
        ),
        "density": 998.0,
        "velocity": 1.5,
        "length": 0.05,
        "dynamic_viscosity": 0.001,
        "kinematic_viscosity": None,  # Not used in this example
    },
    "Example 2: Oil in large pipe (ν-form)": {
        "mode": "kinematic",
        "description": (
            "Example 2: Heavy oil flowing in a large pipe.\n\n"
            "Given:\n"
            "  Velocity V = 0.8 m/s\n"
            "  Pipe diameter L = 0.3 m\n"
            "  Kinematic viscosity ν = 5×10⁻⁵ m²/s\n\n"
            "Task: Compute the Reynolds number and classify the flow."
        ),
        "density": None,             # Not used in this example
        "velocity": 0.8,
        "length": 0.3,
        "dynamic_viscosity": None,   # Not used in this example
        "kinematic_viscosity": 5e-5,
    },
}


# -----------------------------
# 3. Tkinter GUI Application
# -----------------------------
class ReynoldsGUI:
    """
    A simple Tkinter-based GUI for computing Reynolds number.

    This class builds the interface and handles user interaction.
    """

    def __init__(self, root):
        """
        Initialize the GUI and create all widgets.

        Parameters
        ----------
        root : tk.Tk
            The main Tkinter window.
        """
        self.root = root
        self.root.title("Reynolds Number Calculator (Educational Demo)")

        # Use a Tkinter 'StringVar' to keep track of which mode is selected:
        #   "dynamic"   -> Re = (ρ V L) / μ
        #   "kinematic" -> Re = (V L) / ν
        self.mode_var = tk.StringVar(value="dynamic")

        # Create all the UI elements
        self.create_widgets()

    def create_widgets(self):
        """
        Create and lay out all the widgets in the GUI.
        """
        # Use a ttk.Frame as a main container with some padding
        main_frame = ttk.Frame(self.root, padding="10 10 10 10")
        main_frame.grid(row=0, column=0, sticky="NSEW")

        # Allow the frame to expand with the window
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # -----------------
        # Mode selection
        # -----------------
        mode_label = ttk.Label(
            main_frame,
            text="Choose formula:",
            font=("Segoe UI", 10, "bold")
        )
        mode_label.grid(row=0, column=0, columnspan=2, sticky="W")

        # Radio button: Dynamic viscosity
        self.radio_dynamic = ttk.Radiobutton(
            main_frame,
            text="Use dynamic viscosity μ   (Re = ρ V L / μ)",
            variable=self.mode_var,
            value="dynamic",
            command=self.update_input_state
        )
        self.radio_dynamic.grid(row=1, column=0, columnspan=2, sticky="W")

        # Radio button: Kinematic viscosity
        self.radio_kinematic = ttk.Radiobutton(
            main_frame,
            text="Use kinematic viscosity ν (Re = V L / ν)",
            variable=self.mode_var,
            value="kinematic",
            command=self.update_input_state
        )
        self.radio_kinematic.grid(row=2, column=0, columnspan=2, sticky="W")

        # A small separator line
        sep = ttk.Separator(main_frame, orient="horizontal")
        sep.grid(row=3, column=0, columnspan=4, sticky="EW", pady=8)

        # -----------------
        # Input fields
        # -----------------
        # We'll use Entry widgets for user input and keep references to them.

        # Row index helper (so we can add rows cleanly)
        row = 4

        # Density ρ (only needed for dynamic viscosity mode)
        ttk.Label(main_frame, text="Density ρ [kg/m³]:").grid(row=row, column=0, sticky="E", pady=2)
        self.entry_density = ttk.Entry(main_frame, width=15)
        self.entry_density.grid(row=row, column=1, sticky="W", pady=2)
        row += 1

        # Velocity V
        ttk.Label(main_frame, text="Velocity V [m/s]:").grid(row=row, column=0, sticky="E", pady=2)
        self.entry_velocity = ttk.Entry(main_frame, width=15)
        self.entry_velocity.grid(row=row, column=1, sticky="W", pady=2)
        row += 1

        # Characteristic length L (e.g., pipe diameter)
        ttk.Label(main_frame, text="Length L [m]:").grid(row=row, column=0, sticky="E", pady=2)
        self.entry_length = ttk.Entry(main_frame, width=15)
        self.entry_length.grid(row=row, column=1, sticky="W", pady=2)
        row += 1

        # Dynamic viscosity μ (only for dynamic mode)
        ttk.Label(main_frame, text="Dynamic viscosity μ [Pa·s]:").grid(row=row, column=0, sticky="E", pady=2)
        self.entry_mu = ttk.Entry(main_frame, width=15)
        self.entry_mu.grid(row=row, column=1, sticky="W", pady=2)
        row += 1

        # Kinematic viscosity ν (only for kinematic mode)
        ttk.Label(main_frame, text="Kinematic viscosity ν [m²/s]:").grid(row=row, column=0, sticky="E", pady=2)
        self.entry_nu = ttk.Entry(main_frame, width=15)
        self.entry_nu.grid(row=row, column=1, sticky="W", pady=2)
        row += 1

        # At initialization, set appropriate enabled/disabled fields
        self.update_input_state()

        # -----------------
        # Buttons
        # -----------------
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=row, column=0, columnspan=2, pady=8, sticky="W")
        row += 1

        # Calculate button
        calc_button = ttk.Button(btn_frame, text="Calculate Re", command=self.calculate_reynolds)
        calc_button.grid(row=0, column=0, padx=5)

        # Clear button
        clear_button = ttk.Button(btn_frame, text="Clear Inputs", command=self.clear_inputs)
        clear_button.grid(row=0, column=1, padx=5)

        # -----------------
        # Example problems
        # -----------------
        example_label = ttk.Label(
            main_frame,
            text="Example problems:",
            font=("Segoe UI", 10, "bold")
        )
        example_label.grid(row=row, column=0, columnspan=2, sticky="W")
        row += 1

        # A combobox to choose which example to load
        self.example_var = tk.StringVar(value="Select an example...")
        self.example_combo = ttk.Combobox(
            main_frame,
            textvariable=self.example_var,
            values=list(EXAMPLES.keys()),
            state="readonly",
            width=40
        )
        self.example_combo.grid(row=row, column=0, columnspan=1, sticky="W", pady=2)
        self.example_combo.bind("<<ComboboxSelected>>", self.load_example_from_combo)

        load_example_button = ttk.Button(main_frame, text="Load Example", command=self.load_example_button)
        load_example_button.grid(row=row, column=1, sticky="W", padx=5)
        row += 1

        # -----------------
        # Output area
        # -----------------
        # Label for results
        result_label = ttk.Label(
            main_frame,
            text="Result:",
            font=("Segoe UI", 10, "bold")
        )
        result_label.grid(row=row, column=0, columnspan=2, sticky="W")
        row += 1

        # A Label widget to show Reynolds number and flow regime
        self.label_result = ttk.Label(
            main_frame,
            text="Reynolds number and flow classification will appear here.",
            wraplength=400,
            justify="left"
        )
        self.label_result.grid(row=row, column=0, columnspan=2, sticky="W")
        row += 1

        # A larger Text widget to show example descriptions and/or steps
        steps_label = ttk.Label(
            main_frame,
            text="Example description / steps:",
            font=("Segoe UI", 10, "bold")
        )
        steps_label.grid(row=row, column=0, columnspan=2, sticky="W")
        row += 1

        self.text_steps = tk.Text(main_frame, width=60, height=10, wrap="word")
        self.text_steps.grid(row=row, column=0, columnspan=2, pady=5, sticky="NSEW")

        # Make the text box expandable
        main_frame.rowconfigure(row, weight=1)
        main_frame.columnconfigure(1, weight=1)

    # -----------------------------
    # GUI helper methods
    # -----------------------------
    def update_input_state(self):
        """
        Enable/disable input fields depending on selected mode.

        - Dynamic mode: need ρ, V, L, μ
        - Kinematic mode: need V, L, ν
        """
        mode = self.mode_var.get()

        if mode == "dynamic":
            # Enable density and μ
            self.entry_density.config(state="normal")
            self.entry_mu.config(state="normal")

            # Disable ν
            self.entry_nu.delete(0, tk.END)
            self.entry_nu.config(state="disabled")

        elif mode == "kinematic":
            # Disable density and μ
            self.entry_density.delete(0, tk.END)
            self.entry_mu.delete(0, tk.END)
            self.entry_density.config(state="disabled")
            self.entry_mu.config(state="disabled")

            # Enable ν
            self.entry_nu.config(state="normal")

    def clear_inputs(self):
        """
        Clear all input fields and output labels.
        """
        for entry in [self.entry_density, self.entry_velocity, self.entry_length,
                      self.entry_mu, self.entry_nu]:
            entry.config(state="normal")  # make sure we can clear even if disabled
            entry.delete(0, tk.END)

        self.update_input_state()  # re-disable correct fields

        # Reset results and text area
        self.label_result.config(text="Reynolds number and flow classification will appear here.")
        self.text_steps.delete("1.0", tk.END)

    def calculate_reynolds(self):
        """
        Read input values from the GUI, compute Reynolds number, and show results.

        Also writes a short 'step-by-step' explanation in the text box.
        """
        mode = self.mode_var.get()

        try:
            # Velocity and length are needed for both modes
            V = float(self.entry_velocity.get())
            L = float(self.entry_length.get())
            if V <= 0 or L <= 0:
                raise ValueError("V and L must be positive.")

            steps = []  # Collect lines of text to show in the steps box

            if mode == "dynamic":
                # Dynamic mode needs density and μ
                rho = float(self.entry_density.get())
                mu = float(self.entry_mu.get())
                if rho <= 0 or mu <= 0:
                    raise ValueError("ρ and μ must be positive.")

                # Compute Re
                Re = reynolds_dynamic_viscosity(rho, V, L, mu)

                # Build explanation text
                steps.append("Using dynamic viscosity form:")
                steps.append("  Re = (ρ * V * L) / μ")
                steps.append(f"  ρ = {rho:.3g} kg/m³")
                steps.append(f"  V = {V:.3g} m/s")
                steps.append(f"  L = {L:.3g} m")
                steps.append(f"  μ = {mu:.3g} Pa·s")
                steps.append("")
                steps.append("Substitute values:")
                steps.append(f"  Re = ({rho:.3g} * {V:.3g} * {L:.3g}) / {mu:.3g}")
                steps.append(f"  Re ≈ {Re:.2f}")

            else:  # mode == "kinematic"
                # Kinematic mode needs ν
                nu = float(self.entry_nu.get())
                if nu <= 0:
                    raise ValueError("ν must be positive.")

                # Compute Re
                Re = reynolds_kinematic_viscosity(V, L, nu)

                # Build explanation text
                steps.append("Using kinematic viscosity form:")
                steps.append("  Re = (V * L) / ν")
                steps.append(f"  V = {V:.3g} m/s")
                steps.append(f"  L = {L:.3g} m")
                steps.append(f"  ν = {nu:.3g} m²/s")
                steps.append("")
                steps.append("Substitute values:")
                steps.append(f"  Re = ({V:.3g} * {L:.3g}) / {nu:.3g}")
                steps.append(f"  Re ≈ {Re:.2f}")

            # Classify the flow
            flow_type = classify_flow(Re)

        except ValueError as e:
            # Show an error dialog if any conversion fails or values are invalid
            messagebox.showerror("Input Error", f"Please check your inputs:\n{e}")
            return

        # Update result label
        result_text = f"Reynolds number Re ≈ {Re:.2f}\nFlow regime: {flow_type}"
        self.label_result.config(text=result_text)

        # Show the step-by-step explanation in the text box
        self.text_steps.delete("1.0", tk.END)
        self.text_steps.insert(tk.END, "\n".join(steps))

    # -----------------------------
    # Example-loading methods
    # -----------------------------
    def load_example_from_combo(self, event):
        """
        Callback when an example is selected from the combobox.

        This just calls 'load_example_button' which does the actual work.
        """
        self.load_example_button()

    def load_example_button(self):
        """
        Load the currently selected example problem into the input fields.

        Also writes the example description into the text area.
        """
        example_name = self.example_var.get()
        if example_name not in EXAMPLES:
            messagebox.showinfo("Example", "Please select a valid example from the list.")
            return

        example = EXAMPLES[example_name]

        # Set the mode (dynamic/kinematic) based on the example
        self.mode_var.set(example["mode"])
        self.update_input_state()

        # Clear any existing inputs
        for entry in [self.entry_density, self.entry_velocity, self.entry_length,
                      self.entry_mu, self.entry_nu]:
            entry.config(state="normal")
            entry.delete(0, tk.END)

        # Fill in the example values if they exist
        if example["density"] is not None:
            self.entry_density.insert(0, str(example["density"]))
        if example["velocity"] is not None:
            self.entry_velocity.insert(0, str(example["velocity"]))
        if example["length"] is not None:
            self.entry_length.insert(0, str(example["length"]))
        if example["dynamic_viscosity"] is not None:
            self.entry_mu.insert(0, str(example["dynamic_viscosity"]))
        if example["kinematic_viscosity"] is not None:
            self.entry_nu.insert(0, str(example["kinematic_viscosity"]))

        # Show the example description in the text box
        self.text_steps.delete("1.0", tk.END)
        self.text_steps.insert(tk.END, example["description"])

        # Clear the result label (students should hit "Calculate" themselves)
        self.label_result.config(
            text="Now click 'Calculate Re' to solve this example."
        )


# -----------------------------
# 4. Main program entry point
# -----------------------------
if __name__ == "__main__":
    # Create the main Tkinter window
    root = tk.Tk()

    # Create our ReynoldsGUI app inside this window
    app = ReynoldsGUI(root)

    # Start the Tkinter event loop (this keeps the window open)
    root.mainloop()
