import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from pathlib import Path


# ============================================================
# Parameters from the paper
# ============================================================

DUFFING = {
    "a": 1.0,
    "b": 1.0,
    "d": 0.25,
}

LANGEVIN = {
    "a": 1.0,
    "b": 1.0,
    "d": 0.25,
    "noise_std": 0.08,
}

SEXTIC = {
    "a": 1.0,
    "b": 0.20,
    "c": 0.40,
    "d": 0.30,
}

SMOOTH_BIQUADRATIC = {
    "k": 0.60,
    "s": 1.0,
    "epsilon": 0.40,
    "d": 0.22,
}


# ============================================================
# System dynamics
# ============================================================

def duffing(t, state, p):
    x, v = state

    dxdt = v
    dvdt = p["a"] * x - p["b"] * x**3 - p["d"] * v

    return [dxdt, dvdt]


def sextic(t, state, p):
    x, v = state

    dxdt = v
    dvdt = (
        p["a"] * x
        - p["b"] * x**3
        - p["c"] * x**5
        - p["d"] * v
    )

    return [dxdt, dvdt]


def smooth_biquadratic(t, state, p):
    x, v = state

    force = p["k"] * (
        p["s"] / np.sqrt(x**2 + p["epsilon"]**2) - 1
    ) * x

    dxdt = v
    dvdt = force - p["d"] * v

    return [dxdt, dvdt]


# ============================================================
# Integrate one deterministic trajectory
# ============================================================

def integrate_trajectory(rhs, initial_state, params, t_max=30.0):
    solution = solve_ivp(
        lambda t, y: rhs(t, y, params),
        (0, t_max),
        initial_state,
        rtol=1e-8,
        atol=1e-10,
        max_step=0.05,
    )

    return solution.y[0], solution.y[1]


# ============================================================
# Determine which well a trajectory ends in
# ============================================================

def classify_trajectory(rhs, initial_state, params):
    x, v = integrate_trajectory(rhs, initial_state, params)

    final_x = x[-1]

    if final_x > 0:
        return 1
    else:
        return -1


# ============================================================
# Calculate basin diagram
# ============================================================

def calculate_basin(
    rhs,
    params,
    x_range=(-2.0, 2.0),
    v_range=(-1.5, 1.5),
    resolution=150,
):
    xs = np.linspace(*x_range, resolution)
    vs = np.linspace(*v_range, resolution)

    basin = np.zeros((resolution, resolution))

    total = resolution * resolution
    count = 0

    for i, v in enumerate(vs):
        for j, x in enumerate(xs):

            basin[i, j] = classify_trajectory(
                rhs,
                [x, v],
                params,
            )

            count += 1

        if i % 10 == 0:
            print(f"    {100 * count / total:.1f}% complete")

    return xs, vs, basin


# ============================================================
# Plot deterministic systems
# ============================================================

def plot_system(
    name,
    rhs,
    params,
    filename,
):
    print(f"\nCalculating basin diagram for {name}...")

    xs, vs, basin = calculate_basin(
        rhs,
        params,
    )

    X, V = np.meshgrid(xs, vs)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Basin regions
    ax.contourf(
        X,
        V,
        basin,
        levels=[-1.5, 0, 1.5],
        alpha=0.25,
    )

    # Basin boundary
    ax.contour(
        X,
        V,
        basin,
        levels=[0],
        linewidths=2,
    )

    # A few representative trajectories
    initial_conditions = [
        [-1.5, 0.0],
        [-1.0, 0.5],
        [-1.0, -0.5],
        [1.0, 0.5],
        [1.0, -0.5],
        [1.5, 0.0],
        [0.2, 0.0],
        [-0.2, 0.0],
    ]

    for initial_state in initial_conditions:

        x, v = integrate_trajectory(
            rhs,
            initial_state,
            params,
        )

        ax.plot(
            x,
            v,
            linewidth=1.2,
        )

        ax.scatter(
            x[0],
            v[0],
            s=20,
        )

    # Unstable equilibrium
    ax.scatter(
        0,
        0,
        marker="x",
        s=60,
        linewidths=2,
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")
    ax.set_title(name)

    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)

    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved {filename}")


# ============================================================
# Langevin simulation
# ============================================================

def simulate_langevin(
    initial_state,
    params,
    t_max=30.0,
    dt=0.01,
    seed=None,
):
    """
    Simulate the Langevin Duffing system.

    The deterministic Duffing component is integrated with RK4.
    The stochastic force is then applied as a Gaussian velocity kick.
    """

    rng = np.random.default_rng(seed)

    n_steps = int(t_max / dt)

    x = np.zeros(n_steps + 1)
    v = np.zeros(n_steps + 1)

    x[0], v[0] = initial_state

    def rhs(state):
        x_val, v_val = state

        return np.array([
            v_val,
            (
                params["a"] * x_val
                - params["b"] * x_val**3
                - params["d"] * v_val
            ),
        ])

    # RK4 + stochastic velocity increment
    for i in range(n_steps):

        state = np.array([x[i], v[i]])

        k1 = rhs(state)
        k2 = rhs(state + 0.5 * dt * k1)
        k3 = rhs(state + 0.5 * dt * k2)
        k4 = rhs(state + dt * k3)

        deterministic_update = (
            dt / 6
            * (k1 + 2*k2 + 2*k3 + k4)
        )

        new_state = state + deterministic_update

        # Langevin stochastic forcing
        noise = (
            params["noise_std"]
            * np.sqrt(dt)
            * rng.normal()
        )

        new_state[1] += noise

        x[i + 1] = new_state[0]
        v[i + 1] = new_state[1]

    return x, v


# ============================================================
# Plot Langevin system
# ============================================================

def plot_langevin(filename):

    print("\nCalculating deterministic basin boundary for Langevin...")

    # The Langevin system has the same deterministic potential as
    # Duffing. We therefore calculate the underlying deterministic
    # basin boundary using the Duffing dynamics.
    xs, vs, basin = calculate_basin(
        duffing,
        {
            "a": LANGEVIN["a"],
            "b": LANGEVIN["b"],
            "d": LANGEVIN["d"],
        },
    )

    X, V = np.meshgrid(xs, vs)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Underlying deterministic basins
    ax.contourf(
        X,
        V,
        basin,
        levels=[-1.5, 0, 1.5],
        alpha=0.25,
    )

    # Deterministic basin boundary
    ax.contour(
        X,
        V,
        basin,
        levels=[0],
        linewidths=2,
    )

    # Stochastic trajectories
    initial_conditions = [
        [-1.2, 0.0],
        [-0.8, 0.5],
        [-0.8, -0.5],
        [0.8, 0.5],
        [0.8, -0.5],
        [1.2, 0.0],
    ]

    for seed, initial_state in enumerate(initial_conditions):

        x, v = simulate_langevin(
            initial_state,
            LANGEVIN,
            seed=seed,
        )

        ax.plot(
            x,
            v,
            linewidth=1.0,
            alpha=0.8,
        )

        ax.scatter(
            x[0],
            v[0],
            s=20,
        )

    ax.scatter(
        0,
        0,
        marker="x",
        s=60,
        linewidths=2,
    )

    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$v$")

    ax.set_title(
        "Langevin Duffing Oscillator"
    )

    ax.set_xlim(-2, 2)
    ax.set_ylim(-1.5, 1.5)

    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    print(f"Saved {filename}")


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    output_dir = Path("figures/phase_space")
    output_dir.mkdir(exist_ok=True)

    plot_system(
        "Quartic Duffing Oscillator",
        duffing,
        DUFFING,
        output_dir / "duffing_phase_space.png",
    )

    plot_langevin(
        output_dir / "langevin_phase_space.png"
    )

    plot_system(
        "Sextic-Potential Oscillator",
        sextic,
        SEXTIC,
        output_dir / "sextic_phase_space.png",
    )

    plot_system(
        "Smooth-Biquadratic Oscillator",
        smooth_biquadratic,
        SMOOTH_BIQUADRATIC,
        output_dir / "smooth_biquadratic_phase_space.png",
    )

    print("\nAll phase-space diagrams generated.")