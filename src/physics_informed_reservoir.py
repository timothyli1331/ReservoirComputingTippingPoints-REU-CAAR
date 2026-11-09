"""Forecast a dynamical system with a reservoir made of Duffing oscillators.

This module is an alternative to an echo-state network with a random recurrent
graph.  The reservoir here is a bank of independent, input-driven Duffing
oscillators.  Their positions and velocities provide nonlinear features with
physical memory, and a linear ridge readout is the only trained component.

The command-line demo learns a periodically forced Duffing trajectory::

    .venv/bin/python src/physics_informed_reservoir.py \
        --samples 8000 --units 48 --plot results/physical_reservoir_demo.png

The phase of the known periodic forcing is supplied as an exogenous input.  In
free-running mode, predicted ``[x, velocity]`` values are fed back while the
future forcing phase is supplied exactly.  This avoids asking the reservoir to
infer a clock that is already known to the simulator.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np

from dynamics import integrate_step


class DuffingPhysicalReservoir:
    """A dynamics-informed reservoir composed of Duffing oscillator nodes.

    Unlike a conventional echo-state network, this class has no random
    recurrent adjacency matrix.  Node ``i`` follows

    ``q_i'' + damping_i*q_i' + alpha_i*q_i + beta_i*q_i**3 = drive_i(input)``.

    Small, fixed variations in the oscillator parameters give the readout a
    collection of different nonlinear time scales.  The deterministic input
    projection distinguishes the nodes without introducing node-to-node
    connections.  Training solves one ridge-regression problem and never
    changes the physical reservoir.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        units: int = 48,
        dt: float = 0.04,
        substeps: int = 2,
        alpha: float = -1.0,
        beta: float = 1.0,
        damping: float = 0.25,
        parameter_spread: float = 0.18,
        input_scaling: float = 0.8,
        ridge: float = 1e-5,
        state_clip: float = 4.0,
        residual_indices: Sequence[int] | None = None,
    ) -> None:
        if input_size < 1 or output_size < 1 or units < 1:
            raise ValueError("input_size, output_size, and units must be positive")
        if dt <= 0 or substeps < 1:
            raise ValueError("dt must be positive and substeps must be at least 1")
        if beta <= 0 or damping < 0:
            raise ValueError("beta must be positive and damping must be nonnegative")
        if not 0 <= parameter_spread < 1:
            raise ValueError("parameter_spread must be in [0, 1)")
        if ridge < 0 or state_clip <= 0:
            raise ValueError("ridge must be nonnegative and state_clip must be positive")

        self.input_size = int(input_size)
        self.output_size = int(output_size)
        self.units = int(units)
        self.dt = float(dt)
        self.substeps = int(substeps)
        self.input_scaling = float(input_scaling)
        self.ridge = float(ridge)
        self.state_clip = float(state_clip)
        if residual_indices is None:
            self.residual_indices: tuple[int, ...] | None = None
        else:
            residual = tuple(int(index) for index in residual_indices)
            if (
                len(residual) != self.output_size
                or len(set(residual)) != len(residual)
                or any(index < 0 or index >= self.input_size for index in residual)
            ):
                raise ValueError(
                    "residual_indices must contain one unique valid input index per output"
                )
            self.residual_indices = residual

        # Smoothly distributed parameter variations make the oscillator bank
        # reproducible.  There is deliberately no reservoir adjacency matrix.
        angles = 2.0 * np.pi * (np.arange(self.units) + 0.5) / self.units
        self.alphas = float(alpha) * (1.0 + parameter_spread * np.cos(angles))
        self.betas = float(beta) * (
            1.0 + parameter_spread * np.sin(angles + np.pi / 5.0)
        )
        self.dampings = float(damping) * (
            1.0 + parameter_spread * np.cos(angles + np.pi / 3.0)
        )

        # A fixed structured projection multiplexes each observed variable into
        # a different drive.  It is deterministic, bounded, and not recurrent.
        node_index = np.arange(1, self.units + 1, dtype=float)[:, None]
        input_index = np.arange(1, self.input_size + 1, dtype=float)[None, :]
        golden_ratio = (1.0 + np.sqrt(5.0)) / 2.0
        projection = np.sin(node_index * input_index * golden_ratio)
        projection += 0.5 * np.cos(node_index * (input_index + 0.5))
        norms = np.linalg.norm(projection, axis=1, keepdims=True)
        self.input_weights = projection / np.maximum(norms, 1e-12)
        self.input_bias = 0.15 * np.sin(angles)

        self.positions = np.zeros(self.units, dtype=float)
        self.velocities = np.zeros(self.units, dtype=float)
        self.input_mean = np.zeros(self.input_size, dtype=float)
        self.input_scale = np.ones(self.input_size, dtype=float)
        self.output_mean = np.zeros(self.output_size, dtype=float)
        self.output_scale = np.ones(self.output_size, dtype=float)
        self.readout_weights: np.ndarray | None = None

    @property
    def feature_size(self) -> int:
        """Number of features seen by the trained linear readout."""
        # bias + instantaneous polynomial terms + physical reservoir state.
        # Cubic input terms match the leading Duffing nonlinearity, while the
        # oscillator states supply history-dependent features.
        return 1 + 3 * self.input_size + 5 * self.units

    def reset(self, state: Sequence[float] | None = None) -> None:
        """Reset every physical node, optionally to one shared ``[q, p]``."""
        if state is None:
            self.positions.fill(0.0)
            self.velocities.fill(0.0)
            return
        oscillator_state = np.asarray(state, dtype=float)
        if oscillator_state.shape != (2,):
            raise ValueError("reservoir reset state must have shape (2,)")
        self.positions.fill(float(oscillator_state[0]))
        self.velocities.fill(float(oscillator_state[1]))

    def _derivative(
        self, positions: np.ndarray, velocities: np.ndarray, drive: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        acceleration = (
            drive
            - self.dampings * velocities
            - self.alphas * positions
            - self.betas * positions**3
        )
        return velocities, acceleration

    def _advance(self, normalized_input: np.ndarray) -> None:
        """Advance all physical nodes by one sample using RK4."""
        drive = self.input_scaling * np.tanh(
            self.input_weights @ normalized_input + self.input_bias
        )
        h = self.dt / self.substeps
        q = self.positions
        p = self.velocities
        for _ in range(self.substeps):
            k1_q, k1_p = self._derivative(q, p, drive)
            k2_q, k2_p = self._derivative(
                q + 0.5 * h * k1_q, p + 0.5 * h * k1_p, drive
            )
            k3_q, k3_p = self._derivative(
                q + 0.5 * h * k2_q, p + 0.5 * h * k2_p, drive
            )
            k4_q, k4_p = self._derivative(q + h * k3_q, p + h * k3_p, drive)
            q = q + (h / 6.0) * (k1_q + 2.0 * k2_q + 2.0 * k3_q + k4_q)
            p = p + (h / 6.0) * (k1_p + 2.0 * k2_p + 2.0 * k3_p + k4_p)
            q = np.clip(q, -self.state_clip, self.state_clip)
            p = np.clip(p, -self.state_clip, self.state_clip)
        self.positions = q
        self.velocities = p

    def _features(self, normalized_input: np.ndarray) -> np.ndarray:
        q = self.positions
        p = self.velocities
        return np.concatenate(
            (
                [1.0],
                normalized_input,
                normalized_input**2,
                normalized_input**3,
                q,
                p,
                q**2,
                q * p,
                q**3,
            )
        )

    @staticmethod
    def _as_matrix(values, columns: int, name: str) -> np.ndarray:
        matrix = np.asarray(values, dtype=float)
        if matrix.ndim != 2 or matrix.shape[1] != columns:
            raise ValueError(f"{name} must have shape (samples, {columns})")
        if len(matrix) == 0 or not np.all(np.isfinite(matrix)):
            raise ValueError(f"{name} must be nonempty and finite")
        return matrix

    def _normalize_input(self, value: np.ndarray) -> np.ndarray:
        return (value - self.input_mean) / self.input_scale

    def _step_features(self, value: np.ndarray) -> np.ndarray:
        normalized = self._normalize_input(value)
        self._advance(normalized)
        return self._features(normalized)

    def _set_normalization(self, inputs: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Fit data scaling and return raw targets for the linear readout."""
        self.input_mean = inputs.mean(axis=0)
        self.input_scale = inputs.std(axis=0)
        self.input_scale[self.input_scale < 1e-12] = 1.0
        regression_targets = targets.copy()
        if self.residual_indices is not None:
            regression_targets -= inputs[:, self.residual_indices]
        self.output_mean = regression_targets.mean(axis=0)
        self.output_scale = regression_targets.std(axis=0)
        self.output_scale[self.output_scale < 1e-12] = 1.0
        return regression_targets

    def _fit_readout(self, design: np.ndarray, regression_targets: np.ndarray) -> None:
        """Solve the normalized ridge-regression readout."""
        normalized_targets = (
            regression_targets - self.output_mean
        ) / self.output_scale
        penalty = self.ridge * np.eye(self.feature_size)
        penalty[0, 0] = 0.0  # Do not regularize the intercept.
        gram = design.T @ design + penalty
        rhs = design.T @ normalized_targets
        try:
            self.readout_weights = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            self.readout_weights = np.linalg.pinv(gram) @ rhs

    def fit(self, inputs, targets, washout: int = 100) -> "DuffingPhysicalReservoir":
        """Fit the linear readout to one teacher-forced trajectory.

        ``inputs[t]`` is used to predict ``targets[t]``.  For one-step state
        forecasting, that normally means ``targets[t] == state[t + 1]``.
        Initial samples are discarded so the physical nodes can synchronize to
        the observed trajectory.
        """
        x = self._as_matrix(inputs, self.input_size, "inputs")
        y = self._as_matrix(targets, self.output_size, "targets")
        if len(x) != len(y):
            raise ValueError("inputs and targets must contain the same number of samples")
        if not 0 <= washout < len(x):
            raise ValueError("washout must be nonnegative and smaller than sample count")

        regression_targets = self._set_normalization(x, y)

        self.reset()
        design = np.empty((len(x) - washout, self.feature_size), dtype=float)
        for index, value in enumerate(x):
            features = self._step_features(value)
            if index >= washout:
                design[index - washout] = features
        self._fit_readout(design, regression_targets[washout:])
        return self

    def fit_sequences(
        self, input_sequences, target_sequences, washout: int = 100
    ) -> "DuffingPhysicalReservoir":
        """Fit on independent teacher-forced trajectories.

        Each sequence is reset and washed out independently.  This is useful
        for multi-system experiments, where carrying physical reservoir state
        from the end of one system's trajectory into another would be an
        artificial source of information.
        """
        input_blocks = [
            self._as_matrix(values, self.input_size, "input sequence")
            for values in input_sequences
        ]
        target_blocks = [
            self._as_matrix(values, self.output_size, "target sequence")
            for values in target_sequences
        ]
        if not input_blocks or len(input_blocks) != len(target_blocks):
            raise ValueError("input_sequences and target_sequences must be nonempty and paired")
        if any(len(x) != len(y) for x, y in zip(input_blocks, target_blocks)):
            raise ValueError("every input sequence must match its target sequence length")
        if any(not 0 <= washout < len(x) for x in input_blocks):
            raise ValueError("washout must be nonnegative and smaller than every sequence")

        all_inputs = np.vstack(input_blocks)
        all_targets = np.vstack(target_blocks)
        self._set_normalization(all_inputs, all_targets)
        design_blocks = []
        regression_blocks = []
        for x, y in zip(input_blocks, target_blocks):
            regression_targets = y.copy()
            if self.residual_indices is not None:
                regression_targets -= x[:, self.residual_indices]
            self.reset()
            design = np.empty((len(x) - washout, self.feature_size), dtype=float)
            for index, value in enumerate(x):
                features = self._step_features(value)
                if index >= washout:
                    design[index - washout] = features
            design_blocks.append(design)
            regression_blocks.append(regression_targets[washout:])
        self._fit_readout(np.vstack(design_blocks), np.vstack(regression_blocks))
        return self

    def _check_fitted(self) -> None:
        if self.readout_weights is None:
            raise RuntimeError("fit the reservoir before requesting predictions")

    def step(self, input_value) -> np.ndarray:
        """Advance the reservoir and return one prediction."""
        self._check_fitted()
        value = np.asarray(input_value, dtype=float)
        if value.shape != (self.input_size,) or not np.all(np.isfinite(value)):
            raise ValueError(f"input_value must be finite with shape ({self.input_size},)")
        normalized_output = self._step_features(value) @ self.readout_weights
        output = self.output_mean + self.output_scale * normalized_output
        if self.residual_indices is not None:
            output = output + value[list(self.residual_indices)]
        return output

    def warmup(self, inputs) -> None:
        """Synchronize reservoir memory to history without producing output."""
        history = np.asarray(inputs, dtype=float)
        if history.size == 0:
            return
        history = self._as_matrix(history, self.input_size, "inputs")
        for value in history:
            self._step_features(value)

    def predict(self, inputs, reset: bool = True, warmup_inputs=None) -> np.ndarray:
        """Return teacher-forced one-step predictions for an input sequence."""
        self._check_fitted()
        x = self._as_matrix(inputs, self.input_size, "inputs")
        if reset:
            self.reset()
        if warmup_inputs is not None:
            self.warmup(warmup_inputs)
        return np.vstack([self.step(value) for value in x])

    def forecast(
        self,
        initial_input,
        steps: int,
        feedback_indices: Sequence[int] | None = None,
        exogenous_future=None,
        warmup_inputs=None,
        reset: bool = True,
    ) -> np.ndarray:
        """Generate an autoregressive forecast with optional exogenous inputs.

        The prediction is written into ``feedback_indices`` to construct the
        next reservoir input.  All remaining input columns are exogenous and
        must be supplied by ``exogenous_future`` for the following time steps.
        For example, with input ``[x, v, sin(phase), cos(phase)]`` and output
        ``[next_x, next_v]``, use ``feedback_indices=(0, 1)`` and pass future
        phase columns in ``exogenous_future``.
        """
        self._check_fitted()
        if steps < 1:
            raise ValueError("steps must be at least 1")
        current = np.asarray(initial_input, dtype=float).copy()
        if current.shape != (self.input_size,) or not np.all(np.isfinite(current)):
            raise ValueError(f"initial_input must be finite with shape ({self.input_size},)")
        if feedback_indices is None:
            feedback_indices = tuple(range(self.output_size))
        feedback = np.asarray(tuple(feedback_indices), dtype=int)
        if feedback.shape != (self.output_size,):
            raise ValueError("feedback_indices must contain one index per output")
        if len(np.unique(feedback)) != len(feedback) or np.any(feedback < 0) or np.any(
            feedback >= self.input_size
        ):
            raise ValueError("feedback_indices must be unique valid input columns")

        exogenous_indices = np.asarray(
            [index for index in range(self.input_size) if index not in set(feedback)],
            dtype=int,
        )
        required_rows = max(0, steps - 1)
        if len(exogenous_indices):
            exogenous = np.asarray(exogenous_future, dtype=float)
            expected = (required_rows, len(exogenous_indices))
            if exogenous.shape != expected or not np.all(np.isfinite(exogenous)):
                raise ValueError(f"exogenous_future must be finite with shape {expected}")
        else:
            exogenous = np.empty((required_rows, 0), dtype=float)

        if reset:
            self.reset()
        if warmup_inputs is not None:
            self.warmup(warmup_inputs)
        predictions = np.empty((steps, self.output_size), dtype=float)
        for index in range(steps):
            prediction = self.step(current)
            predictions[index] = prediction
            if index + 1 < steps:
                current[feedback] = prediction
                current[exogenous_indices] = exogenous[index]
        return predictions

    def save(self, path: str | Path) -> Path:
        """Serialize the fitted physical reservoir."""
        destination = Path(path)
        if destination.suffix != ".pkl":
            destination = destination.with_suffix(".pkl")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "duffing-physical-reservoir-v1",
            "config": {
                "input_size": self.input_size,
                "output_size": self.output_size,
                "units": self.units,
                "dt": self.dt,
                "substeps": self.substeps,
                "input_scaling": self.input_scaling,
                "ridge": self.ridge,
                "state_clip": self.state_clip,
                "residual_indices": self.residual_indices,
            },
            "arrays": {
                "alphas": self.alphas,
                "betas": self.betas,
                "dampings": self.dampings,
                "input_weights": self.input_weights,
                "input_bias": self.input_bias,
                "positions": self.positions,
                "velocities": self.velocities,
                "input_mean": self.input_mean,
                "input_scale": self.input_scale,
                "output_mean": self.output_mean,
                "output_scale": self.output_scale,
                "readout_weights": self.readout_weights,
            },
        }
        with destination.open("wb") as file:
            # Store plain data rather than the class instance so a model saved
            # by running this file as __main__ can be loaded after importing it.
            pickle.dump(payload, file)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "DuffingPhysicalReservoir":
        """Load a reservoir saved by :meth:`save`."""
        source = Path(path)
        if source.suffix != ".pkl":
            source = source.with_suffix(".pkl")
        with source.open("rb") as file:
            payload = pickle.load(file)
        if not isinstance(payload, dict) or payload.get("format") != (
            "duffing-physical-reservoir-v1"
        ):
            raise TypeError(f"{source} does not contain a supported physical reservoir")
        model = cls(**payload["config"])
        for name, value in payload["arrays"].items():
            setattr(model, name, None if value is None else np.asarray(value).copy())
        return model


def generate_duffing_data(
    samples: int,
    dt: float,
    params: dict[str, float],
    initial_state: Sequence[float] = (-0.7, 0.15),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return inputs, next-state targets, and the full Duffing trajectory."""
    if samples < 2 or dt <= 0:
        raise ValueError("samples must be at least 2 and dt must be positive")
    trajectory = np.empty((samples + 1, 2), dtype=float)
    trajectory[0] = np.asarray(initial_state, dtype=float)
    times = dt * np.arange(samples + 1)
    for index in range(samples):
        trajectory[index + 1] = integrate_step(
            trajectory[index], params, dt, t=float(times[index])
        )
    phase = float(params.get("omega", 1.0)) * times[:-1]
    inputs = np.column_stack(
        (trajectory[:-1], np.sin(phase), np.cos(phase))
    )
    return inputs, trajectory[1:], trajectory


def _save_demo_plot(
    path: Path, truth: np.ndarray, prediction: np.ndarray, dt: float
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required when --plot is used") from exc

    time = dt * np.arange(1, len(prediction) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(time, truth[:, 0], label="truth", lw=1.5)
    axes[0].plot(time, prediction[:, 0], label="physical reservoir", lw=1.2)
    axes[0].set(title="Free-running position", xlabel="forecast time", ylabel="x")
    axes[0].legend()
    axes[1].plot(time, truth[:, 1], label="truth", lw=1.5)
    axes[1].plot(time, prediction[:, 1], label="physical reservoir", lw=1.2)
    axes[1].set(title="Free-running velocity", xlabel="forecast time", ylabel="v")
    axes[2].plot(truth[:, 0], truth[:, 1], label="truth", lw=1.3)
    axes[2].plot(prediction[:, 0], prediction[:, 1], label="forecast", lw=1.1)
    axes[2].set(title="Phase portrait", xlabel="x", ylabel="v")
    axes[2].legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--units", type=int, default=48)
    parser.add_argument("--dt", type=float, default=0.04)
    parser.add_argument("--washout", type=int, default=150)
    parser.add_argument("--forecast-steps", type=int, default=500)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    if not 0 < args.train_fraction < 1:
        parser.error("--train-fraction must be between zero and one")
    if args.samples < 10:
        parser.error("--samples must be at least 10")

    params = {
        "alpha": -1.0,
        "beta": 1.0,
        "damping": 0.25,
        "gamma": 0.30,
        "omega": 1.20,
    }
    inputs, targets, _ = generate_duffing_data(args.samples, args.dt, params)
    train_count = int(args.train_fraction * args.samples)
    if not args.washout < train_count:
        parser.error("--washout must be smaller than the training sample count")
    test_count = args.samples - train_count
    horizon = min(args.forecast_steps, test_count)
    if horizon < 1:
        parser.error("the requested split leaves no forecast samples")

    model = DuffingPhysicalReservoir(
        input_size=4,
        output_size=2,
        units=args.units,
        dt=args.dt,
        ridge=args.ridge,
        residual_indices=(0, 1),
    )
    model.fit(inputs[:train_count], targets[:train_count], washout=args.washout)

    synchronization = inputs[max(0, train_count - args.washout) : train_count]
    one_step = model.predict(
        inputs[train_count:], reset=True, warmup_inputs=synchronization
    )
    one_step_rmse = np.sqrt(np.mean((one_step - targets[train_count:]) ** 2, axis=0))

    future_phase = inputs[train_count + 1 : train_count + horizon, 2:]
    free_run = model.forecast(
        inputs[train_count],
        steps=horizon,
        feedback_indices=(0, 1),
        exogenous_future=future_phase,
        warmup_inputs=synchronization,
    )
    truth = targets[train_count : train_count + horizon]
    free_run_rmse = np.sqrt(np.mean((free_run - truth) ** 2, axis=0))

    print(
        "Teacher-forced one-step RMSE: "
        f"x={one_step_rmse[0]:.6f}, v={one_step_rmse[1]:.6f}"
    )
    print(
        f"Free-running RMSE over {horizon} steps: "
        f"x={free_run_rmse[0]:.6f}, v={free_run_rmse[1]:.6f}"
    )
    if args.model_path is not None:
        saved = model.save(args.model_path)
        print(f"Saved physical reservoir to {saved}")
    if args.plot is not None:
        _save_demo_plot(args.plot, truth, free_run, args.dt)
        print(f"Saved forecast plot to {args.plot}")


if __name__ == "__main__":
    main()
