"""Matched recurrent Soft Actor-Critic agents with LSTM or fixed ESN memory.

The implementation follows standard SAC losses and recurrent replay practice:
complete episodes are stored, contiguous sequences are sampled, and a burn-in
prefix reconstructs recurrent state before loss-bearing timesteps.  The LSTM
and ESN variants share the same actor/critic heads and differ only in whether
the recurrent core is learned or fixed.

This is intentionally a research implementation rather than an SB3 extension.
Stable-Baselines3 does not currently provide recurrent SAC.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0
ENCODER_TYPES = ("lstm", "esn")


@dataclass(frozen=True)
class RecurrentSACConfig:
    observation_size: int
    action_size: int
    action_limit: float
    encoder_type: str = "lstm"
    recurrent_units: int = 100
    actor_hidden_sizes: tuple[int, ...] = (128,)
    critic_hidden_sizes: tuple[int, ...] = (128, 128)
    spectral_radius: float = 0.9
    leaking_rate: float = 0.3
    input_scaling: float = 0.5
    actor_learning_rate: float = 3e-4
    critic_learning_rate: float = 3e-4
    temperature_learning_rate: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    initial_temperature: float = 0.2
    target_entropy: float | None = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.encoder_type not in ENCODER_TYPES:
            raise ValueError(f"encoder_type must be one of {ENCODER_TYPES}")
        if self.observation_size < 1 or self.action_size < 1:
            raise ValueError("observation_size and action_size must be positive")
        if self.action_limit <= 0 or self.recurrent_units < 1:
            raise ValueError("action_limit and recurrent_units must be positive")
        if not 0.0 < self.leaking_rate <= 1.0:
            raise ValueError("leaking_rate must be in (0, 1]")
        if self.spectral_radius < 0 or self.input_scaling <= 0:
            raise ValueError(
                "spectral_radius must be nonnegative and input_scaling positive"
            )
        if not 0.0 <= self.gamma <= 1.0 or not 0.0 < self.tau <= 1.0:
            raise ValueError("gamma must be in [0, 1] and tau in (0, 1]")
        if self.initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")


@dataclass
class Episode:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    terminals: np.ndarray

    @property
    def length(self) -> int:
        return int(self.actions.shape[0])


class EpisodeSequenceReplayBuffer:
    """Replay complete episodes and sample boundary-safe contiguous sequences."""

    def __init__(self, capacity: int = 100_000, seed: int = 0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.episodes: deque[Episode] = deque()
        self.transition_count = 0

    def __len__(self) -> int:
        return self.transition_count

    def add_episode(
        self,
        observations: Sequence[np.ndarray],
        actions: Sequence[np.ndarray],
        rewards: Sequence[float],
        terminals: Sequence[bool],
    ) -> None:
        observation_array = np.asarray(observations, dtype=np.float32)
        action_array = np.asarray(actions, dtype=np.float32)
        reward_array = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        terminal_array = np.asarray(terminals, dtype=np.float32).reshape(-1, 1)
        length = action_array.shape[0]
        if length < 1:
            raise ValueError("cannot add an empty episode")
        if observation_array.shape[0] != length + 1:
            raise ValueError("an episode needs one more observation than actions")
        if reward_array.shape[0] != length or terminal_array.shape[0] != length:
            raise ValueError("actions, rewards, and terminals must have equal length")
        episode = Episode(
            observations=observation_array,
            actions=action_array,
            rewards=reward_array,
            terminals=terminal_array,
        )
        self.episodes.append(episode)
        self.transition_count += length
        while self.transition_count > self.capacity and len(self.episodes) > 1:
            removed = self.episodes.popleft()
            self.transition_count -= removed.length

    def can_sample(self, burn_in: int, sequence_length: int) -> bool:
        required = int(burn_in) + int(sequence_length)
        return any(episode.length >= required for episode in self.episodes)

    def sample(
        self,
        batch_size: int,
        burn_in: int,
        sequence_length: int,
    ) -> dict[str, np.ndarray]:
        if batch_size < 1 or burn_in < 0 or sequence_length < 1:
            raise ValueError("invalid batch, burn-in, or sequence length")
        total = burn_in + sequence_length
        eligible = [episode for episode in self.episodes if episode.length >= total]
        if not eligible:
            raise ValueError("no stored episode is long enough for this sequence")

        observations = []
        actions = []
        rewards = []
        terminals = []
        # Sampling episodes proportional to their number of valid sequence
        # starts avoids overweighting short episodes.
        counts = np.asarray(
            [episode.length - total + 1 for episode in eligible], dtype=float
        )
        probabilities = counts / counts.sum()
        for _ in range(batch_size):
            episode_index = int(self.rng.choice(len(eligible), p=probabilities))
            episode = eligible[episode_index]
            start = int(self.rng.integers(0, episode.length - total + 1))
            stop = start + total
            observations.append(episode.observations[start : stop + 1])
            actions.append(episode.actions[start:stop])
            rewards.append(episode.rewards[start:stop])
            terminals.append(episode.terminals[start:stop])
        return {
            "observations": np.asarray(observations, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.float32),
            "rewards": np.asarray(rewards, dtype=np.float32),
            "terminals": np.asarray(terminals, dtype=np.float32),
        }


class LSTMEncoder(nn.Module):
    """Trainable recurrent history encoder."""

    def __init__(self, input_size: int, units: int) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.units = int(units)
        self.recurrent = nn.LSTM(self.input_size, self.units, batch_first=True)

    def initial_state(self, batch_size: int, device: torch.device):
        zeros = torch.zeros(1, batch_size, self.units, device=device)
        return zeros, zeros.clone()

    def forward(self, values: torch.Tensor, state=None):
        return self.recurrent(values, state)


class ESNEncoder(nn.Module):
    """Fixed echo-state recurrent core implemented as torch buffers."""

    def __init__(
        self,
        input_size: int,
        units: int,
        spectral_radius: float,
        leaking_rate: float,
        input_scaling: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.input_size = int(input_size)
        self.units = int(units)
        self.leaking_rate = float(leaking_rate)
        rng = np.random.default_rng(seed)
        recurrent = rng.normal(0.0, 1.0 / np.sqrt(units), size=(units, units))
        radius = float(np.max(np.abs(np.linalg.eigvals(recurrent))))
        if radius > 0:
            recurrent *= float(spectral_radius) / radius
        inputs = rng.uniform(
            -float(input_scaling),
            float(input_scaling),
            size=(units, input_size),
        )
        bias = rng.uniform(-float(input_scaling), float(input_scaling), size=units)
        self.register_buffer("recurrent_weights", torch.as_tensor(recurrent, dtype=torch.float32))
        self.register_buffer("input_weights", torch.as_tensor(inputs, dtype=torch.float32))
        self.register_buffer("bias", torch.as_tensor(bias, dtype=torch.float32))

    def initial_state(self, batch_size: int, device: torch.device):
        return torch.zeros(batch_size, self.units, device=device)

    def forward(self, values: torch.Tensor, state=None):
        if state is None:
            state = self.initial_state(values.shape[0], values.device)
        outputs = []
        for index in range(values.shape[1]):
            candidate = torch.tanh(
                F.linear(state, self.recurrent_weights)
                + F.linear(values[:, index], self.input_weights, self.bias)
            )
            state = (1.0 - self.leaking_rate) * state + self.leaking_rate * candidate
            outputs.append(state)
        if outputs:
            sequence = torch.stack(outputs, dim=1)
        else:
            sequence = values.new_zeros((values.shape[0], 0, self.units))
        return sequence, state


class GaussianActor(nn.Module):
    def __init__(
        self,
        feature_size: int,
        action_size: int,
        action_limit: float,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current = feature_size
        for hidden in hidden_sizes:
            layers.extend((nn.Linear(current, hidden), nn.ReLU()))
            current = hidden
        self.backbone = nn.Sequential(*layers)
        self.mean = nn.Linear(current, action_size)
        self.log_std = nn.Linear(current, action_size)
        self.register_buffer(
            "action_scale", torch.full((action_size,), float(action_limit))
        )

    def parameters_for(self, features: torch.Tensor):
        hidden = self.backbone(features)
        mean = self.mean(hidden)
        log_std = self.log_std(hidden).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, features: torch.Tensor):
        mean, log_std = self.parameters_for(features)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        unsquashed = distribution.rsample()
        normalized = torch.tanh(unsquashed)
        action = normalized * self.action_scale
        log_probability = distribution.log_prob(unsquashed)
        log_probability -= torch.log(1.0 - normalized.pow(2) + 1e-6)
        log_probability = log_probability.sum(dim=-1, keepdim=True)
        deterministic = torch.tanh(mean) * self.action_scale
        return action, log_probability, deterministic


class QNetwork(nn.Module):
    def __init__(
        self,
        feature_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...],
    ) -> None:
        super().__init__()
        sizes = (feature_size + action_size, *hidden_sizes, 1)
        layers: list[nn.Module] = []
        for input_size, output_size in zip(sizes[:-2], sizes[1:-1]):
            layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, normalized_actions: torch.Tensor):
        return self.network(torch.cat((features, normalized_actions), dim=-1))


def _detach_state(state):
    if isinstance(state, tuple):
        return tuple(value.detach() for value in state)
    return state.detach()


class RecurrentSACAgent:
    """SAC with matched recurrent actor and twin recurrent critic encoders."""

    def __init__(self, config: RecurrentSACConfig, device: str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        def make_encoder(seed_offset: int) -> nn.Module:
            if config.encoder_type == "lstm":
                return LSTMEncoder(
                    config.observation_size, config.recurrent_units
                ).to(self.device)
            return ESNEncoder(
                config.observation_size,
                config.recurrent_units,
                config.spectral_radius,
                config.leaking_rate,
                config.input_scaling,
                config.seed + seed_offset,
            ).to(self.device)

        # Recurrent off-policy baselines conventionally give the actor and each
        # critic their own recurrent function approximator.  The ESN comparison
        # mirrors that layout exactly, replacing learned cores with fixed ones.
        self.encoder = make_encoder(0)
        self.critic_encoder_1 = make_encoder(10_007)
        self.critic_encoder_2 = make_encoder(20_011)
        self.target_critic_encoder_1 = copy.deepcopy(self.critic_encoder_1).to(
            self.device
        )
        self.target_critic_encoder_2 = copy.deepcopy(self.critic_encoder_2).to(
            self.device
        )
        for module in (
            self.target_critic_encoder_1,
            self.target_critic_encoder_2,
        ):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        self.feature_size = config.observation_size + config.recurrent_units
        self.actor = GaussianActor(
            self.feature_size,
            config.action_size,
            config.action_limit,
            config.actor_hidden_sizes,
        ).to(self.device)
        self.critic_1 = QNetwork(
            self.feature_size,
            config.action_size,
            config.critic_hidden_sizes,
        ).to(self.device)
        self.critic_2 = QNetwork(
            self.feature_size,
            config.action_size,
            config.critic_hidden_sizes,
        ).to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
        for module in (self.target_critic_1, self.target_critic_2):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

        actor_encoder_parameters = [
            parameter
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        ]
        critic_encoder_parameters = [
            parameter
            for module in (self.critic_encoder_1, self.critic_encoder_2)
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        self.actor_optimizer = torch.optim.Adam(
            [*actor_encoder_parameters, *self.actor.parameters()],
            lr=config.actor_learning_rate,
        )
        self.critic_optimizer = torch.optim.Adam(
            [
                *critic_encoder_parameters,
                *self.critic_1.parameters(),
                *self.critic_2.parameters(),
            ],
            lr=config.critic_learning_rate,
        )
        self.log_temperature = torch.tensor(
            np.log(config.initial_temperature),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=config.temperature_learning_rate
        )
        self.target_entropy = float(
            -config.action_size
            if config.target_entropy is None
            else config.target_entropy
        )
        self._online_state = None

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def reset(self) -> None:
        self._online_state = None

    def _normalized(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(observations)

    def _encode_with_burn(
        self,
        encoder: nn.Module,
        observations: torch.Tensor,
        burn_in: int,
    ) -> torch.Tensor:
        normalized = self._normalized(observations)
        state = None
        if burn_in:
            with torch.no_grad():
                _, state = encoder(normalized[:, :burn_in])
            state = _detach_state(state)
        recurrent, _ = encoder(normalized[:, burn_in:], state)
        return torch.cat((observations[:, burn_in:], recurrent), dim=-1)

    def _online_features(self, observation) -> torch.Tensor:
        raw = np.asarray(observation, dtype=np.float32).reshape(-1)
        if raw.shape != (self.config.observation_size,):
            raise ValueError(
                f"expected observation shape ({self.config.observation_size},), "
                f"got {raw.shape}"
            )
        value = torch.as_tensor(raw, device=self.device).view(1, 1, -1)
        normalized = self._normalized(value)
        with torch.no_grad():
            recurrent, self._online_state = self.encoder(
                normalized, self._online_state
            )
            return torch.cat((value, recurrent), dim=-1).reshape(1, -1)

    def predict(self, observation, deterministic: bool = True):
        features = self._online_features(observation)
        with torch.no_grad():
            sampled, _, mean_action = self.actor.sample(features)
            action = mean_action if deterministic else sampled
        return action.squeeze(0).cpu().numpy().astype(np.float32), None

    def update(
        self,
        batch: dict[str, np.ndarray],
        burn_in: int,
    ) -> dict[str, float]:
        observations = torch.as_tensor(
            batch["observations"], dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            batch["actions"], dtype=torch.float32, device=self.device
        )[:, burn_in:]
        rewards = torch.as_tensor(
            batch["rewards"], dtype=torch.float32, device=self.device
        )[:, burn_in:]
        terminals = torch.as_tensor(
            batch["terminals"], dtype=torch.float32, device=self.device
        )[:, burn_in:]
        normalized_actions = actions / self.config.action_limit

        critic_features_1 = self._encode_with_burn(
            self.critic_encoder_1, observations, burn_in
        )
        critic_features_2 = self._encode_with_burn(
            self.critic_encoder_2, observations, burn_in
        )
        with torch.no_grad():
            target_actor_features = self._encode_with_burn(
                self.encoder, observations, burn_in
            )
            next_actions, next_log_probabilities, _ = self.actor.sample(
                target_actor_features[:, 1:]
            )
            normalized_next_actions = next_actions / self.config.action_limit
            target_features_1 = self._encode_with_burn(
                self.target_critic_encoder_1, observations, burn_in
            )[:, 1:]
            target_features_2 = self._encode_with_burn(
                self.target_critic_encoder_2, observations, burn_in
            )[:, 1:]
            next_q = torch.minimum(
                self.target_critic_1(target_features_1, normalized_next_actions),
                self.target_critic_2(target_features_2, normalized_next_actions),
            )
            target = rewards + self.config.gamma * (1.0 - terminals) * (
                next_q - self.temperature.detach() * next_log_probabilities
            )

        q_1 = self.critic_1(critic_features_1[:, :-1], normalized_actions)
        q_2 = self.critic_2(critic_features_2[:, :-1], normalized_actions)
        critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [
                *self.critic_encoder_1.parameters(),
                *self.critic_encoder_2.parameters(),
                *self.critic_1.parameters(),
                *self.critic_2.parameters(),
            ],
            max_norm=10.0,
        )
        self.critic_optimizer.step()

        # Reconstruct critic histories after their recurrent parameters change.
        # Critic features are constants for the actor step, while gradients pass
        # through the actor's own recurrent encoder and Gaussian policy head.
        with torch.no_grad():
            actor_critic_features_1 = self._encode_with_burn(
                self.critic_encoder_1, observations, burn_in
            )[:, :-1]
            actor_critic_features_2 = self._encode_with_burn(
                self.critic_encoder_2, observations, burn_in
            )[:, :-1]
        actor_features = self._encode_with_burn(
            self.encoder, observations, burn_in
        )[:, :-1]
        new_actions, log_probabilities, _ = self.actor.sample(actor_features)
        normalized_new_actions = new_actions / self.config.action_limit
        for critic in (self.critic_1, self.critic_2):
            critic.requires_grad_(False)
        actor_q = torch.minimum(
            self.critic_1(actor_critic_features_1, normalized_new_actions),
            self.critic_2(actor_critic_features_2, normalized_new_actions),
        )
        actor_loss = (
            self.temperature.detach() * log_probabilities - actor_q
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()
        for critic in (self.critic_1, self.critic_2):
            critic.requires_grad_(True)

        temperature_loss = -(
            self.log_temperature * (log_probabilities.detach() + self.target_entropy)
        ).mean()
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()

        self._soft_update_targets()
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "temperature_loss": float(temperature_loss.detach().cpu()),
            "temperature": float(self.temperature.detach().cpu()),
        }

    def _soft_update_targets(self) -> None:
        tau = self.config.tau
        with torch.no_grad():
            for target_encoder, source_encoder in (
                (self.target_critic_encoder_1, self.critic_encoder_1),
                (self.target_critic_encoder_2, self.critic_encoder_2),
            ):
                for target, source in zip(
                    target_encoder.parameters(), source_encoder.parameters()
                ):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)
            for target_network, source_network in (
                (self.target_critic_1, self.critic_1),
                (self.target_critic_2, self.critic_2),
            ):
                for target, source in zip(
                    target_network.parameters(), source_network.parameters()
                ):
                    target.mul_(1.0 - tau).add_(source, alpha=tau)

    def parameter_counts(self) -> dict[str, int]:
        modules = {
            "encoders": nn.ModuleList(
                (self.encoder, self.critic_encoder_1, self.critic_encoder_2)
            ),
            "actor": self.actor,
            "critics": nn.ModuleList((self.critic_1, self.critic_2)),
        }
        counts = {
            f"{name}_parameters": sum(
                parameter.numel() for parameter in module.parameters()
            )
            for name, module in modules.items()
        }
        counts["trainable_parameters"] = sum(
            parameter.numel()
            for module in modules.values()
            for parameter in module.parameters()
            if parameter.requires_grad
        ) + 1
        counts["trainable_actor_parameters"] = counts["actor_parameters"] + sum(
            parameter.numel()
            for parameter in self.encoder.parameters()
            if parameter.requires_grad
        )
        counts["fixed_encoder_parameters"] = sum(
            value.numel()
            for module in (
                self.encoder,
                self.critic_encoder_1,
                self.critic_encoder_2,
            )
            for value in module.buffers()
        )
        return counts

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.suffix != ".pt":
            destination = destination.with_suffix(".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "config": asdict(self.config),
                "encoder": self.encoder.state_dict(),
                "critic_encoder_1": self.critic_encoder_1.state_dict(),
                "critic_encoder_2": self.critic_encoder_2.state_dict(),
                "target_critic_encoder_1": self.target_critic_encoder_1.state_dict(),
                "target_critic_encoder_2": self.target_critic_encoder_2.state_dict(),
                "actor": self.actor.state_dict(),
                "critic_1": self.critic_1.state_dict(),
                "critic_2": self.critic_2.state_dict(),
                "target_critic_1": self.target_critic_1.state_dict(),
                "target_critic_2": self.target_critic_2.state_dict(),
                "log_temperature": self.log_temperature.detach().cpu(),
                "parameter_counts": self.parameter_counts(),
            },
            destination,
        )
        return destination

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "RecurrentSACAgent":
        source = Path(path)
        checkpoint = torch.load(source, map_location=device, weights_only=False)
        config_values: dict[str, Any] = checkpoint["config"]
        for name in ("actor_hidden_sizes", "critic_hidden_sizes"):
            config_values[name] = tuple(config_values[name])
        agent = cls(RecurrentSACConfig(**config_values), device=device)
        for name in (
            "encoder",
            "critic_encoder_1",
            "critic_encoder_2",
            "target_critic_encoder_1",
            "target_critic_encoder_2",
            "actor",
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
        ):
            getattr(agent, name).load_state_dict(checkpoint[name])
        with torch.no_grad():
            agent.log_temperature.copy_(checkpoint["log_temperature"].to(device))
        agent.reset()
        return agent
