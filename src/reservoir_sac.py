"""Soft Actor-Critic with a fixed recurrent reservoir feature encoder."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0
MEMORY_MODES = ("persistent", "reset_each_step")


class ReplayBuffer:
    """Fixed-size replay storage for already-computed reservoir features."""

    def __init__(
        self,
        capacity: int,
        feature_size: int,
        action_size: int,
        seed: int = 0,
    ) -> None:
        if capacity < 1 or feature_size < 1 or action_size < 1:
            raise ValueError("capacity and dimensions must be positive")
        self.capacity = int(capacity)
        self.feature_size = int(feature_size)
        self.action_size = int(action_size)
        self.features = np.empty((capacity, feature_size), dtype=np.float32)
        self.actions = np.empty((capacity, action_size), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.next_features = np.empty((capacity, feature_size), dtype=np.float32)
        self.terminals = np.empty((capacity, 1), dtype=np.float32)
        self.position = 0
        self.size = 0
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        features,
        action,
        reward: float,
        next_features,
        terminal: bool,
    ) -> None:
        index = self.position
        self.features[index] = np.asarray(features, dtype=np.float32)
        self.actions[index] = np.asarray(action, dtype=np.float32)
        self.rewards[index, 0] = float(reward)
        self.next_features[index] = np.asarray(next_features, dtype=np.float32)
        self.terminals[index, 0] = float(terminal)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> dict[str, np.ndarray]:
        if batch_size < 1 or batch_size > self.size:
            raise ValueError("batch_size must be between 1 and the current buffer size")
        indices = self.rng.integers(0, self.size, size=batch_size)
        return {
            "features": self.features[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_features": self.next_features[indices],
            "terminals": self.terminals[indices],
        }


class GaussianReadoutActor(nn.Module):
    """Linear Gaussian SAC actor operating on fixed reservoir features."""

    def __init__(self, feature_size: int, action_size: int, action_limit: float) -> None:
        super().__init__()
        self.mean = nn.Linear(feature_size, action_size)
        self.log_std = nn.Linear(feature_size, action_size)
        self.register_buffer(
            "action_scale", torch.full((action_size,), float(action_limit))
        )
        nn.init.normal_(self.mean.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.mean.bias)
        nn.init.zeros_(self.log_std.weight)
        nn.init.constant_(self.log_std.bias, -1.0)

    def distribution_parameters(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = self.mean(features)
        log_std = self.log_std(features).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(
        self, features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self.distribution_parameters(features)
        std = log_std.exp()
        distribution = torch.distributions.Normal(mean, std)
        unsquashed = distribution.rsample()
        normalized_action = torch.tanh(unsquashed)
        action = normalized_action * self.action_scale
        # Entropy is measured in normalized action coordinates. The omitted
        # constant scale correction does not affect the actor gradient and
        # keeps the usual SAC target entropy meaningful for small action limits.
        log_probability = distribution.log_prob(unsquashed)
        log_probability -= torch.log(1.0 - normalized_action.pow(2) + 1e-6)
        log_probability = log_probability.sum(dim=-1, keepdim=True)
        deterministic_action = torch.tanh(mean) * self.action_scale
        return action, log_probability, deterministic_action


class QNetwork(nn.Module):
    """Small nonlinear critic for one reservoir-feature/action pair."""

    def __init__(
        self,
        feature_size: int,
        action_size: int,
        hidden_sizes: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        sizes = (feature_size + action_size, *hidden_sizes, 1)
        layers: list[nn.Module] = []
        for input_size, output_size in zip(sizes[:-2], sizes[1:-1]):
            layers.extend((nn.Linear(input_size, output_size), nn.ReLU()))
        layers.append(nn.Linear(sizes[-2], sizes[-1]))
        self.network = nn.Sequential(*layers)

    def forward(
        self, features: torch.Tensor, normalized_actions: torch.Tensor
    ) -> torch.Tensor:
        return self.network(torch.cat((features, normalized_actions), dim=-1))


class ReservoirSACAgent:
    """Fixed recurrent encoder with a SAC-trained actor and twin critics."""

    def __init__(
        self,
        encoder,
        action_size: int = 2,
        action_limit: float = 0.05,
        hidden_sizes: tuple[int, ...] = (128, 128),
        actor_learning_rate: float = 3e-4,
        critic_learning_rate: float = 3e-4,
        temperature_learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        initial_temperature: float = 0.2,
        target_entropy: float | None = None,
        memory_mode: str = "persistent",
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        if action_size < 1 or action_limit <= 0:
            raise ValueError("action_size and action_limit must be positive")
        if not 0.0 <= gamma <= 1.0 or not 0.0 < tau <= 1.0:
            raise ValueError("gamma must be in [0, 1] and tau in (0, 1]")
        if initial_temperature <= 0:
            raise ValueError("initial_temperature must be positive")
        if memory_mode not in MEMORY_MODES:
            raise ValueError(
                f"memory_mode must be one of {MEMORY_MODES}, got {memory_mode!r}"
            )
        if not hasattr(encoder, "features") or not hasattr(encoder, "feature_size"):
            raise TypeError("encoder must provide features(), reset(), and feature_size")

        torch.manual_seed(seed)
        self.encoder = encoder
        self.feature_size = int(encoder.feature_size)
        self.action_size = int(action_size)
        self.action_limit = float(action_limit)
        self.hidden_sizes = tuple(int(size) for size in hidden_sizes)
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.target_entropy = float(
            -action_size if target_entropy is None else target_entropy
        )
        self.memory_mode = memory_mode
        self.seed = int(seed)
        self.device = torch.device(device)

        network_args = (self.feature_size, self.action_size, self.hidden_sizes)
        self.actor = GaussianReadoutActor(
            self.feature_size, self.action_size, self.action_limit
        ).to(self.device)
        self.critic_1 = QNetwork(*network_args).to(self.device)
        self.critic_2 = QNetwork(*network_args).to(self.device)
        self.target_critic_1 = copy.deepcopy(self.critic_1).to(self.device)
        self.target_critic_2 = copy.deepcopy(self.critic_2).to(self.device)
        for target in (self.target_critic_1, self.target_critic_2):
            target.requires_grad_(False)

        self.log_temperature = nn.Parameter(
            torch.tensor(np.log(initial_temperature), dtype=torch.float32, device=self.device)
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_learning_rate
        )
        critic_parameters = list(self.critic_1.parameters()) + list(
            self.critic_2.parameters()
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_parameters, lr=critic_learning_rate
        )
        self.temperature_optimizer = torch.optim.Adam(
            [self.log_temperature], lr=temperature_learning_rate
        )
        self.learning_rates = {
            "actor": float(actor_learning_rate),
            "critic": float(critic_learning_rate),
            "temperature": float(temperature_learning_rate),
        }
        self.update_steps = 0

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_temperature.exp()

    def reset(self) -> None:
        self.encoder.reset()

    def encode_with(self, encoder, observation) -> np.ndarray:
        """Encode one observation using this agent's memory ablation mode.

        ``persistent`` carries the encoder state across control steps.
        ``reset_each_step`` clears it immediately before every observation,
        retaining the same nonlinear feature map but removing temporal memory.
        """
        if self.memory_mode == "reset_each_step":
            encoder.reset()
        return np.asarray(encoder.features(observation), dtype=np.float32)

    def encode(self, observation) -> np.ndarray:
        return self.encode_with(self.encoder, observation)

    def act_features(self, features, deterministic: bool = False) -> np.ndarray:
        tensor = torch.as_tensor(
            np.asarray(features, dtype=np.float32), device=self.device
        ).unsqueeze(0)
        with torch.no_grad():
            sampled, _, mean_action = self.actor.sample(tensor)
        action = mean_action if deterministic else sampled
        return action.squeeze(0).cpu().numpy().astype(np.float32)

    def predict(self, observation, deterministic: bool = True):
        """SB3-compatible prediction interface used by existing evaluators."""
        features = self.encode(observation)
        return self.act_features(features, deterministic=deterministic), None

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        tensors = {
            name: torch.as_tensor(values, dtype=torch.float32, device=self.device)
            for name, values in batch.items()
        }
        features = tensors["features"]
        actions = tensors["actions"]
        rewards = tensors["rewards"]
        next_features = tensors["next_features"]
        terminals = tensors["terminals"]
        normalized_actions = actions / self.action_limit

        with torch.no_grad():
            next_actions, next_log_probabilities, _ = self.actor.sample(next_features)
            normalized_next_actions = next_actions / self.action_limit
            next_q = torch.minimum(
                self.target_critic_1(next_features, normalized_next_actions),
                self.target_critic_2(next_features, normalized_next_actions),
            )
            target = rewards + self.gamma * (1.0 - terminals) * (
                next_q - self.temperature.detach() * next_log_probabilities
            )

        q_1 = self.critic_1(features, normalized_actions)
        q_2 = self.critic_2(features, normalized_actions)
        critic_loss = F.mse_loss(q_1, target) + F.mse_loss(q_2, target)
        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_optimizer.step()

        new_actions, log_probabilities, _ = self.actor.sample(features)
        normalized_new_actions = new_actions / self.action_limit
        actor_q = torch.minimum(
            self.critic_1(features, normalized_new_actions),
            self.critic_2(features, normalized_new_actions),
        )
        actor_loss = (
            self.temperature.detach() * log_probabilities - actor_q
        ).mean()
        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        temperature_loss = -(
            self.log_temperature *
            (log_probabilities + self.target_entropy).detach()
        ).mean()
        self.temperature_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        self.temperature_optimizer.step()

        with torch.no_grad():
            for target_parameter, parameter in zip(
                self.target_critic_1.parameters(), self.critic_1.parameters()
            ):
                target_parameter.lerp_(parameter, self.tau)
            for target_parameter, parameter in zip(
                self.target_critic_2.parameters(), self.critic_2.parameters()
            ):
                target_parameter.lerp_(parameter, self.tau)

        self.update_steps += 1
        return {
            "critic_loss": float(critic_loss.detach().cpu()),
            "actor_loss": float(actor_loss.detach().cpu()),
            "temperature_loss": float(temperature_loss.detach().cpu()),
            "temperature": float(self.temperature.detach().cpu()),
            "mean_target_q": float(target.mean().detach().cpu()),
        }

    def _checkpoint(self) -> dict[str, Any]:
        return {
            "version": 2,
            "encoder": self.encoder,
            "feature_size": self.feature_size,
            "action_size": self.action_size,
            "action_limit": self.action_limit,
            "hidden_sizes": self.hidden_sizes,
            "gamma": self.gamma,
            "tau": self.tau,
            "target_entropy": self.target_entropy,
            "memory_mode": self.memory_mode,
            "seed": self.seed,
            "learning_rates": self.learning_rates,
            "actor": self.actor.state_dict(),
            "critic_1": self.critic_1.state_dict(),
            "critic_2": self.critic_2.state_dict(),
            "target_critic_1": self.target_critic_1.state_dict(),
            "target_critic_2": self.target_critic_2.state_dict(),
            "log_temperature": self.log_temperature.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "temperature_optimizer": self.temperature_optimizer.state_dict(),
            "update_steps": self.update_steps,
        }

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        if destination.suffix != ".pt":
            destination = destination.with_suffix(".pt")
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._checkpoint(), destination)
        return destination

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "ReservoirSACAgent":
        source = Path(path)
        if source.suffix != ".pt":
            source = source.with_suffix(".pt")
        checkpoint = torch.load(source, map_location=device, weights_only=False)
        rates = checkpoint["learning_rates"]
        agent = cls(
            encoder=checkpoint["encoder"],
            action_size=checkpoint["action_size"],
            action_limit=checkpoint["action_limit"],
            hidden_sizes=tuple(checkpoint["hidden_sizes"]),
            actor_learning_rate=rates["actor"],
            critic_learning_rate=rates["critic"],
            temperature_learning_rate=rates["temperature"],
            gamma=checkpoint["gamma"],
            tau=checkpoint["tau"],
            initial_temperature=float(checkpoint["log_temperature"].exp()),
            target_entropy=checkpoint["target_entropy"],
            memory_mode=checkpoint.get("memory_mode", "persistent"),
            seed=checkpoint["seed"],
            device=device,
        )
        for name in (
            "actor",
            "critic_1",
            "critic_2",
            "target_critic_1",
            "target_critic_2",
        ):
            getattr(agent, name).load_state_dict(checkpoint[name])
        with torch.no_grad():
            agent.log_temperature.copy_(checkpoint["log_temperature"].to(agent.device))
        agent.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        agent.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        agent.temperature_optimizer.load_state_dict(checkpoint["temperature_optimizer"])
        agent.update_steps = int(checkpoint["update_steps"])
        return agent
