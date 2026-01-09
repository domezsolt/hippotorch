from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class DQNConfig:
    state_dim: int
    action_dim: int
    hidden_dim: int = 128
    gamma: float = 0.99
    lr: float = 1e-3
    tau: float = 0.01  # soft target update
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50_000
    device: str = "cpu"


class QNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class DQNAgent:
    """Minimal DQN with target network and epsilon-greedy policy.

    Expects replay batches with keys: states, actions, rewards, next_states, dones.
    """

    def __init__(self, cfg: DQNConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.q = QNet(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(self.device)
        self.q_target = QNet(cfg.state_dim, cfg.action_dim, cfg.hidden_dim).to(
            self.device
        )
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = torch.optim.Adam(self.q.parameters(), lr=cfg.lr)
        self._steps = 0

    @torch.no_grad()
    def act(self, state, explore: bool = True) -> int:
        state_t = (
            state
            if isinstance(state, Tensor)
            else torch.as_tensor(state, dtype=torch.float32)
        )
        state_t = (
            state_t.to(self.device).unsqueeze(0) if state_t.dim() == 1 else state_t
        )
        eps = self._epsilon() if explore else 0.0
        if torch.rand(()) < eps:
            return int(torch.randint(0, self.cfg.action_dim, (1,)).item())
        q_vals = self.q(state_t)
        return int(q_vals.argmax(dim=-1).item())

    def update(self, batch: dict) -> dict:
        self._steps += 1
        states = batch["states"].to(self.device).float()
        actions = batch["actions"].to(self.device)
        if actions.dim() > 1:
            actions = actions.squeeze(-1)
        actions = actions.long().clamp_min(0)
        rewards = batch["rewards"].to(self.device).float()
        next_states = batch["next_states"].to(self.device).float()
        dones = batch["dones"].to(self.device).float()

        q_pred = self.q(states).gather(1, actions.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            q_next = self.q_target(next_states).max(dim=-1).values
            target = rewards + self.cfg.gamma * (1.0 - dones) * q_next
        loss = nn.functional.mse_loss(q_pred, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self._soft_update()

        return {"loss": float(loss.item()), "epsilon": self._epsilon()}

    def _soft_update(self) -> None:
        with torch.no_grad():
            for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
                p_t.data.mul_(1.0 - self.cfg.tau).add_(p.data, alpha=self.cfg.tau)

    def _epsilon(self) -> float:
        progress = min(1.0, self._steps / max(1, self.cfg.epsilon_decay_steps))
        return float(
            self.cfg.epsilon_start
            + progress * (self.cfg.epsilon_end - self.cfg.epsilon_start)
        )


__all__ = ["DQNAgent", "DQNConfig"]
