"""DQN prey agent (same network shape as Deer-Hunt so old weights load)."""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def masked_argmax(q_values: np.ndarray, valid_actions: list[int]) -> int:
    masked = np.full_like(q_values, -np.inf)
    masked[valid_actions] = q_values[valid_actions]
    return int(np.argmax(masked))


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_size: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10_000,
        batch_size: int = 64,
        device: str | None = None,
    ):
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_size)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.q_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.target_network = DQN(state_size, hidden_size, action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.sync_target()

    def sync_target(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

    def remember(self, *transition):
        self.memory.append(transition)

    def q_values(self, state: np.ndarray) -> np.ndarray:
        self.q_network.eval()
        with torch.no_grad():
            t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q = self.q_network(t).cpu().numpy()[0]
        self.q_network.train()
        return q

    def act(self, state: np.ndarray, valid_actions: list[int], greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.choice(valid_actions)
        return masked_argmax(self.q_values(state), valid_actions)

    def replay(self) -> float | None:
        if len(self.memory) < self.batch_size:
            return None
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        to_t = lambda a, dt: torch.as_tensor(a, dtype=dt, device=self.device)
        states, next_states = to_t(states, torch.float32), to_t(next_states, torch.float32)
        actions, rewards = to_t(actions, torch.long), to_t(rewards, torch.float32)
        dones = to_t(dones, torch.bool)

        q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
        target = rewards + self.gamma * next_q * (~dones)

        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 10.0)
        self.optimizer.step()
        return float(loss.item())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
