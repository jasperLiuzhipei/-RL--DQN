import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List


class DuelingQNetwork(nn.Module):
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.value = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.storage: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.ptr = 0

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool):
        data = (state, action, reward, next_state, done)
        if len(self.storage) < self.capacity:
            self.storage.append(data)
        else:
            self.storage[self.ptr] = data
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int):
        idxs = np.random.choice(len(self.storage), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.storage[i] for i in idxs])
        return (
            np.stack(states, axis=0),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0),
            np.array(dones, dtype=np.float32),
        )


class FlappyBirdDQNAgent:
    def __init__(
        self,
        env,
        learning_rate: float = 5e-4,
        gamma: float = 0.99,
        initial_epsilon: float = 0.2,
        epsilon_decay: float = 1e-5,
        final_epsilon: float = 0.2,
        buffer_capacity: int = 200_000,
        batch_size: int = 128,
        target_update_interval: int = 1000,
        min_buffer_size: int = 1000,
        grad_clip: float = 10.0,
    ):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = self._infer_input_dim(env)
        self.action_dim = env.action_space.n

        self.q = DuelingQNetwork(self.input_dim, self.action_dim).to(self.device)
        self.q_target = DuelingQNetwork(self.input_dim, self.action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate)
        self.gamma = gamma

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.update_steps = 0
        self.min_buffer_size = min_buffer_size
        self.grad_clip = grad_clip

        self.training_loss: List[float] = []

    def _infer_input_dim(self, env) -> int:
        obs, _ = env.reset()
        vals = []
        for key in sorted(obs.keys()):
            v = obs[key]
            if isinstance(v, (list, np.ndarray)):
                vals.extend(list(np.array(v).flatten()))
            else:
                vals.append(v)
        return len(vals)

    def _process_obs(self, obs: Dict[str, Any]) -> torch.Tensor:
        vals = []
        for key in sorted(obs.keys()):
            v = obs[key]
            if isinstance(v, (list, np.ndarray)):
                vals.extend(list(np.array(v).flatten()))
            else:
                vals.append(v)
        return torch.tensor(vals, dtype=torch.float32, device=self.device)

    def get_action(self, obs: Dict[str, Any]) -> int:
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state = self._process_obs(obs)
            q_values = self.q(state)
            return int(torch.argmax(q_values, dim=-1).item())

    def store_transition(self, obs: Dict[str, Any], action: int, reward: float, done: bool, next_obs: Dict[str, Any]):
        s = self._process_obs(obs).detach().cpu().numpy()
        ns = self._process_obs(next_obs).detach().cpu().numpy()
        self.buffer.add(s, action, reward, ns, done)

    def update(self):
        if len(self.buffer) < max(self.batch_size, self.min_buffer_size):
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)

        q_values = self.q(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = self.q(next_states_t).argmax(dim=1, keepdim=True)
            next_q_target = self.q_target(next_states_t).gather(1, next_actions)
            target_q = rewards_t + (1.0 - dones_t) * self.gamma * next_q_target

        loss = nn.SmoothL1Loss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optimizer.step()
        self.training_loss.append(loss.item())

        self.update_steps += 1
        if self.update_steps % self.target_update_interval == 0:
            self.q_target.load_state_dict(self.q.state_dict())

    def decay_epsilon(self):
        if self.epsilon_decay < 1.0:
            self.epsilon = max(self.final_epsilon, self.epsilon * self.epsilon_decay)
        else:
            self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.q.state_dict(),
            'target_state_dict': self.q_target.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_loss': self.training_loss,
            'epsilon': self.epsilon
        }, filepath)

    @classmethod
    def load(cls, filepath: str, env):
        agent = cls(env)
        if os.path.exists(filepath):
            data = torch.load(filepath, map_location=agent.device)
            agent.q.load_state_dict(data['model_state_dict'])
            agent.q_target.load_state_dict(data['target_state_dict'])
            agent.optimizer.load_state_dict(data['optimizer_state_dict'])
            agent.training_loss = data.get('training_loss', [])
            agent.epsilon = data.get('epsilon', agent.epsilon)
        return agent
