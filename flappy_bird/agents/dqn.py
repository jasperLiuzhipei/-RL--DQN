import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN with additional action probability output for behavior cloning.
    Uses Tanh activation (like PPO) for better gradient flow.
    """
    def __init__(self, input_dim: int, action_dim: int):
        super().__init__()
        # Shared feature extractor - use Tanh like PPO for stability
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
        )
        # Value stream
        self.value = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # Advantage stream
        self.advantage = nn.Sequential(
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim)
        )
        # Policy head for behavior cloning (action probabilities)
        self.policy = nn.Sequential(
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f = self.feature(x)
        v = self.value(f)
        a = self.advantage(f)
        return v + a - a.mean(dim=1, keepdim=True)
    
    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities for behavior cloning loss."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f = self.feature(x)
        return self.policy(f)


class ReplayBuffer:
    """Prioritized Experience Replay (PER) buffer with proportional sampling."""
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.storage: List[Tuple[np.ndarray, int, float, np.ndarray, bool, int]] = []
        self.priorities: List[float] = []
        self.ptr = 0
        self.alpha = alpha
        self.beta = beta
        self.eps = 1e-6

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool, steps: int):
        data = (state, action, reward, next_state, done, steps)
        max_prio = max(self.priorities) if self.priorities else 1.0
        if len(self.storage) < self.capacity:
            self.storage.append(data)
            self.priorities.append(max_prio)
        else:
            self.storage[self.ptr] = data
            self.priorities[self.ptr] = max_prio
        self.ptr = (self.ptr + 1) % self.capacity

    def sample(self, batch_size: int):
        prios = np.array(self.priorities, dtype=np.float32)
        probs = prios ** self.alpha
        probs /= probs.sum()
        idxs = np.random.choice(len(self.storage), batch_size, replace=False, p=probs)
        weights = (len(self.storage) * probs[idxs]) ** (-self.beta)
        weights /= weights.max() + self.eps
        states, actions, rewards, next_states, dones, steps = zip(*[self.storage[i] for i in idxs])
        return (
            np.stack(states, axis=0),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states, axis=0),
            np.array(dones, dtype=np.float32),
            np.array(steps, dtype=np.int64),
            idxs,
            np.array(weights, dtype=np.float32),
        )

    def update_priorities(self, idxs: np.ndarray, td_errors: np.ndarray):
        for i, e in zip(idxs, td_errors):
            self.priorities[int(i)] = float(abs(e)) + self.eps


class FlappyBirdDQNAgent:
    """
    DQN Agent with forced heuristic guidance and behavior cloning loss.
    Mimics the successful training strategy from PPO.
    """
    def __init__(
        self,
        env,
        learning_rate: float = 3e-3,  # Even higher LR for faster learning
        initial_epsilon: float = 0.0,  # No random exploration - always use heuristic or network
        epsilon_decay: float = 0.9999,
        final_epsilon: float = 0.0,
        gamma: float = 0.99,
        buffer_capacity: int = 50_000,  # Smaller buffer, fresher data
        batch_size: int = 256,  # Larger batch
        target_update_interval: int = 100,  # More frequent updates
        n_step: int = 1,
        min_buffer_size: int = 256,  # Start learning earlier
        grad_clip: float = 1.0,  # Tighter gradient clipping
        tau: float = 0.01,  # Soft update
        device: str | None = None,
        bc_weight: float = 2.0,  # Higher BC weight to force learning heuristic
        heuristic_threshold: float = 0.0,  # Always force heuristic
        force_heuristic: bool = True  # Force heuristic like PPO
    ):
        self.env = env
        # Respect passed device; default to CUDA if available
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = self._infer_input_dim(env)
        self.action_dim = env.action_space.n

        self.q = DuelingQNetwork(self.input_dim, self.action_dim).to(self.device)
        self.q_target = DuelingQNetwork(self.input_dim, self.action_dim).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.tau = float(tau)

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.buffer = ReplayBuffer(buffer_capacity)
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.update_steps = 0
        self.n_step = max(1, int(n_step))
        self.n_step_buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
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

    def _get_heuristic_action(self, obs: Dict[str, Any]) -> Tuple[int, bool]:
        # Pure RL: no heuristic forcing
        return None, False

    def get_action(self, obs: Dict[str, Any], training: bool = True) -> int:
        if training and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        with torch.no_grad():
            state = self._process_obs(obs)
            q_values = self.q(state)
            return int(torch.argmax(q_values, dim=-1).item())

    def store_transition(self, obs: Dict[str, Any], action: int, reward: float, done: bool, next_obs: Dict[str, Any]):
        s = self._process_obs(obs).detach().cpu().numpy()
        ns = self._process_obs(next_obs).detach().cpu().numpy()
        transition = (s, action, reward, ns, done)
        self.n_step_buffer.append(transition)

        # When we have at least n transitions, push one n-step transition
        if len(self.n_step_buffer) >= self.n_step:
            R = 0.0
            for i in range(self.n_step):
                r_i = self.n_step_buffer[i][2]
                R += (self.gamma ** i) * r_i
            s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
            next_s_n, done_n = self.n_step_buffer[self.n_step - 1][3], self.n_step_buffer[self.n_step - 1][4]
            self.buffer.add(s0, a0, R, next_s_n, done_n, self.n_step)
            # pop the oldest
            self.n_step_buffer.pop(0)

        # If episode ended, flush remaining transitions with shorter horizon
        if done:
            while len(self.n_step_buffer) > 0:
                k = len(self.n_step_buffer)
                R = 0.0
                for i in range(k):
                    r_i = self.n_step_buffer[i][2]
                    R += (self.gamma ** i) * r_i
                s0, a0 = self.n_step_buffer[0][0], self.n_step_buffer[0][1]
                next_s_k, done_k = self.n_step_buffer[-1][3], self.n_step_buffer[-1][4]
                self.buffer.add(s0, a0, R, next_s_k, done_k, k)
                self.n_step_buffer.pop(0)

    def update(self):
        if len(self.buffer) < max(self.batch_size, self.min_buffer_size):
            return
        states, actions, rewards, next_states, dones, steps, idxs, is_weights = self.buffer.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        steps_t = torch.tensor(steps, dtype=torch.int64, device=self.device).unsqueeze(-1)
        is_w_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(-1)

        # Double DQN target
        q_values = self.q(states_t).gather(1, actions_t)
        with torch.no_grad():
            next_actions = self.q(next_states_t).argmax(dim=1, keepdim=True)
            next_q_target = self.q_target(next_states_t).gather(1, next_actions)
            gamma_pow = torch.pow(torch.tensor(self.gamma, dtype=torch.float32, device=self.device), steps_t.float())
            target_q = rewards_t + (1.0 - dones_t) * gamma_pow * next_q_target

        td_errors = target_q - q_values
        loss = (is_w_t * nn.SmoothL1Loss(reduction='none')(q_values, target_q)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optimizer.step()
        self.training_loss.append(loss.item())
        # update priorities
        self.buffer.update_priorities(idxs, td_errors.detach().cpu().numpy().squeeze())

        self.update_steps += 1
        if self.tau > 0.0:
            with torch.no_grad():
                for target_param, param in zip(self.q_target.parameters(), self.q.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        else:
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
