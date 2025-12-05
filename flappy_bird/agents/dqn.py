import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, Tuple, List


class NoisyLinear(nn.Module):
    """Factorized NoisyNet layer for exploration (Fortunato et al., 2017)."""
    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.017):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('eps_in', torch.zeros(1, in_features))
        self.register_buffer('eps_out', torch.zeros(out_features, 1))
        self.sigma_init = sigma_init
        self.reset_parameters()

    def reset_parameters(self):
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma_init)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma_init)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            eps_in = torch.randn_like(self.eps_in)
            eps_out = torch.randn_like(self.eps_out)
            eps_w = self._f(eps_out) @ self._f(eps_in)
            weight = self.weight_mu + self.weight_sigma * eps_w
            bias = self.bias_mu + self.bias_sigma * self._f(eps_out).squeeze(1)
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return torch.nn.functional.linear(x, weight, bias)


class DuelingQNetwork(nn.Module):
    """Dueling network with optional NoisyNet. Supports C51 by returning per-action atom logits."""
    def __init__(self, input_dim: int, action_dim: int, use_noisy: bool = True, num_atoms: int | None = None):
        super().__init__()
        self.use_noisy = use_noisy
        self.num_atoms = num_atoms  # if set, output shape becomes [B, action_dim, num_atoms]
        # Feature extractor (ReLU for DQN style)
        self.feature = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        # Value stream
        if use_noisy:
            self.value_fc = NoisyLinear(128, 128)
            self.value_out = NoisyLinear(128, 1 if not num_atoms else num_atoms)
            self.adv_fc = NoisyLinear(128, 128)
            self.adv_out = NoisyLinear(128, action_dim if not num_atoms else action_dim * (num_atoms))
        else:
            self.value_fc = nn.Linear(128, 128)
            self.value_out = nn.Linear(128, 1 if not num_atoms else num_atoms)
            self.adv_fc = nn.Linear(128, 128)
            self.adv_out = nn.Linear(128, action_dim if not num_atoms else action_dim * (num_atoms))

        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        f = self.feature(x)
        v = self.value_out(self.activation(self.value_fc(f)))
        a = self.adv_out(self.activation(self.adv_fc(f)))
        if self.num_atoms is None:
            # Standard dueling Q
            return v + a - a.mean(dim=1, keepdim=True)
        else:
            # Reshape to distributions
            # v: [B, num_atoms]; a: [B, action_dim*num_atoms] -> [B, action_dim, num_atoms]
            a = a.view(a.size(0), -1, self.num_atoms)
            v = v.unsqueeze(1)  # [B, 1, num_atoms]
            # Broadcast v across actions, subtract mean over actions
            q_atoms = v + a - a.mean(dim=1, keepdim=True)
            return q_atoms


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

        self.beta_max = 1.0
        self.beta_anneal = 1e-5  # increase beta slightly per update

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

    def anneal_beta(self):
        self.beta = min(self.beta_max, self.beta + self.beta_anneal)


class FlappyBirdDQNAgent:
    """Rainbow-style DQN: Double+Dueling+N-step+PER+NoisyNet + C51 (distributional)."""
    def __init__(
        self,
        env,
        learning_rate: float = 3e-4,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        final_epsilon: float = 0.05,
        gamma: float = 0.99,
        buffer_capacity: int = 200000,
        batch_size: int = 256,
        target_update_interval: int = 1000,
        n_step: int = 3,
        min_buffer_size: int = 5000,
        grad_clip: float = 1.0,
        tau: float = 0.01,
        device: str | None = None,
        use_per: bool = True,
        use_noisy: bool = True,
        use_c51: bool = False,
        v_min: float = -5.0,
        v_max: float = 5.0,
        num_atoms: int = 51,
    ):
        self.env = env
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dim = self._infer_input_dim(env)
        self.action_dim = env.action_space.n

        self.use_c51 = use_c51
        self.num_atoms = num_atoms if use_c51 else None
        self.v_min = v_min
        self.v_max = v_max
        if self.use_c51:
            self.delta_z = (self.v_max - self.v_min) / (num_atoms - 1)
            self.support = torch.linspace(self.v_min, self.v_max, num_atoms, device=self.device)

        self.q = DuelingQNetwork(self.input_dim, self.action_dim, use_noisy=use_noisy, num_atoms=self.num_atoms).to(self.device)
        self.q_target = DuelingQNetwork(self.input_dim, self.action_dim, use_noisy=use_noisy, num_atoms=self.num_atoms).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optimizer = optim.Adam(self.q.parameters(), lr=learning_rate)
        # Cosine LR schedule for long runs
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=500000, eta_min=learning_rate * 0.1)
        self.gamma = gamma
        self.tau = float(tau)

        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.use_noisy = use_noisy

        self.use_per = use_per
        if self.use_per:
            self.buffer = ReplayBuffer(buffer_capacity)
        else:
            # Fallback to a standard buffer if PER is disabled
            from collections import deque
            self.buffer = deque(maxlen=buffer_capacity)


        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.update_steps = 0
        self.n_step = max(1, int(n_step))
        self.n_step_buffer: List[Tuple[np.ndarray, int, float, np.ndarray, bool]] = []
        self.min_buffer_size = min_buffer_size
        self.grad_clip = grad_clip
        
        # Track environment interaction steps
        self.step_count = 0

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

    def get_action(self, obs: Dict[str, Any], training: bool = True) -> int:
        # Exploration: NoisyNet (if enabled) else epsilon-greedy
        if training and (not self.use_noisy) and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()
        # Greedy action from Q
        with torch.no_grad():
            state = self._process_obs(obs)
            out = self.q(state)
            if self.use_c51:
                # out: [1, action_dim, num_atoms]; expectation over support
                probs = torch.softmax(out, dim=-1)
                q_values = torch.sum(probs * self.support.view(1, 1, -1), dim=-1)
            else:
                q_values = out
            return int(torch.argmax(q_values, dim=-1).item())

    def store_transition(self, obs: Dict[str, Any], action: int, reward: float, done: bool, next_obs: Dict[str, Any]):
        # Count environment step
        self.step_count += 1
        s = self._process_obs(obs).detach().cpu().numpy()
        ns = self._process_obs(next_obs).detach().cpu().numpy()
        # Reward clipping to stabilize TD targets
        r = float(np.clip(reward, -1.0, 1.0))
        transition = (s, action, r, ns, done)
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
        # Anneal PER beta toward 1.0
        if self.use_per:
            self.buffer.anneal_beta()
        states, actions, rewards, next_states, dones, steps, idxs, is_weights = self.buffer.sample(self.batch_size)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(-1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(-1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(-1)
        steps_t = torch.tensor(steps, dtype=torch.int64, device=self.device).unsqueeze(-1)
        is_w_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device).unsqueeze(-1)
        if not self.use_c51:
            # Standard Double DQN TD loss
            q_values = self.q(states_t).gather(1, actions_t)
            with torch.no_grad():
                next_actions = self.q(next_states_t).argmax(dim=1, keepdim=True)
                next_q_target = self.q_target(next_states_t).gather(1, next_actions)
                gamma_pow = torch.pow(torch.tensor(self.gamma, dtype=torch.float32, device=self.device), steps_t.float())
                target_q = rewards_t + (1.0 - dones_t) * gamma_pow * next_q_target
            td_errors = target_q - q_values
            td_loss = (is_w_t * nn.SmoothL1Loss(reduction='none')(q_values, target_q)).mean()
            loss = td_loss
        else:
            # C51 distributional loss with Double DQN for next action
            logits = self.q(states_t)  # [B, A, Z]
            log_probs = torch.log_softmax(logits, dim=-1)
            chosen_log_probs = log_probs.gather(1, actions_t.unsqueeze(-1).expand(-1, -1, self.num_atoms)).squeeze(1)  # [B, Z]

            with torch.no_grad():
                next_logits = self.q(next_states_t)  # [B, A, Z]
                next_probs = torch.softmax(next_logits, dim=-1)
                # Compute expectations for action selection
                next_q = torch.sum(next_probs * self.support.view(1, 1, -1), dim=-1)  # [B, A]
                next_actions = next_q.argmax(dim=1)  # [B]
                next_target_logits = self.q_target(next_states_t)  # [B, A, Z]
                next_target_probs = torch.softmax(next_target_logits, dim=-1)
                next_target_probs_a = next_target_probs[torch.arange(next_target_probs.size(0)), next_actions]  # [B, Z]

                gamma_pow = torch.pow(torch.tensor(self.gamma, dtype=torch.float32, device=self.device), steps_t.float()).squeeze(1)  # [B]
                Tz = rewards_t.squeeze(1).unsqueeze(1) + (1.0 - dones_t.squeeze(1)).unsqueeze(1) * gamma_pow.unsqueeze(1) * self.support.view(1, -1)
                Tz = Tz.clamp(self.v_min, self.v_max)
                b = (Tz - self.v_min) / self.delta_z  # [B, Z]
                l = b.floor().long()
                u = b.ceil().long()
                # Projection onto support
                B = states_t.size(0)
                m = torch.zeros(B, self.num_atoms, device=self.device)
                for i in range(self.num_atoms):
                    l_idx = l[:, i]
                    u_idx = u[:, i]
                    prob = next_target_probs_a[:, i]
                    m.scatter_add_(1, l_idx.clamp(0, self.num_atoms - 1).unsqueeze(1), (prob * (u[:, i].float() - b[:, i])).unsqueeze(1))
                    m.scatter_add_(1, u_idx.clamp(0, self.num_atoms - 1).unsqueeze(1), (prob * (b[:, i] - l[:, i].float())).unsqueeze(1))

            # Cross-entropy loss between projected target m and predicted log_probs
            ce = -(m * chosen_log_probs).sum(dim=1)  # [B]
            loss = (is_w_t.squeeze(1) * ce).mean()
            td_errors = (m * self.support.view(1, -1)).sum(dim=1, keepdim=True) - (torch.exp(chosen_log_probs) * self.support.view(1, -1)).sum(dim=1, keepdim=True)


        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optimizer.step()
        self.lr_scheduler.step()
        self.training_loss.append(loss.item())
        
        if self.use_per:
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
        # If using NoisyNet, epsilon is not needed
        if self.use_noisy:
            self.epsilon = 0.0
            return
        if self.epsilon > self.final_epsilon:
            self.epsilon *= self.epsilon_decay

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
