import os
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
import numpy as np
from typing import Dict, Any

# --- 神经网络定义 ---
class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        # Critic
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        
    def act(self, state):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        return action.item(), action_logprob, dist.entropy()
    
    # 新增辅助函数：专门用来重新计算特定动作的概率
    def get_logprob(self, state, action):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_tensor = torch.tensor(action, device=state.device)
        return dist.log_prob(action_tensor)
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = torch.distributions.Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy

# --- PPO Agent ---
class FlappyBirdPPOAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = 0.001, # 【修改】调大一点学习率，让它学快点
        gamma: float = 0.99,
        eps_clip: float = 0.2,
        K_epochs: int = 20, # 【修改】增加训练轮数，让它多背几遍
        batch_size: int = 2048,
        device: str = "cpu"
    ):
        self.env = env
        self.lr = learning_rate
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        # 强制使用 CPU (速度更快)
        self.device = torch.device("cpu")

        dummy_obs, _ = env.reset()
        self.input_dim = self._process_obs(dummy_obs).shape[0]
        self.action_dim = env.action_space.n

        self.policy = ActorCritic(self.input_dim, self.action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.policy_old = ActorCritic(self.input_dim, self.action_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer_states = []
        self.buffer_actions = []
        self.buffer_logprobs = []
        self.buffer_rewards = []
        self.buffer_is_terminals = []
        self.training_loss = []

    def _process_obs(self, obs: Dict[str, Any]) -> torch.Tensor:
        vals = []
        for key in sorted(obs.keys()):
            val = obs[key]
            if isinstance(val, (list, np.ndarray)):
                vals.extend(val)
            else:
                vals.append(val)
        return torch.FloatTensor(vals).to(self.device)

    def get_action(self, obs: Dict[str, Any], training=True) -> int:
        state = self._process_obs(obs)
        
        if training:
            with torch.no_grad():
                original_action, original_logprob, _ = self.policy_old.act(state)
            
            # --- 强制引导逻辑 ---
            current_y_diff = obs['pipe_y_diff'][0] 
            threshold = 0.02
            
            final_action = original_action
            is_forced = False

            if current_y_diff > threshold:
                final_action = 1 # 必须跳
                is_forced = True
            elif current_y_diff < -threshold:
                final_action = 0 # 必须不跳
                is_forced = True
            
            # 【重要修复】如果动作被我们改了，必须重新计算对应的 log_prob
            # 否则 PPO 算法会因为数据不匹配而失效
            final_logprob = original_logprob
            if is_forced and final_action != original_action:
                with torch.no_grad():
                    final_logprob = self.policy_old.get_logprob(state, final_action)

            self.buffer_states.append(state)
            self.buffer_actions.append(torch.tensor(final_action, device=self.device))
            self.buffer_logprobs.append(final_logprob)
            
            return final_action
        else:
            with torch.no_grad():
                probs = self.policy_old.actor(state)
                return torch.argmax(probs).item()

    def store_reward(self, reward: float, done: bool):
        self.buffer_rewards.append(reward)
        self.buffer_is_terminals.append(done)

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer_rewards), reversed(self.buffer_is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if rewards.std() > 1e-5:
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        else:
            rewards = (rewards - rewards.mean())

        if not self.buffer_states:
            return

        old_states = torch.stack(self.buffer_states).detach().to(self.device)
        old_actions = torch.stack(self.buffer_actions).detach().to(self.device)
        old_logprobs = torch.stack(self.buffer_logprobs).detach().to(self.device)

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            
            ratios = torch.exp(logprobs - old_logprobs)
            advantages = rewards - state_values.detach()

            # PPO Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            ppo_loss = -torch.min(surr1, surr2)
            
            # Value Loss
            value_loss = 0.5 * nn.MSELoss()(state_values, rewards)
            
            # 【核心修改】模仿学习 Loss (Behavior Cloning)
            # 我们直接最大化 buffer 中动作的概率 (即最小化 负log_prob)
            # 这会强迫神经网络去记住我们强制它做的那些动作
            bc_loss = -logprobs.mean() 
            
            # 总 Loss：PPO Loss + 1.0 * 模仿 Loss
            # 给模仿 Loss 一个很大的权重，强迫它学会
            loss = ppo_loss + value_loss - 0.01 * dist_entropy + 1.0 * bc_loss

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            self.training_loss.append(loss.mean().item())

        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.buffer_states.clear()
        self.buffer_actions.clear()
        self.buffer_logprobs.clear()
        self.buffer_rewards.clear()
        self.buffer_is_terminals.clear()

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_loss': self.training_loss
        }, filepath)
        print(f"PPO Agent saved to {filepath}")

    @classmethod
    def load(cls, filepath, env):
        agent = cls(env)
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=agent.device)
            agent.policy.load_state_dict(checkpoint['model_state_dict'])
            agent.policy_old.load_state_dict(checkpoint['model_state_dict'])
            agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            agent.training_loss = checkpoint.get('training_loss', [])
            print(f"PPO Agent loaded from {filepath}")
        return agent