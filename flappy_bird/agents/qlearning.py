import os
import pickle
from collections import defaultdict
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple


# --- 状态离散化参数 ---
# ⚠️ 这些值需要根据 Flappy Bird 游戏的实际观测范围和效果进行调整
# 它们定义了连续观测被划分成多少个“桶”
Y_DIFF_BIN_SIZE = 50  # 将与水管的垂直距离差每 50 像素作为一个桶
VELOCITY_BIN_SIZE = 1  # 将垂直速度每 1 个单位作为一个桶


class FlappyBirdAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # Q-table: maps (state, action) to expected reward
        # Q-Table的键将是离散化的元组 (y_discrete, velocity_discrete)
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def _obs_to_key(self, obs: Dict[str, Any]) -> Tuple[int, int]:
        """
        核心函数：将连续的观测字典离散化为可哈希的元组 (状态键)。
        
        Args:
            obs: 环境返回的原始观测字典，例如包含 'y_difference_to_pipe' 和 'velocity'。
            
        Returns:
            一个由整数组成的元组，作为Q-Table的键。
        """
        
        # 假设连续观测字典包含 'y_difference_to_pipe' 和 'velocity'
        
        # 1. 获取关键的连续观测值
        # 垂直距离差: 鸟中心到下一根水管中心Y轴上的距离差
        y_diff = obs.get("y_difference_to_pipe", 0.0)
        # 垂直速度
        velocity = obs.get("velocity", 0.0)

        # 2. 离散化 (Quantization)
        
        # 将连续值除以桶大小，然后取整（使用 int() 截断）。
        # 需要确保这些值在合理的范围内，不会导致 Q-Table 过大。
        
        # 离散化 Y 距离差
        # np.clip 用于限制数值范围，防止在游戏极端情况下产生巨大的索引
        y_discrete = int(np.clip(y_diff // Y_DIFF_BIN_SIZE, -10, 10))

        # 离散化 速度
        velocity_discrete = int(np.clip(velocity // VELOCITY_BIN_SIZE, -10, 10))

        # 3. 返回离散化的状态键
        return (y_discrete, velocity_discrete)


    def get_action(self, obs: Dict[str, Any]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stay) or 1 (flap)
        """
        # First, determine the intended action via epsilon-greedy
        state_key = self._obs_to_key(obs)
        if np.random.random() < self.epsilon:
            intended_action = self.env.action_space.sample()  # Explore
        else:
            intended_action = int(np.argmax(self.q_values[state_key]))  # Exploit

        # Now, apply the hardcoded safety rule.
        # Rule: Prevent jumping if the bird is at or above the center of the next pipe gap.
        if intended_action == 1:  # Agent wants to jump
            if self.env.unwrapped._pipes:
                bird_y = self.env.unwrapped._bird.y
                # Y-coordinate of the center of the gap in the next pipe
                pipe_gap_center_y = (
                    self.env.unwrapped._pipes[0].height
                    + self.env.unwrapped._pipes[0].gap / 2
                )

                # The game's coordinate system has Y=0 at the top.
                # So, a smaller Y value means a higher position.
                if bird_y <= pipe_gap_center_y:
                    return 0  # Override the jump action and do nothing

        return intended_action

    def update(
        self,
        obs: Dict[str, Any],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Dict[str, Any],
    ):
        """Update Q-value based on experience (Q-Learning Equation)."""
        state_key = self._obs_to_key(obs)
        next_state_key = self._obs_to_key(next_obs)

        # 1. 估算未来Q值： max_{a'} Q(s', a')
        # 如果回合结束 (terminated)，未来奖励为 0。
        future_q_value = (not terminated) * np.max(self.q_values[next_state_key])

        # 2. 计算目标 Q 值 (Target): r + gamma * max Q(s', a')
        target = reward + self.discount_factor * future_q_value

        # 3. 计算时序差分误差 (TD Error): Target - Current Q(s, a)
        temporal_difference = target - self.q_values[state_key][action]

        # 4. 更新 Q 值: Q(s, a) <- Q(s, a) + alpha * TD Error
        self.q_values[state_key][action] = (
            self.q_values[state_key][action] + self.lr * temporal_difference
        )

        # Track learning progress
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, filepath: str = "./results/flappy-bird/qlearning/best_agent.pkl"):
        """Save agent using pickle."""
        # Ensure directory exists
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'q_values': dict(self.q_values),
                'lr': self.lr,
                'discount_factor': self.discount_factor,
                'epsilon': self.epsilon,
                'epsilon_decay': self.epsilon_decay,
                'final_epsilon': self.final_epsilon,
                'training_error': self.training_error
            }, f)
        print(f"Agent saved to {filepath} using pickle")

    @classmethod
    def load(cls, filepath, env):
        """Load agent using pickle."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Create agent instance with dummy parameters (will be overwritten)
        agent = cls(env, 0.1, 1.0, 0.01, 0.1)
        
        # Restore all attributes
        agent.q_values = defaultdict(lambda: np.zeros(env.action_space.n))
        agent.q_values.update(data['q_values'])
        agent.lr = data['lr']
        agent.discount_factor = data['discount_factor']
        agent.epsilon = data['epsilon']
        agent.epsilon_decay = data['epsilon_decay']
        agent.final_epsilon = data['final_epsilon']
        agent.training_error = data['training_error']
        
        print(f"Agent loaded from {filepath} using pickle")
        return agent
