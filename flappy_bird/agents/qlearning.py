import os
import pickle
from collections import defaultdict
import gymnasium as gym
import numpy as np
from typing import Dict, Any


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
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        self.lr = learning_rate
        self.discount_factor = discount_factor  # How much we care about future rewards

        # Exploration parameters
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Track learning progress
        self.training_error = []

    def _obs_to_key(self, obs: Dict[str, Any]) -> tuple:
        return tuple((key, tuple(value) if isinstance(value, (list, np.ndarray)) else value) 
                    for key, value in sorted(obs.items()))

    def get_action(self, obs: Dict[str, Any]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stay) or 1 (flap)
        """
        state_key = self._obs_to_key(obs)

        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        else:
            return int(np.argmax(self.q_values[state_key]))

    def update(
        self,
        obs: Dict[str, Any],
        action: int,
        reward: float,
        terminated: bool,
        next_obs: Dict[str, Any],
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
        state_key = self._obs_to_key(obs)
        next_state_key = self._obs_to_key(next_obs)

        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not terminated) * np.max(self.q_values[next_state_key])

        # What should the Q-value be? (Bellman equation)
        target = reward + self.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[state_key][action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[state_key][action] = (
            self.q_values[state_key][action] + self.lr * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    def save(self, filepath: str = "./results/flappy-bird/qlearning/best_agent.pkl"):
        """Save agent using pickle (preserves all data types and structure)."""
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