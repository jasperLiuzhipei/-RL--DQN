import numpy as np
import gymnasium as gym
from gymnasium.spaces import Dict, Discrete

from flappy_bird_env import FlappyBirdEnv


def discretize(value, min_val, max_val, n_bins):
    clipped_value = np.clip(value, min_val, max_val)
    normalized = (clipped_value - min_val) / (max_val - min_val)
    discrete = int(normalized * (n_bins - 1))
    return discrete


class FlappyBirdEnvWithCustomedObs(FlappyBirdEnv):
    
    observation_space = Dict({
        "bird_height": Discrete(10),
        "x_to_pipe": Discrete(10),
        "y_to_gap_center": Discrete(20),
    })

    @property
    def observation(self):
        return {
            "bird_height": discretize(
                self._bird.y,
                0, 730 - self._bird.image.get_height(),
                self.observation_space["bird_height"].n
            ),
            "x_to_pipe": discretize(
                self._pipes[0].x + self._pipes[0].pipe_bottom.get_width() - self._bird.x,
                0, self._base.width - self._bird.x,
                self.observation_space["x_to_pipe"].n
            ),
            "y_to_gap_center": discretize(
                self._bird.y - (self._pipes[0].height + self._pipes[0].gap / 2),
                -550, 580 - self._bird.image.get_height(),
                self.observation_space["y_to_gap_center"].n
            )
        }
    
    @property
    def reward(self):
        if any([not pipe.passed and pipe.x < self._bird.x
                for pipe in self._pipes]):
            return 1
        elif not self.terminated:
            return 0.01
        else:
            return -1