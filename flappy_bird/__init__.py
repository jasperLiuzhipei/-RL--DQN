import gymnasium as gym
from flappy_bird_env import FlappyBirdEnv
from .envs.env_with_customed_obs import FlappyBirdEnvWithCustomedObs


gym.register(id="FlappyBirdEnvWithImageObs", entry_point=FlappyBirdEnv)
gym.register(id="FlappyBirdEnvWithCustomedObs", entry_point=FlappyBirdEnvWithCustomedObs)