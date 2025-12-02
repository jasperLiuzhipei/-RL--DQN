import gymnasium as gym
from flappy_bird_env import FlappyBirdEnv
from .envs.env_with_customed_obs import FlappyBirdEnvWithCustomedObs
# 导入我们新写的连续环境
from .envs.env_with_continuous_obs import FlappyBirdEnvWithContinuousObs

gym.register(id="FlappyBirdEnvWithImageObs", entry_point=FlappyBirdEnv)
gym.register(id="FlappyBirdEnvWithCustomedObs", entry_point=FlappyBirdEnvWithCustomedObs)
# 注册 ID
gym.register(id="FlappyBirdEnvWithContinuousObs", entry_point=FlappyBirdEnvWithContinuousObs)