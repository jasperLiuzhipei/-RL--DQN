import os
# 1. 设置无头模式 (必须在 import pygame 之前)
os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame
import gymnasium as gym
from gymnasium.spaces import Box, Dict
from flappy_bird_env import FlappyBirdEnv

class FlappyBirdEnvWithContinuousObs(FlappyBirdEnv):
    """
    专为 PPO/DQN 等深度学习算法设计。
    直接返回归一化后的连续浮点数坐标，不进行离散化。
    """
    def __init__(self, render_mode=None):
        super().__init__(render_mode=render_mode)
        
        # 定义连续观测空间 (Box)
        self.observation_space = Dict({
            "bird_y": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "bird_v": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "pipe_x": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
            "pipe_y_diff": Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        })
        
        self._has_printed_debug = False
        
        # 【核心修复】显式初始化 _screen 变量，防止 AttributeError
        self._screen = None 

    def _get_bird_velocity(self):
        """安全地获取鸟的速度"""
        possible_names = ['vel', 'vel_y', 'velocity', 'v', 'speed', 'playerVelY', 'vals']
        for name in possible_names:
            if hasattr(self._bird, name):
                return getattr(self._bird, name)
        
        if not self._has_printed_debug:
            self._has_printed_debug = True
        return 0.0

    @property
    def observation(self):
        # 1. 获取原始数据
        bird_y = self._bird.y
        bird_v = self._get_bird_velocity()
        
        if len(self._pipes) > 0:
            pipe = self._pipes[0]
            pipe_x = pipe.x
            pipe_gap_center = pipe.height + pipe.gap / 2 
        else:
            pipe_x = 288.0
            pipe_gap_center = 256.0

        # 2. 归一化处理
        raw_bird_y = (bird_y / 512.0) * 2 - 1      
        raw_bird_v = bird_v / 10.0                 
        raw_pipe_x = (pipe_x / 288.0) * 2 - 1      
        raw_y_diff = (bird_y - pipe_gap_center) / 512.0 

        # 使用 np.clip 强制限制在 [-1, 1] 之间
        norm_bird_y = np.clip(raw_bird_y, -1.0, 1.0)
        norm_bird_v = np.clip(raw_bird_v, -1.0, 1.0)
        norm_pipe_x = np.clip(raw_pipe_x, -1.0, 1.0)
        norm_y_diff = np.clip(raw_y_diff, -1.0, 1.0)

        return {
            "bird_y": np.array([norm_bird_y], dtype=np.float32),
            "bird_v": np.array([norm_bird_v], dtype=np.float32),
            "pipe_x": np.array([norm_pipe_x], dtype=np.float32),
            "pipe_y_diff": np.array([norm_y_diff], dtype=np.float32) 
        }
    
    @property
    def reward(self):
        if any([not pipe.passed and pipe.x < self._bird.x for pipe in self._pipes]):
            return 1.0 
        elif not self.terminated:
            return 0.1 
        else:
            return -1.0 

    # 重写 render 函数，支持无头服务器视频录制
    def render(self):
        if self.render_mode == "rgb_array":
            # 如果屏幕还没初始化，现在初始化
            if self._screen is None:
                pygame.init()
                # 这里的尺寸 (288, 512) 是 Flappy Bird 的标准尺寸
                self._screen = pygame.display.set_mode((288, 512))
            
            # 绘制背景和精灵 (复用父类的逻辑，如果父类有 draw 方法)
            # 但为了保险，我们直接让父类画到我们的 dummy screen 上
            # 这里的 super().render() 可能会尝试 update display，在 dummy 模式下是安全的
            try:
                # 尝试调用父类的绘图逻辑
                super().render()
            except Exception:
                # 如果父类 render 报错（比如它试图创建一个窗口），我们这里捕获它
                # 但通常设置了 SDL_VIDEODRIVER="dummy" 后，父类操作也是安全的
                pass
            
            # 核心：从 pygame 的 surface 获取像素数组
            # 转置：(Width, Height, Channel) -> (Height, Width, Channel)
            return np.transpose(pygame.surfarray.array3d(self._screen), axes=(1, 0, 2))
        else:
            return super().render()