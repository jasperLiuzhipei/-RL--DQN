import numpy as np
import torch
import matplotlib.pyplot as plt
import gymnasium as gym
import os
from flappy_bird.agents.ppo import FlappyBirdPPOAgent
import flappy_bird # 注册环境

# 设置 Matplotlib 后端，防止无头服务器报错
plt.switch_backend('Agg')

def plot_trajectory(agent, env_name="FlappyBirdEnvWithContinuousObs"):
    """
    1. 飞行轨迹图：画出小鸟的飞行高度 vs 管子缝隙高度
    """
    print("Generating Trajectory Plot...")
    env = gym.make(env_name)
    obs, _ = env.reset()
    
    bird_y_history = []
    gap_y_history = []
    actions = []
    
    done = False
    step = 0
    # 跑一局，限制最多 500 步，防止跑太久
    while not done and step < 300000:
        # 获取原始位置信息 (需要访问环境私有变量)
        bird_real_y = env.unwrapped._bird.y
        if len(env.unwrapped._pipes) > 0:
            gap_center = env.unwrapped._pipes[0].height + env.unwrapped._pipes[0].gap / 2
        else:
            gap_center = 256 # 默认中心
            
        bird_y_history.append(bird_real_y)
        gap_y_history.append(gap_center)
        
        action = agent.get_action(obs, training=False)
        actions.append(action)
        
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step += 1
    
    # 开始绘图
    plt.figure(figsize=(12, 6))
    
    # 1. 画管子中心线
    plt.plot(gap_y_history, color='green', linestyle='--', linewidth=2, label='Pipe Gap Center')
    
    # 2. 画小鸟轨迹
    plt.plot(bird_y_history, color='orange', linewidth=2, label='Bird Height')
    
    # 3. 标记跳跃点
    # 找到所有跳跃 (action=1) 的时刻
    jump_steps = [i for i, a in enumerate(actions) if a == 1]
    jump_y = [bird_y_history[i] for i in jump_steps]
    plt.scatter(jump_steps, jump_y, color='red', s=10, marker='^', label='Jump Action', zorder=5)

    plt.title(f"Agent Flight Trajectory (Score: {step/100:.2f})")
    plt.xlabel("Time Steps")
    plt.ylabel("Height (Pixels)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Y轴反转，因为游戏坐标系 0 在上面
    plt.gca().invert_yaxis()
    
    save_path = "./results/trajectory_plot.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

def plot_policy_heatmap(agent):
    """
    2. 策略热力图：展示神经网络在不同状态下的决策边界
    X轴: 离管子的水平距离
    Y轴: 离管子缝隙的垂直距离
    颜色: 跳跃的概率
    """
    print("Generating Policy Heatmap...")
    
    # 定义分辨率
    resolution = 100
    
    # 构造状态网格
    # 归一化后的范围大概是 [-1, 1]
    y_diffs = np.linspace(-0.5, 0.5, resolution) # Y轴：高度差
    x_dists = np.linspace(-1.0, 1.0, resolution) # X轴：水平距离
    
    # 固定的其他状态 (假设鸟的速度为 0，位置居中)
    fixed_bird_y = 0.0
    fixed_bird_v = 0.0 
    
    prob_grid = np.zeros((resolution, resolution))
    
    # 遍历网格
    with torch.no_grad():
        for i, y in enumerate(y_diffs):
            for j, x in enumerate(x_dists):
                # 构造符合 ContinuousObs 环境的输入 Tensor
                # 顺序参考 env 定义: bird_y, bird_v, pipe_x, pipe_y_diff
                # 注意：这里我们只改变 x 和 y_diff，观察决策变化
                obs_tensor = torch.tensor([
                    fixed_bird_y, 
                    fixed_bird_v, 
                    x, 
                    y
                ], dtype=torch.float32).to(agent.device)
                
                # 获取 Actor 输出的概率
                probs = agent.policy_old.actor(obs_tensor)
                # 获取跳跃 (Action 1) 的概率
                jump_prob = probs[1].item()
                prob_grid[i, j] = jump_prob

    # 绘图
    plt.figure(figsize=(10, 8))
    
    # 使用 imshow 绘制热力图
    plt.imshow(
        prob_grid, 
        extent=[-1, 1, -0.5, 0.5], # [x_min, x_max, y_min, y_max]
        origin='lower', 
        aspect='auto', 
        cmap='coolwarm' # 蓝色不跳，红色跳
    )
    
    plt.colorbar(label='Probability of Jumping')
    plt.title("Policy Decision Boundary (Brain Scan)")
    plt.xlabel("Normalized Distance to Pipe (Negative=Close, Positive=Far)")
    plt.ylabel("Normalized Height Diff (Negative=Too High, Positive=Too Low)")
    
    # 画一条中心线
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.text(-0.9, 0.02, "Bird is too LOW", color='black', fontsize=10)
    plt.text(-0.9, -0.05, "Bird is too HIGH", color='black', fontsize=10)
    
    save_path = "./results/policy_heatmap.png"
    plt.savefig(save_path)
    print(f"Saved: {save_path}")
    plt.close()

if __name__ == "__main__":
    # 1. 自动适配环境
    # 如果你现在用的是离散环境，这里可能会报错。
    # 建议切回 env_with_continuous_obs.py 使用此脚本效果最佳
    env_name = "FlappyBirdEnvWithContinuousObs"
    env = gym.make(env_name)
    
    # 2. 加载模型
    model_path = "./results/flappy-bird/ppo/best_agent.pth"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        exit()
        
    agent = FlappyBirdPPOAgent.load(model_path, env)
    
    # 3. 生成图表
    os.makedirs("./results", exist_ok=True)
    plot_trajectory(agent, env_name)
    plot_policy_heatmap(agent)