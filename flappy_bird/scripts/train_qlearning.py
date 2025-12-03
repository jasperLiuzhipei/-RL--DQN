import numpy as np
from argparse import ArgumentParser
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt
import os

# 导入环境注册逻辑 (flappy_bird/__init__.py)
import flappy_bird 
# 导入 Q-Learning Agent 类
from flappy_bird.agents.qlearning import FlappyBirdAgent 

# ----------------------------------------------------------------------
# 辅助函数：解析命令行参数
# ----------------------------------------------------------------------
def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate (alpha)")
    parser.add_argument("--num-episodes", type=int, default=50_000, help="Number of games to practice")
    parser.add_argument("--initial-epsilon", type=float, default=1.0, help="Starting exploration rate")
    parser.add_argument("--final-epsilon", type=float, default=0.01, help="Final exploration rate")
    parser.add_argument("--save-path", type=str, default="./results/flappy-bird/qlearning/best_agent.pkl")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved agent, rather than train a new one")

    args = parser.parse_args()
    return args

# ----------------------------------------------------------------------
# 辅助函数：初始化环境
# ----------------------------------------------------------------------
def initialize_env(args):
    # Q-Learning 必须使用离散化的观测空间环境
    env = gym.make("FlappyBirdEnvWithCustomedObs") 
    
    # 使用 RecordEpisodeStatistics 记录每个回合的奖励和长度
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=args.num_episodes)

    return env

# ----------------------------------------------------------------------
# 辅助函数：初始化智能体
# ----------------------------------------------------------------------
def initialize_agent(env, args):
    # 计算 epsilon 衰减值：在 num_episodes/2 的回合内从 initial 降到 final
    epsilon_decay_rate = (args.initial_epsilon - args.final_epsilon) / (args.num_episodes / 2)
    
    return FlappyBirdAgent(
        env=env,
        learning_rate=args.lr,
        initial_epsilon=args.initial_epsilon,
        epsilon_decay=epsilon_decay_rate, # 每回合衰减量
        final_epsilon=args.final_epsilon,
    )

# ----------------------------------------------------------------------
# 核心函数：训练循环
# ----------------------------------------------------------------------
def train(agent, env, args):
    print(f"Starting Q-Learning training for {args.num_episodes} episodes...")
    
    for episode in tqdm(range(args.num_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            # 1. 选择动作 (使用 epsilon-greedy 策略)
            action = agent.get_action(obs) 
            
            # 2. 执行动作
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # 3. Q 值更新 (每步更新)
            agent.update(obs, action, reward, terminated, next_obs)
            
            done = terminated or truncated
            obs = next_obs

        # 4. 探索率衰减 (每回合衰减)
        agent.decay_epsilon() 

        # 打印训练进度
        if (episode + 1) % 1000 == 0:
            if len(env.return_queue) > 0:
                avg_return = np.mean(np.array(env.return_queue)[-1000:])
                print(f"\nEpisode {episode+1} | Epsilon: {agent.epsilon:.4f}")
                print(f"Average reward of the last 1000 episodes: {avg_return:.3f}")
                avg_length = np.mean(np.array(env.length_queue)[-1000:])
                print(f"Average steps of the last 1000 episodes: {avg_length:.1f}")

# ----------------------------------------------------------------------
# 辅助函数：绘图逻辑
# ----------------------------------------------------------------------
def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


def plot_training_results(agent, env):
    # 设置 Matplotlib 后端，防止无头服务器报错
    plt.switch_backend('Agg')
    # Smooth over a 1000-episode window
    rolling_length = 1000
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning - TD Error)
    axs[2].set_title("Training Error (TD Error)")
    # 注意：Q-Agent 的 training_error 记录的是每一步的 TD Error，通常比 Episode 数量多得多
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length * 50, # 增大窗口，因为数据点更多
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    
    save_path = "./results/qlearning_training_summary.png"
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path)
    print(f"\nTraining summary plot saved to {save_path}")
    plt.close()


def save_agent(agent, args):
    """保存 Q-Table 和配置到 .pkl 文件中。"""
    agent.save(args.save_path)


def test_agent(agent, env, num_episodes=100):
    """Test agent performance without learning or exploration."""
    total_rewards = []
    total_steps = []

    # 临时禁用探索，以测试纯粹的策略利用能力
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

    print(f"\nTesting agent performance over {num_episodes} episodes (Pure Exploitation)...")

    for _ in tqdm(range(num_episodes)):
        obs, info = env.reset()
        episode_reward = 0
        episode_step = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_step += 1
            done = terminated or truncated

        total_rewards.append(episode_reward)
        total_steps.append(episode_step)

    # 恢复原始 epsilon
    agent.epsilon = old_epsilon

    average_reward = np.mean(total_rewards)
    average_steps = np.mean(total_steps)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Average Steps: {average_steps:.1f}")
    print(f"Standard Deviation of Reward: {np.std(total_rewards):.3f}")


def test_with_render(agent):
    """渲染一局游戏以便人工观察性能（不录制视频）。"""
    print("\nStarting visual test run (render_mode='human')...")
    # 创建新的环境实例用于渲染
    env = gym.make("FlappyBirdEnvWithCustomedObs", render_mode="human")
    
    # 临时禁用探索
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0 

    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = next_obs
        total_reward += reward

    # 恢复 epsilon 并关闭环境
    agent.epsilon = old_epsilon
    print(f"Visual episode finished! Total reward: {total_reward}")
    env.close()


# ----------------------------------------------------------------------
# 主程序入口
# ----------------------------------------------------------------------
if __name__ == "__main__":

    args = parse_args()

    # 1. 初始化环境
    env = initialize_env(args)

    if args.eval:
        # 评估模式：加载模型并测试
        agent = FlappyBirdAgent.load(args.save_path, env)
    else:
        # 训练模式：初始化、训练、绘图、保存
        agent = initialize_agent(env, args)
        train(agent, env, args)
        plot_training_results(agent, env)
        save_agent(agent, args)
        
    # 3. 最终测试和可视化
    test_agent(agent, env)
    env.close() # 关闭用于测试的环境
    
    # 运行可视化游戏（不录制视频）
    test_with_render(agent)
