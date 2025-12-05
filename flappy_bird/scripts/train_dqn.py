import numpy as np
from argparse import ArgumentParser
import gymnasium as gym
from tqdm import tqdm
import os
import sys

# 将项目根目录加入路径，确保能导入 flappy_bird 包
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import flappy_bird
from flappy_bird.agents.dqn import FlappyBirdDQNAgent


def parse_args():
    parser = ArgumentParser()

    # Optimized defaults - aggressive settings
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num-episodes", type=int, default=10000, help="Number of games to practice")
    parser.add_argument("--initial-epsilon", type=float, default=0.3, help="Initial epsilon for exploration")
    parser.add_argument("--epsilon_decay", type=float, default=0.9995, help="Epsilon decay rate")
    parser.add_argument("--final-epsilon", type=float, default=0.01, help="Final epsilon value")
    parser.add_argument("--save-path", type=str, default="./results/flappy-bird/dqn/best_agent.pth")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved agent")
    parser.add_argument("--env-id", type=str, default="FlappyBirdEnvWithContinuousObs")
    parser.add_argument("--target-update", type=int, default=1000, help="Target network update frequency (steps)")
    parser.add_argument("--tau", type=float, default=0.01, help="Soft update coefficient for target network")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size for training")
    parser.add_argument("--buffer-capacity", type=int, default=200000, help="Replay buffer capacity")
    parser.add_argument("--min-buffer-size", type=int, default=2000, help="Minimum buffer size before training")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda", help="Device to use: cpu or cuda")
    parser.add_argument("--n-step", type=int, default=3, help="N-step returns")
    parser.add_argument("--train-freq", type=int, default=4, help="How often to train the network (in steps)")
    parser.add_argument("--gradient-steps", type=int, default=1, help="Number of gradient steps per update")
    parser.add_argument("--bc-weight", type=float, default=2.0, help="Behavior cloning weight")

    args = parser.parse_args()
    return args


def initialize_env(args):
    env = gym.make(args.env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=args.num_episodes)
    obs, info = env.reset()
    print(obs)
    print(info)
    return env


def initialize_agent(env, args):
    return FlappyBirdDQNAgent(
        env=env,
        learning_rate=args.lr,
        gamma=0.99,
        initial_epsilon=args.initial_epsilon,
        epsilon_decay=args.epsilon_decay,
        final_epsilon=args.final_epsilon,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_interval=args.target_update,
        n_step=args.n_step,
        tau=args.tau,
        device=args.device,
        bc_weight=args.bc_weight,
        force_heuristic=True,  # Force heuristic like PPO
    )


def train(agent, env, args):
    global_step = 0
    best_avg_reward = -float('inf')
    
    for episode in tqdm(range(args.num_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, done, next_obs)
            
            if global_step % args.train_freq == 0:
                for _ in range(args.gradient_steps):
                    agent.update()
            
            obs = next_obs
            global_step += 1

        agent.decay_epsilon()

        # Print and save best model every 10 episodes (like PPO)
        if (episode + 1) % 10 == 0:
            if len(env.return_queue) >= 10:
                avg_return = np.mean(np.array(env.return_queue)[-10:])
                print(f"\nEpisode {episode+1} | Average Reward (last 10): {avg_return:.2f}")
                
                # Save best model
                if avg_return > best_avg_reward:
                    best_avg_reward = avg_return
                    agent.save(args.save_path)
                    print(f"  -> New best! Saved to {args.save_path}")


def test_agent(agent, env, num_episodes=50):
    total_rewards = []

    old_epsilon = agent.epsilon
    agent.epsilon = 0.0

    for _ in range(num_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.get_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_rewards.append(episode_reward)

    agent.epsilon = old_epsilon
    average_reward = np.mean(total_rewards)
    print(f"Test Results over {num_episodes} episodes:")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")


def save_agent(agent, args):
    agent.save(args.save_path)
    print(f"DQN Agent saved to {args.save_path}")


def test_with_render(agent, env_id):
    env = gym.make(env_id, render_mode="human")
    obs, info = env.reset()
    total_reward = 0
    done = False

    while not done:
        action = agent.get_action(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = next_obs
        total_reward += reward

    print(f"Episode finished! Total reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_path) if os.path.dirname(args.save_path) else '.', exist_ok=True)

    env = initialize_env(args)

    if args.eval:
        agent = FlappyBirdDQNAgent.load(args.save_path, env)
    else:
        agent = initialize_agent(env, args)
        train(agent, env, args)
        save_agent(agent, args)

    test_agent(agent, env)
    env.close()

    test_with_render(agent, args.env_id)
