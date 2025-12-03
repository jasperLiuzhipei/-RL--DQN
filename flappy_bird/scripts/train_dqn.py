import numpy as np
from argparse import ArgumentParser
import gymnasium as gym
from tqdm import tqdm
import os

import flappy_bird
from flappy_bird.agents.dqn import FlappyBirdDQNAgent


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--num-episodes", type=int, default=5_000, help="Number of games to practice")
    parser.add_argument("--initial-epsilon", type=float, default=0.2, help="Initial epsilon for epsilon-greedy exploration")
    parser.add_argument("--final-epsilon", type=float, default=0.2, help="Final epsilon for epsilon-greedy exploration")
    parser.add_argument("--epsilon-decay", type=float, default=1.0, help="Epsilon decay rate")
    parser.add_argument("--save-path", type=str, default="./results/flappy-bird/dqn/best_agent.pth")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved agent, rather than train a new one")
    parser.add_argument("--env-id", type=str, default="FlappyBirdEnvWithContinuousObs")
    parser.add_argument("--target-update", type=int, default=500, help="Target network update interval (steps)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)

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
    )


def train(agent, env, args):
    global_step = 0
    for episode in tqdm(range(args.num_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.store_transition(obs, action, reward, done, next_obs)
            agent.update()
            obs = next_obs

            global_step += 1

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            avg_return = np.mean(np.array(env.return_queue)[-100:])
            print(f"\nAverage return of the last 100 episodes: {avg_return:.3f}")
            avg_length = np.mean(np.array(env.length_queue)[-100:])
            print(f"Average steps of the last 100 episodes: {avg_length:.1f}")


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

