import numpy as np
from argparse import ArgumentParser
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt

import flappy_bird
from flappy_bird.agents.qlearning import FlappyBirdAgent


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-episodes", type=int, default=10_000, help="Number of games to practice")
    parser.add_argument("--initial-epsilon", type=float, default=0.1, help="Exploration rate to begin with")
    parser.add_argument("--final-epsilon", type=float, default=0.001, help="Final exploration rate")
    parser.add_argument("--save-path", type=str, default="./results/flappy-bird/qlearning/best_agent.pkl")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved agent, rather than train a new one")

    args = parser.parse_args()
    return args


def initialize_env(args):
    env = gym.make("FlappyBirdEnvWithCustomedObs")
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=args.num_episodes)

    obs, info = env.reset()
    print(obs)
    print(info)

    return env


def initialize_agent(env, args):
    return FlappyBirdAgent(
        env=env,
        learning_rate=args.lr,
        initial_epsilon=args.initial_epsilon,
        epsilon_decay=args.initial_epsilon / (args.num_episodes / 2),
        final_epsilon=args.final_epsilon,
    )


def train(agent, env, args):
    for episode in tqdm(range(args.num_episodes)):
        obs, info = env.reset()
        done = False

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            agent.update(obs, action, reward, terminated, next_obs)
            done = terminated or truncated
            obs = next_obs

        agent.decay_epsilon()

        if (episode + 1) % 100 == 0:
            avg_return = np.mean(np.array(env.return_queue)[-100:])
            print(f"\nAverage return of the last 100 episodes: {avg_return:.3f}")
            avg_length = np.mean(np.array(env.length_queue)[-100:])
            print(f"Average steps of the last 100 episodes: {avg_length:.1f}")


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


def plot_training_results(agent, env):
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

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


def test_agent(agent, env, num_episodes=100):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = agent.epsilon
    agent.epsilon = 0.0  # Pure exploitation

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

    # Restore original epsilon
    agent.epsilon = old_epsilon

    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")


def save_agent(agent, args):
    agent.save(args.save_path)


def test_with_render(agent):
    env = gym.make("FlappyBirdEnvWithCustomedObs", render_mode="human")
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

    env = initialize_env(args)

    if args.eval:
        agent = FlappyBirdAgent.load(args.save_path, env)
    else:
        agent = initialize_agent(env, args)
        train(agent, env, args)
        plot_training_results(agent, env)
        save_agent(agent, args)
        
    test_agent(agent, env)
    env.close()
    
    test_with_render(agent)
