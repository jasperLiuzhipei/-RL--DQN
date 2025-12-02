import numpy as np
from argparse import ArgumentParser
import gymnasium as gym
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import shutil # 用于删除低分视频文件夹

# 导入环境注册逻辑
import flappy_bird 
from flappy_bird.agents.ppo import FlappyBirdPPOAgent

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--num-episodes", type=int, default=500, help="Number of games to practice")
    parser.add_argument("--update-timestep", type=int, default=2048, help="Update policy every n timesteps")
    parser.add_argument("--save-path", type=str, default="./results/flappy-bird/ppo/best_agent.pth")
    parser.add_argument("--eval", action="store_true", help="Evaluate saved agent")
    args = parser.parse_args()
    return args

def initialize_env():
    env = gym.make("FlappyBirdEnvWithContinuousObs")
    env = gym.wrappers.RecordEpisodeStatistics(env, deque_size=100)
    return env

def initialize_agent(env, args):
    return FlappyBirdPPOAgent(
        env=env,
        learning_rate=args.lr
    )

def train(agent, env, args):
    time_step = 0
    for episode in tqdm(range(args.num_episodes)):
        obs, info = env.reset()
        done = False
        while not done:
            time_step += 1
            action = agent.get_action(obs, training=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            agent.store_reward(reward, done)
            if time_step % args.update_timestep == 0:
                agent.update()
            obs = next_obs
        if (episode + 1) % 10 == 0:
            if len(env.return_queue) > 0:
                avg_return = np.mean(np.array(env.return_queue)[-10:])
                print(f"\nEpisode {episode+1} | Avg Reward: {avg_return:.2f}")

def plot_training_results(agent, env):
    # 略去画图逻辑以节省篇幅，重点在下面的 test 函数
    pass

def test_with_render(agent):
    """
    修改版：循环运行，直到跑出 300 分以上的成绩才停止
    """
    target_score = 300
    print(f"Searching for a high score run (> {target_score})...")
    
    attempt = 0
    
    while True:
        attempt += 1
        # 为每一次尝试创建一个独立的临时文件夹，防止视频文件名冲突
        current_video_folder = f"./videos/attempt_{attempt}"
        
        try:
            # 创建环境
            env = gym.make("FlappyBirdEnvWithContinuousObs", render_mode="rgb_array")
            
            # 套上录像 Wrapper
            env = gym.wrappers.RecordVideo(
                env, 
                video_folder=current_video_folder, 
                episode_trigger=lambda x: True, # 录制这一局
                name_prefix="high_score_run"
            )
        except Exception as e:
            print(f"Error creating env: {e}")
            return

        obs, _ = env.reset()
        done = False
        total_reward = 0
        
        # --- 开始这一局 ---
        while not done:
            action = agent.get_action(obs, training=False)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        
        # 关闭环境以保存视频文件
        env.close()
        
        print(f"Attempt {attempt}: Score = {total_reward:.2f}", end="")
        
        # --- 判定逻辑 ---
        if total_reward > target_score:
            print(f"  ---> SUCCESS! (> {target_score})")
            print(f"Video saved in: {current_video_folder}")
            print("You can download this folder now.")
            
            # 为了方便你下载，我们可以把最好的视频复制到 videos 根目录
            best_video_path = f"./videos/BEST_SCORE_{int(total_reward)}.mp4"
            try:
                # 找到生成的 mp4 文件 (通常是 high_score_run-episode-0.mp4)
                src_file = os.path.join(current_video_folder, "high_score_run-episode-0.mp4")
                shutil.copy(src_file, best_video_path)
                print(f"Also copied to: {best_video_path}")
            except Exception as e:
                print(f"Could not copy file: {e}")
                
            break # 退出循环
        else:
            print("  (Too low, retrying...)")
            # 删除低分视频的文件夹，节省空间
            try:
                shutil.rmtree(current_video_folder)
            except:
                pass

if __name__ == "__main__":
    args = parse_args()
    env = initialize_env()

    if args.eval:
        agent = FlappyBirdPPOAgent.load(args.save_path, env)
        test_with_render(agent)
    else:
        agent = FlappyBirdPPOAgent(env, learning_rate=args.lr)
        train(agent, env, args)
        agent.save(args.save_path)
        test_with_render(agent)