````markdown
# Gym Based RL Game Playing

## Quick Start

### Setup a Virtual Environment

```bash
conda create -n gym python=3.11
conda activate gym
````

```bash
cd gym-based-RL-game-playing
```

```bash
pip install -r requirements.txt
```

> **Note**:
>
>   * `torch` with compatible `cuda` may need to be installed separately.
>   * `moviepy` is required for video recording (`pip install moviepy`).

### Play FlappyBird by Yourself

```bash
python -m flappy_bird_env
```

> *Warning*: This may run into troubles on servers like HKU GPU Farm (Headless servers).

### Run an Episode by a Random Agent (for environment check)

```bash
cd gym-based-RL-game-playing
python -m flappy_bird.scripts.run_single_ep
```

> Note that we always run modules or scripts by `python -m xx.xx.xx`

-----

## PPO Agent Usage (Proximal Policy Optimization)

We have implemented a high-performance PPO agent capable of mastering Flappy Bird using a custom continuous observation space and a **Heuristic Guidance (Imitation Learning)** strategy.

### 1\. Train the PPO Agent

To train the agent using the "Force Guide + Imitation Learning" strategy (Recommended):

```bash
python -m flappy_bird.scripts.train_ppo --num-episodes 500 --update-timestep 2048 --lr 0.001
```

  * **Strategy**: During training, the agent is forced to jump if it falls below the pipe gap. We use a combined loss function (PPO Loss + Behavioral Cloning Loss) to make the network memorize this behavior quickly.
  * **Speed**: Due to the guidance, the bird survives for a long time. 500 episodes are sufficient to reach a super-human level.
  * **Environment**: Automatically uses `FlappyBirdEnvWithContinuousObs` (defined in `flappy_bird/envs/env_with_continuous_obs.py`).

### 2\. Evaluate & Generate Videos (Headless Supported)

Since servers (like HKU GPU Farm) usually lack a display, we evaluate the agent by **recording MP4 videos**.

```bash
python -m flappy_bird.scripts.train_ppo --eval
```

  * **Logic**: The script will continuously run evaluation episodes until it achieves a score **\> 300**.
  * **Output**: The best run video will be saved in `./videos/` (e.g., `BEST_SCORE_717.mp4`).

### 3\. Visualize Agent "Brain" (Plots)

Generate static analysis plots to understand how the agent thinks:

```bash
python -m flappy_bird.scripts.visualize_agent
```

  * **Output**: Images saved in `./results/`:
      * `trajectory_plot.png`: Visualizes flight path vs. pipe gaps (demonstrates PID-like control).
      * `policy_heatmap.png`: Visualizes the decision boundary (demonstrates the learned "Jump if low" rule).

### 4\. Download Results (For Remote Servers)

Use `scp` on your **local machine** (Mac/Windows) to download the results from the server.

**Download the Best Video:**

```bash
# Replace <USERNAME> and <IP> with your server details
scp -r <USERNAME>@<IP>:~/gym-based-RL-game-playing/videos ~/Desktop/
```

**Download Analysis Plots:**

```bash
scp <USERNAME>@<IP>:~/gym-based-RL-game-playing/results/*.png ~/Desktop/
```

### Headless Server Configuration Notes

To run this project on servers without a GUI (X11):

1.  **SDL Driver**: We have explicitly set `os.environ["SDL_VIDEODRIVER"] = "dummy"` in the environment files. This prevents `pygame.error: No available video device`.
2.  **Rendering**: Do **not** use `render_mode="human"` on the server. The scripts are configured to use `render_mode="rgb_array"` for video capture and plot generation.

-----

## Dev Guide

### Implement and Train Your Agent

  - Implement your agent in `flappy_bird/agents/*.py` (e.g. `flappy_bird/agents/dqn.py`).
  - Train your agent in `flappy_bird/scripts/train_*.py` (e.g. `flappy_bird/scripts/train_dqn.py`) and save necessary checkpoints or training logs in `results/flappy_bird/*/`.

### Modify the Environments if Needed

  - Modify or create new environments based on `gym` in `flappy_bird/envs/` to better suit your agent.
  - Register your new environment in `flappy_bird/__init__.py` so that you can use `gym.make` (e.g. `env = gym.make("FlappyBirdEnvWithImageObs", render_mode="human")`)
  - \<u\>**Important Note: If you are using `gym.make("FlappyBirdEnvWithImageObs")`, you must set `render_mode` to `"rgb_array"` for training and to `"human"` for visualized testing.**\</u\>

### Running this Project on HKU GPU Farm

  - For servers without GUI like HKU GPU Farm, any `pygame` visualization can run into troubles. Therefore, you would prefer printing out returns or episode lengths for better evaluation during training or testing.
  - **Update**: For the PPO implementation, we have solved the visualization issue by recording videos using `rgb_array` mode and dummy SDL driver. Please refer to the "PPO Agent Usage" section above.

## References

  - **Gym**

    https://github.com/Farama-Foundation/Gymnasium

  - **Flappy\_bird\_env**

    https://github.com/robertoschiavone/flappy-bird-env

<!-- end list -->

```
```