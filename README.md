# Gym Based RL Game Playing

## Quick Start

### Setup a Virtual Environment

```
conda create -n gym python=3.11
conda activate gym
```

```
cd gym
```

```
pip install -r requirements.txt
```

> `torch` with compatiable `cuda` may need to be installed separately.

### Play FlappyBird by  Yourself

```
python -m flappy_bird_env
```

> This may run into troubles on servers like HKU GPU Farm.

### Run an Episode by a Random Agent (for environment check)

```
cd gym
python -m flappy_bird.scripts.run_single_ep
```

> Note that we always run modules or scripts by `python -m xx.xx.xx`

## Dev Guide

### Implement and Train Your Agent

- Implement your agent in `gym/flappy_bird/agents/*.py` (e.g. `gym/flappy_bird/agents/dqn.py`).
- Train your agent in `gym/flappy_bird/scripts/train_*.py` (e.g. `gym/flappy_bird/train_dqn.py`) and save necessary checkpoints or training logs in `gym/results/flappy_bird/*/`.

### Modify the Environments if Needed

- Modify or create new environments based on `gym` in `gym/flappy_bird/envs/` to better suit your agent.
- Register your new environment in `gym/flappy_bird/__init__.py` so that you can use `gym.make` (e.g. `env = gym.make("FlappyBirdEnvWithImageObs", render_mode="human")`)
- <u>**Important Note: If you are using `gym.make("FlappyBirdEnvWithImageObs")`, you must set `render_mode` to `"rgb_array"` for training and to `"human"` for visualized testing.**</u>

### Running this Project on HKU GPU Farm

- For servers without GUI like HKU GPU Farm, any `pygame` visualization can run into troubles. Therefore, you would prefer printing out returns or episode lengths for better evaluation during training or testing.

## References

- **Gym**

  https://github.com/Farama-Foundation/Gymnasium

- **Flappy_bird_env**

  https://github.com/robertoschiavone/flappy-bird-env

