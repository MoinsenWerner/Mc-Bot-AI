"""Simple Minecraft bot using reinforcement learning.

This script uses the MineRL ObtainDiamond environment with a
Proximal Policy Optimization (PPO) agent from Stable-Baselines3.
The agent attempts to learn how to obtain a diamond, which is a
proxy for completing major parts of Minecraft survival gameplay.
"""

import argparse
from pathlib import Path

import gym
import minerl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv


def create_env(env_id: str):
    """Create a MineRL environment wrapped for Stable-Baselines3."""
    return DummyVecEnv([lambda: gym.make(env_id)])


def train(env_id: str, timesteps: int, model_path: Path):
    """Train a PPO agent on the specified MineRL environment."""
    env = create_env(env_id)
    model = PPO("CnnPolicy", env, verbose=1)
    model.learn(total_timesteps=timesteps)
    model.save(model_path)
    env.close()


def run(env_id: str, model_path: Path):
    """Run a trained model in the environment for a single episode."""
    env = create_env(env_id)
    model = PPO.load(model_path, env=env)
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _info = env.step(action)
        env.render()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train and run a simple MineRL bot")
    parser.add_argument("--timesteps", type=int, default=1000, help="Training timesteps")
    parser.add_argument(
        "--model-path", type=Path, default=Path("ppo_minerl"), help="Path to save the trained model"
    )
    parser.add_argument(
        "--env-id", default="MineRLObtainDiamond-v0", help="MineRL environment ID"
    )
    parser.add_argument("--eval", action="store_true", help="Run the trained model instead of training")

    args = parser.parse_args()

    if args.eval:
        run(args.env_id, args.model_path)
    else:
        train(args.env_id, args.timesteps, args.model_path)


if __name__ == "__main__":
    main()
