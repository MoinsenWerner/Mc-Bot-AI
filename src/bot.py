"""Minecraft bot using reinforcement learning with adaptive networks.

The bot can train in the MineRL environment and adjusts its neural
network architecture during training by adding hidden layers when
progress stalls. Credentials or offline player names are loaded from
``config.json``.
"""

import argparse
import json
from pathlib import Path
from uuid import uuid4

import gym
import minerl
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"


def load_config():
    """Load configuration and ensure a player name exists."""
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text())
    else:
        cfg = {}
    if not cfg.get("microsoft_email") or not cfg.get("microsoft_password"):
        if not cfg.get("player_name"):
            cfg["player_name"] = f"Bot{uuid4().hex[:8]}"
            CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return cfg


def login(cfg):
    """Return the username to use based on config values."""
    if cfg.get("microsoft_email") and cfg.get("microsoft_password"):
        import minecraft_launcher_lib

        login_data = minecraft_launcher_lib.microsoft_account.login_with_credentials(
            cfg["microsoft_email"], cfg["microsoft_password"]
        )
        return login_data["name"]
    return cfg.get("player_name")


def create_env(env_id: str, _host: str, _port: int):
    """Create a MineRL environment wrapped for Stable-Baselines3.

    Host and port are accepted for connecting to remote servers but are
    currently unused and kept for future integration.
    """

    return DummyVecEnv([lambda: gym.make(env_id)])


def evaluate(model, env) -> float:
    """Evaluate current model for a single episode."""
    obs = env.reset()
    done = False
    total = 0.0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        total += float(reward)
    return total


def train(env_id: str, timesteps: int, model_path: Path, host: str, port: int):
    """Train a PPO agent that can expand its network when progress stalls."""
    env = create_env(env_id, host, port)
    net_arch = [64, 64]
    model = PPO("CnnPolicy", env, policy_kwargs={"net_arch": net_arch}, verbose=1)
    best_reward = -float("inf")
    phase_steps = max(1, timesteps // 3)
    total_steps = 0

    while total_steps < timesteps:
        model.learn(total_timesteps=phase_steps)
        total_steps += phase_steps
        reward = evaluate(model, env)
        if reward <= best_reward:
            net_arch.append(64)
            model = PPO("CnnPolicy", env, policy_kwargs={"net_arch": net_arch}, verbose=1)
        else:
            best_reward = reward

    model.save(model_path)
    env.close()


def run(env_id: str, model_path: Path, host: str, port: int):
    """Run a trained model in the environment for a single episode."""
    env = create_env(env_id, host, port)
    model = PPO.load(model_path, env=env)
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _info = env.step(action)
        env.render()
    env.close()


def main():
    parser = argparse.ArgumentParser(description="Train and run a MineRL bot")
    parser.add_argument("--timesteps", type=int, default=1000, help="Training timesteps")
    parser.add_argument(
        "--model-path", type=Path, default=Path("ppo_minerl"), help="Path for model"
    )
    parser.add_argument("--env-id", default="MineRLObtainDiamond-v0", help="MineRL environment ID")
    parser.add_argument("--host", default="", help="Minecraft server IP")
    parser.add_argument("--port", type=int, default=25565, help="Minecraft server port")
    parser.add_argument("--train", action="store_true", help="Run training instead of evaluation")

    args = parser.parse_args()
    cfg = load_config()
    username = login(cfg)
    print(f"Using Minecraft username: {username}")

    if args.train:
        train(args.env_id, args.timesteps, args.model_path, args.host, args.port)
    else:
        run(args.env_id, args.model_path, args.host, args.port)


if __name__ == "__main__":
    main()
