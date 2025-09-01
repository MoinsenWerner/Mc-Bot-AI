# Mc-Bot-AI

This project contains a simple reinforcement learning bot for the
[MineRL](https://minerl.readthedocs.io) Minecraft environment. The bot
uses a PPO agent to learn how to obtain a diamond, mirroring the
survival gameplay loop.

## Usage

```bash
bash start_bot.sh --timesteps 2000
```

The first run installs all necessary dependencies and then starts
training. To evaluate a trained model, pass the `--eval` flag:

```bash
bash start_bot.sh --eval
```
