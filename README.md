# Mc-Bot-AI

This project contains a reinforcement learning bot for the
[MineRL](https://minerl.readthedocs.io) Minecraft environment. During
training the bot can expand its neural network with additional hidden
layers when rewards stop improving, allowing it to adapt its own
architecture.

## Configuration

Edit `config.json` to provide Microsoft account credentials or a
fallback `player_name`. If no credentials are supplied, the script will
generate a random name and store it in the file.

## Usage

Training on a server:

```bash
./start_bot.sh -t -ip <server-ip> -p <port>
```

Running without training (defaults to port `25565` when `-p` is
omitted):

```bash
./start_bot.sh -r -ip <server-ip> -p <port>
# or simply
./start_bot.sh -ip <server-ip> -p <port>
```

When running with a Microsoft account the script launches a VNC server
on port `5900` for live viewing. You can tunnel it over SSH:

```bash
ssh -L 5900:localhost:5900 user@server
```

Then connect a VNC viewer to `localhost:5900` to watch the bot play.