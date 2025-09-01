#!/usr/bin/env bash
set -e

MODE="run"
HOST=""
PORT=25565

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t)
      MODE="train"
      shift
      ;;
    -r)
      MODE="run"
      shift
      ;;
    -ip)
      HOST="$2"
      shift 2
      ;;
    -p)
      PORT="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Update package list and install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip openjdk-8-jdk-headless x11vnc xvfb

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

if [[ "$MODE" == "run" ]]; then
  echo "Starting VNC server for remote viewing on port 5900"
  x11vnc -display :0 -forever -nopw -bg || true
  echo "Tunnel with: ssh -L 5900:localhost:5900 <server>" 
fi

# Execute the bot
python3 src/bot.py --host "$HOST" --port "$PORT" $( [[ "$MODE" == "train" ]] && echo "--train" )
