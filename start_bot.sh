#!/usr/bin/env bash
# Shell script to install dependencies and start the Minecraft AI bot.
# Intended for Ubuntu environments.
set -e

# Update package list and install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip openjdk-8-jdk-headless

# Install Python dependencies
pip3 install --upgrade pip
pip3 install -r requirements.txt

# Start training the bot
python3 src/bot.py "$@"
