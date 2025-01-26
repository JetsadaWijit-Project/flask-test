#!/bin/bash

# Navigate to the source directory
cd src || exit

# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install required dependencies
pip install -r requirements.txt

# Run the chatbot training script
python train_chatbot.py
