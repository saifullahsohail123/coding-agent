#!/usr/bin/env python3
# This script will run the main game and update it if necessary.
import os
import subprocess

# Check if the game needs to be updated
if os.path.exists('update_game_files.py'):
    print('Updating game files...')
    subprocess.run(['python', 'update_game_files.py'])
else:
    print('No update available.')

# Run the main game script
print('Starting the game...')
subprocess.run(['python', 'game.py'])