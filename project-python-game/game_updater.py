import os

def update_game_files():
    files = ['snake_game.py', 'game.py']
    for file in files:
        with open(file, 'a') as f:
            f.write('\n# This line was added by the game updater.')

if __name__ == '__main__':
    update_game_files()
