'''A simple game application.'''

import pygame
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Set up display
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Simple Game')

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False

    # Game logic goes here

    # Drawing code goes here
    screen.fill((0, 0, 0))  # Fill the screen with black

    # Update display
    pygame.display.flip()

# Clean up
pygame.quit()