import pygame

class SnakeGame:
    def __init__(self):
        # Initialize Pygame and set up the display
        pygame.init()
        self.display = pygame.display.set_mode((800, 600))

        # Set up the title of the window
        pygame.display.set_caption('Snake Game')

        # Define some colors
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)

        # Initialize the snake and food positions
        self.snake_pos = [100, 100]
        self.food_pos = [300, 300]

    def run(self):
        clock = pygame.time.Clock()
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                self.snake_pos[1] -= 10
            elif keys[pygame.K_DOWN]:
                self.snake_pos[1] += 10
            elif keys[pygame.K_LEFT]:
                self.snake_pos[0] -= 10
            elif keys[pygame.K_RIGHT]:
                self.snake_pos[0] += 10
            self.display.fill(self.black)
            pygame.draw.rect(self.display, self.white, (self.snake_pos[0], self.snake_pos[1], 10, 10))
            pygame.display.update()
            clock.tick(60)
        pygame.quit()