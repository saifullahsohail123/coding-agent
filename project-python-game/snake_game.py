import pygame

class SnakeGame:
    def __init__(self):
        self.screen_width = 800
        self.screen_height = 600
        self.block_size = 20
        self.bg_color = (0, 0, 255)

        self.snake_pos = [100, 50]
        self.snake_body = [[100, 50], [90, 50], [80, 50]]
        self.direction = 'RIGHT'

        self.apple_pos = [200, 150]

    def run_game(self):
        pygame.init()
        screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        clock = pygame.time.Clock()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and self.direction != 'DOWN':
                        self.direction = 'UP'
                    elif event.key == pygame.K_DOWN and self.direction != 'UP':
                        self.direction = 'DOWN'
                    elif event.key == pygame.K_LEFT and self.direction != 'RIGHT':
                        self.direction = 'LEFT'
                    elif event.key == pygame.K_RIGHT and self.direction != 'LEFT':
                        self.direction = 'RIGHT'

            screen.fill(self.bg_color)
            for pos in self.snake_body:
                pygame.draw.rect(screen, (0, 255, 0), Rect(pos[0], pos[1], self.block_size, self.block_size))
            pygame.draw.rect(screen, (255, 0, 0), Rect(self.apple_pos[0], self.apple_pos[1], self.block_size, self.block_size))

            if self.direction == 'UP':
                self.snake_pos[1] -= self.block_size
                self.snake_body.insert(0, list(self.snake_pos))
                if self.snake_pos[1] < 0:
                    self.snake_pos[1] = self.screen_height - self.block_size
            elif self.direction == 'DOWN':
                self.snake_pos[1] += self.block_size
                self.snake_body.insert(0, list(self.snake_pos))
                if self.snake_pos[1] >= self.screen_height:
                    self.snake_pos[1] = 0
            elif self.direction == 'LEFT':
                self.snake_pos[0] -= self.block_size
                self.snake_body.insert(0, list(self.snake_pos))
                if self.snake_pos[0] < 0:
                    self.snake_pos[0] = self.screen_width - self.block_size
            elif self.direction == 'RIGHT':
                self.snake_pos[0] += self.block_size
                self.snake_body.insert(0, list(self.snake_pos))
                if self.snake_pos[0] >= self.screen_width:
                    self.snake_pos[0] = 0

            if self.snake_pos == self.apple_pos:
                self.apple_pos = [random.randint(0, self.screen_width - self.block_size) // self.block_size * self.block_size,
                                 random.randint(0, self.screen_height - self.block_size) // self.block_size * self.block_size]
            else:
                self.snake_body.pop()

            pygame.display.flip()
            clock.tick(10)
