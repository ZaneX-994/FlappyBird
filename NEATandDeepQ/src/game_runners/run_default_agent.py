import os
import random
import pygame
from bird import Bird
from pipe import Pipe
from constants import *

class RunDefaultAgent:
    def __init__(self, screen):
        # game features
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.score = 0
        self.birds = [Bird()]
        self.nearest_pipe = Pipe(SCREEN_WIDTH, random.randint(GAP_SIZE / 2, SCREEN_HEIGHT / 2))
        self.farthest_pipe = Pipe((1.5 * SCREEN_WIDTH + 40), random.randint(GAP_SIZE / 2, SCREEN_HEIGHT / 2))
        # image loading
        self.bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (SCREEN_WIDTH, SCREEN_HEIGHT))
        self.floor_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","floor.png")).convert_alpha())

    #-------------------------------------------------------------------------------
    # default template for game at runtime
    #-------------------------------------------------------------------------------
    def run(self):
        if VISUALISE: 
            self.initial_screen()
        while True:
            self.process_game_events()
            self.move_game_entities()
            if self.check_collisions():
                return self.score
            if VISUALISE:
                self.ready_next_frame()

    #-------------------------------------------------------------------------------
    # default run methods
    #-------------------------------------------------------------------------------
    def initial_screen(self):
        # draw initial screen and wait for user keypress
        self.draw_game()
        intial_running = True
        while intial_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.exit_game()
                if event.type == pygame.KEYDOWN:
                    intial_running = False

    def process_game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit_game()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    for bird in self.birds:
                        bird.flap()

    def move_game_entities(self):
        # move game entities
        for bird in self.birds:
            bird.move()
        self.farthest_pipe.move()
        if self.nearest_pipe.move():
            self.switch_pipes()

    def check_collisions(self):
        for bird in self.birds:
            if self.nearest_pipe.collides_with(bird): 
                self.birds.remove(bird)
        if not self.birds: 
            return True
        return False

    def ready_next_frame(self):
        self.draw_game()
        pygame.display.update()
        self.clock.tick(MAX_FRAME_RATE)

    #-------------------------------------------------------------------------------
    # additional helper methods
    #-------------------------------------------------------------------------------
    def switch_pipes(self):
        self.score += 1
        # switch nearest and farthest pipe
        self.nearest_pipe, self.farthest_pipe = self.farthest_pipe, self.nearest_pipe

    def exit_game(self):
        print(self.score)
        pygame.quit()
        exit(0)

    def draw_game(self):
        self.screen.blit(self.bg_img, (0,0))
        for bird in self.birds:
            bird.draw(self.screen)
        self.nearest_pipe.draw(self.screen)
        self.farthest_pipe.draw(self.screen)
        pygame.display.update()