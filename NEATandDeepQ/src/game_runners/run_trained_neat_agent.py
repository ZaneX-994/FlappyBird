import pygame
from constants import *
from game_runners.run_default_agent import RunDefaultAgent

class RunBestNEATAgent(RunDefaultAgent):
    def __init__(self, network, screen):
        super().__init__(screen)
        self.network = network

    #-------------------------------------------------------------------------------
    # overriding run methods
    #-------------------------------------------------------------------------------
    def process_game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit_game()
        # compute NEAT AGENT decisions
        for bird in self.birds:
            output = self.network.activate((bird.y, abs(bird.y - self.nearest_pipe.y), abs(bird.y - (self.nearest_pipe.y + GAP_SIZE))))
            if output[0] > 0.5:
                bird.flap()