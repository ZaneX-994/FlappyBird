from constants import *
from game_runners.run_default_agent import RunDefaultAgent
import pygame

class RunTrainedQAgent(RunDefaultAgent):
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
        # compute Q agent decisions (by output node index)
        action = self.network.act(self.get_state())
        if action:
            self.birds[0].flap()
    
    #-------------------------------------------------------------------------------
    # helper methods for deep-q network
    #-------------------------------------------------------------------------------

    def get_state(self):
        bird = self.birds[0]
        return [bird.y, bird.vel, abs(bird.y - self.nearest_pipe.y), abs(bird.y - (self.nearest_pipe.y + GAP_SIZE))]