import neat
import pygame
from constants import *
from bird import Bird
from game_runners.run_default_agent import RunDefaultAgent

class RunNeatAgents(RunDefaultAgent):
    def __init__(self, genome_tuples, config, screen):
        super().__init__(screen)
        self.config = config
        self.networks = []
        self.birds = []
        self.genomes = []
        for _, genome in genome_tuples:
            self.networks.append(neat.nn.FeedForwardNetwork.create(genome, self.config))
            self.birds.append(Bird())
            genome.fitness = 0
            self.genomes.append(genome)

    #-------------------------------------------------------------------------------
    # overriding run methods
    #-------------------------------------------------------------------------------
    def process_game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit_game()
        # compute NEAT AGENT decisions
        for i, bird in enumerate(self.birds):
            output = self.networks[i].activate((bird.y, abs(bird.y - self.nearest_pipe.y), abs(bird.y - (self.nearest_pipe.y + GAP_SIZE))))
            if output[0] > 0.5:
                self.birds[i].flap()
            # increment fitness of all birds that haven't died
            self.genomes[i].fitness += 0.1

    def switch_pipes(self):
        super().switch_pipes()
        for genome in self.genomes:
            genome.fitness += 5

    def check_collisions(self):
        for i, bird in enumerate(self.birds):
            if self.nearest_pipe.collides_with(bird): 
                self.genomes[i].fitness -= 1
                self.networks.pop(i)
                self.birds.pop(i)
                self.genomes.pop(i)
        # if all birds die or max score exceeds fitness threshold, return end game signal 
        if not self.birds: 
            return True
        max_fitness = 0
        for genome in self.genomes:
            if genome.fitness > max_fitness:
                max_fitness = genome.fitness
        return max_fitness > FITNESS_THRESHOLD
    
    # no initial screen
    def initial_screen(self):
        return
