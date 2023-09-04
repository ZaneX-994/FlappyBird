import pygame
from constants import *
from bird import Bird
import random
import neat
import pickle
import os
import torch
from convolution import Conv



SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pipe_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha(), (PIPE_WIDTH, PIPE_HEIGHT))
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
bird_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bird1.png")).convert_alpha(), (BIRD_SIZE, BIRD_SIZE))
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())
screenshots = torch.empty(0,3,600,800)
frames = []
output = []
outputs = torch.empty(0,3)

class Game:
    def __init__(self, screen):
        self.screen = screen
        self.clock = pygame.time.Clock()
        self.bird = Bird()
        self.pipe_height1 = random.randint(100, 400)
        self.pipe_x1 = SCREEN_WIDTH
        self.pipe_height2 = random.randint(100, 400)
        # manually found starting value for x2 for pipes to be equidistant
        self.pipe_x2 = 1.5 * SCREEN_WIDTH + 40
        self.score = 0

    def move_pipes(self):
        self.pipe_x1 -= 5
        self.pipe_x2 -= 5
        if self.pipe_x1 < -PIPE_WIDTH:
            self.pipe_x1 = SCREEN_WIDTH
            self.pipe_height1 = random.randint(100, SCREEN_HEIGHT - GAP_SIZE)
            self.score += 1
            return True
        elif self.pipe_x2 < -PIPE_WIDTH:
            self.pipe_x2 = SCREEN_WIDTH
            self.pipe_height2 = random.randint(100, SCREEN_HEIGHT - GAP_SIZE)
            self.score += 1
            return True
        return False


    def check_collisions(self, bird):
        collision_occurred = False
        if bird.y > SCREEN_HEIGHT or bird.y < -BIRD_SIZE:
            collision_occurred = True
        if bird.x + BIRD_SIZE > self.pipe_x1 and bird.x < self.pipe_x1 + PIPE_WIDTH:
            if bird.y < self.pipe_height1 or bird.y + BIRD_SIZE > self.pipe_height1 + GAP_SIZE:
                collision_occurred = True
        elif bird.x + BIRD_SIZE > self.pipe_x2 and bird.x < self.pipe_x2 + PIPE_WIDTH:
            if bird.y < self.pipe_height2 or bird.y + BIRD_SIZE > self.pipe_height2 + GAP_SIZE:
                collision_occurred = True
        return collision_occurred

    def process_game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.bird.flap()
        return False

    def draw_game(self, birds, conv):
        global SCREEN
        global screenshots
        global frames
        SCREEN.blit(bg_img, (0,0))
        for bird in birds:
            #pygame.draw.rect(self.screen, (255, 255, 255,10), (bird.x, bird.y, BIRD_SIZE, BIRD_SIZE))
            SCREEN.blit(bird_img, (bird.x,bird.y))
        #pygame.draw.rect(self.screen, (255, 255, 255), (self.pipe_x1, 0, PIPE_WIDTH, self.pipe_height1))
        #pygame.draw.rect(self.screen, (255, 255, 255), (self.pipe_x1, self.pipe_height1 + GAP_SIZE, PIPE_WIDTH, SCREEN_HEIGHT - self.pipe_height1 - GAP_SIZE))
        SCREEN.blit(pipe_img, (self.pipe_x1, self.pipe_height1 + GAP_SIZE))
        SCREEN.blit(pygame.transform.flip(pipe_img, False, True), (self.pipe_x1, self.pipe_height1 - PIPE_HEIGHT))
        
        #pygame.draw.rect(self.screen, (255, 255, 255), (self.pipe_x2, 0, PIPE_WIDTH, self.pipe_height2))
        #pygame.draw.rect(self.screen, (255, 255, 255), (self.pipe_x2, self.pipe_height2 + GAP_SIZE, PIPE_WIDTH, SCREEN_HEIGHT - self.pipe_height2 - GAP_SIZE))
        SCREEN.blit(pipe_img, (self.pipe_x2, self.pipe_height2 + GAP_SIZE))
        SCREEN.blit(pygame.transform.flip(pipe_img, False, True), (self.pipe_x2, self.pipe_height2 - PIPE_HEIGHT))
        
        
        tensor = torch.from_numpy(pygame.surfarray.array3d(SCREEN)).permute(2,0,1).view(1, 3,600,800)
        tensor = tensor / 255
        print(conv(tensor))
        #frames.append(tensor)

        pygame.display.update()

    def run(self):
        global outputs
        global screenshots
        global frames
        global output
        local_dir = os.path.dirname(__file__)
        file_path = os.path.join(local_dir, "conv.pth")
        conv = Conv()
        conv.load_state_dict(torch.load(file_path))



        # initial screen
        if VISUALISE: 
            #tensor = torch.tensor([self.bird.y / SCREEN_HEIGHT, self.pipe_height1 / SCREEN_HEIGHT, (self.pipe_x1 - self.bird.x) / SCREEN_WIDTH])
            #output.append(tensor)
            self.draw_game([self.bird], conv)
        intial_running = True
        while intial_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit(0)
                if event.type == pygame.KEYDOWN:
                    intial_running = False
        # Game loop
        while True:
            # if quit signal returned, end game 
            if self.process_game_events(): break
            # move game entities
            self.bird.move_bird()
            self.move_pipes()
            # if collision occurred, end game
            if self.check_collisions(self.bird): break
            # Update display
            if self.pipe_x1 < self.pipe_x2 and self.bird.x < self.pipe_x1:
                pipe_height = self.pipe_height1
                pipe_distance = self.pipe_x1 - self.bird.x
            elif self.pipe_x2 < self.pipe_x1 and self.bird.x > self.pipe_x2:
                pipe_height = self.pipe_height1
                pipe_distance = self.pipe_x1 - self.bird.x
            else:
                pipe_height = self.pipe_height2
                pipe_distance = self.pipe_x2 - self.bird.x
            #tensor = torch.tensor([self.bird.y / SCREEN_HEIGHT, pipe_height / SCREEN_HEIGHT, pipe_distance / SCREEN_WIDTH])
            #output.append(tensor)
            torch.save(screenshots, 'outputs.pt')
            if VISUALISE: self.draw_game([self.bird], conv)
            pygame.display.update()
            self.clock.tick(MAX_FRAME_RATE)
        print(self.score)
        #screenshots = torch.stack(frames)
        #print(screenshots)
        #outputs = torch.stack(output)
        #print(outputs.size())
        #torch.save(screenshots, 'tensors.pt')
        #torch.save(outputs, 'outputs.pt')

    def run_NEAT_training(self, genome_tuples, config):
        networks = []
        birds = []
        genomes = []
        for _, genome in genome_tuples:
            networks.append(neat.nn.FeedForwardNetwork.create(genome, config))
            birds.append(Bird())
            genome.fitness = 0
            genomes.append(genome)
        # game loop
        pipe = 1
        while True:
            # compute NEAT AGENT decisions
            if pipe == 1:
                pipe_height = self.pipe_height1
            else:
                pipe_height = self.pipe_height2
            for i, bird in enumerate(birds):
                output = networks[i].activate((bird.vel, bird.y, abs(bird.y - pipe_height), abs(bird.y - (pipe_height + GAP_SIZE))))
                if output[0] > 0.5:
                    birds[i].flap()
                # increment fitness of all birds that haven't died
                genomes[i].fitness += 0.1
            # move game entities
            for bird in birds:
                bird.move_bird()
            if self.move_pipes():
                if pipe == 1:
                    pipe = 2
                else:
                    pipe = 1
                for genome in genomes:
                    genome.fitness += 5
            # if collision occurred, remove bird, if no birds left, end game
            for i, bird in enumerate(birds):
                if self.check_collisions(bird):
                    genomes[i].fitness -= 1
                    networks.pop(i)
                    birds.pop(i)
                    genomes.pop(i)
            if not birds: break
            # terminate training if fitness threshold reached
            if self.score > FITNESS_THRESHOLD: break
            # Update display
            if VISUALISE: self.draw_game(birds)
            pygame.display.update()
            self.clock.tick(MAX_FRAME_RATE)

    def run_best_agent(self, network):
        # initial screen
        if VISUALISE:
            self.draw_game([self.bird])
        intial_running = True
        while intial_running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit(0)
                if event.type == pygame.KEYDOWN:
                    intial_running = False
        # Game loop
        while True:
            if self.pipe_x1 < self.pipe_x2 and self.bird.x < self.pipe_x1:
                pipe_height = self.pipe_height1
            elif self.pipe_x2 < self.pipe_x1 and self.bird.x > self.pipe_x2:
                pipe_height = self.pipe_height1
            else:
                pipe_height = self.pipe_height2
            # if quit signal returned, end game 
            if self.process_game_events(): break
            # compute agent decision
            output = network.activate((self.bird.vel, self.bird.y, abs(self.bird.y - pipe_height), abs(self.bird.y - (pipe_height + GAP_SIZE))))
            if output[0] > 0.5:
                self.bird.flap()
            # move game entities
            self.bird.move_bird()
            self.move_pipes()
            # if collision occurred, end game
            if self.check_collisions(self.bird): break
            # Update display
            if VISUALISE: self.draw_game([self.bird])
            pygame.display.update()
            self.clock.tick(MAX_FRAME_RATE)
        print(self.score)

def load_network():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)
    with open("best.pickle", "rb") as f:
        genome = pickle.load(f)
    return neat.nn.FeedForwardNetwork.create(genome, config)

if __name__ == "__main__":
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    game_instance = Game(screen)
    network = load_network()
    game_instance.run()
    """ for input_node in network.input_nodes:
        print("INPUT NODE: ", input_node)
    for output_node in network.output_nodes:
        print("OUTPUT NODE: ", output_node)
    for eval_node in network.eval_nodes:
        print("eval_node: ", eval_node) """

    game_instance.run_best_agent(network)
    pygame.quit()