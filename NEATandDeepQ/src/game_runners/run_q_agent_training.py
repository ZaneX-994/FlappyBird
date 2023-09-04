import pygame
from constants import *
from game_runners.run_default_agent import RunDefaultAgent

class RunDeepQAgent(RunDefaultAgent):
    def __init__(self, agent, screen):
        super().__init__(screen)
        # agent network (exists across multiple RunDeepQAgent instances)
        self.agent = agent
        # initialise state variables
        self.state = None
        self.action = None
        self.reward = None
        self.next_state = None
        self.done = False

    #-------------------------------------------------------------------------------
    # overriding run methods
    #-------------------------------------------------------------------------------
    def process_game_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit_game()
        # get game state
        self.state = self.get_state()
        # choose an action (run state through q network)
        self.action = self.agent.act(self.state)
        # perform action
        if self.action:
            self.birds[0].flap()

    def move_game_entities(self):
        # move game entities
        for bird in self.birds:
            bird.move()
        self.farthest_pipe.move()
        if self.nearest_pipe.move():
            self.switch_pipes()
        # get next state based on previous action
        self.next_state = self.get_state()

    def check_collisions(self):
        # get reward based on previous action
        if super().check_collisions():
            self.done = True
            self.reward = -100
        elif self.birds[0].y > self.nearest_pipe.y and self.birds[0].y < self.nearest_pipe.y + GAP_SIZE:
            self.reward = 0
        else:
            self.reward = -10
        # call q network to remember experience 
        self.agent.remember(self.state, self.action, self.reward, self.next_state, self.done)
        # Perform the replay step
        self.agent.replay(Q_BATCH_SIZE)
        # end game if collision occurred or fitness threshold exceeded
        return self.done or self.score > FITNESS_THRESHOLD
    
    # no initial screen
    def initial_screen(self):
        return

    #-------------------------------------------------------------------------------
    # helper methods for deep-q network
    #-------------------------------------------------------------------------------

    def get_state(self):
        bird = self.birds[0]
        return [bird.y, bird.vel, abs(bird.y - self.nearest_pipe.y), abs(bird.y - (self.nearest_pipe.y + GAP_SIZE))]