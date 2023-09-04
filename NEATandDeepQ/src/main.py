import os
import neat
import pickle
import pygame
from constants import *
from deepQ import DeepQNetwork
from game_runners.run_neat_training import RunNeatAgents
from game_runners.run_default_agent import RunDefaultAgent
from game_runners.run_trained_neat_agent import RunBestNEATAgent
from game_runners.run_q_agent_training import RunDeepQAgent
from game_runners.run_trained_q_agent import RunTrainedQAgent


#-------------------------------------------------------------------------------
# main helper methods for neat training
#-------------------------------------------------------------------------------

def eval_genomes(genomes, config):
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    game = RunNeatAgents(genomes, config, screen)
    game.run()

def run_neat(config):
    pygame.init()
    #p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-0')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))
    winner = p.run(eval_genomes, 50)
    with open("best.pickle", "wb") as f:
        pickle.dump(winner, f)
    pygame.quit()

#-------------------------------------------------------------------------------
# main helper methods for best agent
#-------------------------------------------------------------------------------

def get_config():
    local_dir = os.getcwd()
    config_path = os.path.join(local_dir, "config.txt")
    return neat.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

def load_best_neat_agent():
    config = get_config()
    with open("best.pickle", "rb") as f:
        genome = pickle.load(f)
    return neat.nn.FeedForwardNetwork.create(genome, config)


#-------------------------------------------------------------------------------
# main helper methods for q learning agent
#-------------------------------------------------------------------------------

def run_q_learning(screen):
    # Initialize the agent
    state_size = 4
    action_size = 2
    network = DeepQNetwork(state_size, action_size)
    # Training loop
    max_score = 0
    for episode in range(Q_TRAINING_EPISODES):
        # create game environment
        game = RunDeepQAgent(network, screen)
        # run q-learning agent
        score = game.run()
        if score > max_score:
            max_score = score 
        # log training progress
        if episode % 100 == 0:
            print("Training episodes complete: ", episode, " max training score: ", max_score)
            max_score = 0
            game = RunTrainedQAgent(network, screen)
            score = game.run()
            print("Training episodes complete: ", episode, " test score: ", max_score)

    # store trained q_learning agent parameters
    network.save_state()

def load_trained_q_agent():
    # Initialize the agent
    state_size = 4
    action_size = 2
    agent = DeepQNetwork(state_size, action_size)
    agent.load_state()
    return agent

#-------------------------------------------------------------------------------
# main method
#-------------------------------------------------------------------------------

if __name__ == "__main__":
    # default | neat | best_neat | qlearn | best_qlearn
    game_type = "qlearn"
    pygame.init()
    if game_type == "default":
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        game = RunDefaultAgent(screen)
        game.run()
    elif game_type == "neat":
        config = get_config()
        run_neat(config)
    elif game_type == "best_neat":
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        network = load_best_neat_agent()
        game = RunBestNEATAgent(network, screen)
        game.run()
    elif game_type == "qlearn":
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        run_q_learning(screen)
    elif game_type == "best_qlearn":
        screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        network = load_trained_q_agent()
        game = RunTrainedQAgent(network, screen)
        game.run()
    pygame.quit()