from constants import *
import pygame
import os

class Bird:
    def __init__(self):
        self.bird_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bird.png")).convert_alpha(), (BIRD_SIZE, BIRD_SIZE))
        self.x = BIRD_SIZE
        self.y = SCREEN_HEIGHT // 2 - BIRD_SIZE // 2
        self.vel = 0

    def move(self):
        self.vel += GRAVITY
        self.y += self.vel

    def flap(self):
        self.vel = -10

    def draw(self, screen):
        screen.blit(self.bird_img, (self.x, self.y))