from constants import *

class Bird:
    def __init__(self):
        self.vel = 0
        self.x = BIRD_SIZE
        self.y = SCREEN_HEIGHT // 2 - 25
        self.tick_count = 0

    def move_bird(self):
        self.vel += GRAVITY
        self.y += self.vel

    def flap(self):
        self.vel = -10.5

